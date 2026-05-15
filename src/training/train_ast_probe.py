"""Linear probe over cached AST embeddings — the backbone-comparison harness.

Loads frozen 768-d AST embeddings cached by ``src.models.ast_features``, attaches a
small linear classifier (768->5), and trains under the same augmentation arms /
optimizer recipe / seed structure as the main matrix. The DDPM synthetic samples
are mel spectrograms, not raw audio, so AST cannot embed them directly; for the
``generative`` arms we substitute "synthetic features" by sampling features from
the real class-conditional distribution under a small Gaussian perturbation in
the embedding space, which is a reasonable proxy when the goal is to test
whether the backbone changes the qualitative conclusions of the main matrix.

Run:
  python -m src.training.train_ast_probe --config configs/ast_baseline.yaml
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

from src.audio.dataset import LABELS, LABEL_TO_IDX, class_weights
from src.eval.metrics import per_class_prf, expected_calibration_error
from src.training.loss import make_loss

REPO = Path(__file__).resolve().parents[2]
LABEL_LIST = LABELS


def md5_path(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_emb(features_root: Path, filepath_relative: str, is_synth: bool, repo_root: Path) -> torch.Tensor | None:
    """Return a cached 768-d embedding or None if it isn't available."""
    if is_synth:
        # Synthetic features are derived per-cell (see SynthFeatureBank below);
        # this function is only used for real rows.
        return None
    abs_path = repo_root / filepath_relative
    cache_path = features_root / f"{md5_path(abs_path)}.pt"
    if not cache_path.exists():
        return None
    return torch.load(cache_path, map_location="cpu", weights_only=False).float()


class CryEmbeddingDataset(Dataset):
    """Items: (emb_768, label_idx, is_synth)."""

    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        it = self.items[idx]
        return {
            "emb": it["emb"].float(),
            "label": it["label"],
            "is_synthetic": int(it["is_synth"]),
        }


def build_real_items(manifest: Path, features_root: Path, repo_root: Path) -> list[dict]:
    items: list[dict] = []
    missing = 0
    with manifest.open() as f:
        for row in csv.DictReader(f):
            label = row["label"]
            if label not in LABEL_TO_IDX:
                continue
            emb = load_emb(features_root, row["filepath"], is_synth=False, repo_root=repo_root)
            if emb is None:
                missing += 1
                continue
            items.append({"emb": emb, "label": LABEL_TO_IDX[label], "is_synth": False})
    if missing:
        print(f"[ast-probe] WARN: {missing} embeddings missing in {manifest.name}")
    return items


def build_synth_proxy_items(
    train_real_items: list[dict],
    rare_classes: tuple[str, ...] = ("belly_pain", "burping"),
    ratio: int = 3,
    sigma: float = 0.05,
    seed: int = 0,
) -> list[dict]:
    """Approximate the DDPM-augmentation pipeline in AST embedding space.

    Since AST consumes waveforms (not mel-spectrograms) we cannot directly embed
    the DDPM-sampled mel patches. To still produce a defensible backbone
    comparison, we synthesize per-class proxy features by sampling around the
    real-class embedding mean with small Gaussian noise. This is conservative:
    it represents the *easiest* possible per-class synth signal, so any failure
    of the AST linear probe to recover the rare class under this proxy is a
    bound on what real synth features would do.
    """
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[np.ndarray]] = {l: [] for l in LABEL_LIST}
    for it in train_real_items:
        by_class[LABEL_LIST[it["label"]]].append(it["emb"].numpy())
    out: list[dict] = []
    for cls in rare_classes:
        cls_idx = LABEL_TO_IDX[cls]
        real = by_class[cls]
        if not real:
            continue
        mu = np.mean(real, axis=0)
        n_synth = int(len(real) * ratio)
        for _ in range(n_synth):
            noise = rng.normal(0, sigma, size=mu.shape).astype(np.float32)
            out.append({"emb": torch.from_numpy(mu + noise), "label": cls_idx, "is_synth": True})
    return out


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int = 768, num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: LinearProbe, loader: DataLoader, device: str) -> dict:
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            emb = batch["emb"].to(device)
            logits = model(emb)
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu() if torch.is_tensor(batch["label"]) else torch.tensor(batch["label"]))
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = probs.argmax(axis=1)
    pc = per_class_prf(labels, preds)
    return {
        "macro_f1": float(np.mean([pc[l]["f1"] for l in LABEL_LIST])),
        "accuracy": float(np.mean(labels == preds)),
        "ece": expected_calibration_error(probs, labels),
        "per_class": pc,
    }


def train(cfg: dict) -> dict:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    device = cfg.get("device", "cpu")
    features_root = REPO / cfg.get("features_root", "data/ast_features")

    train_real = build_real_items(REPO / cfg["manifests"]["train"], features_root, REPO)
    val_items = build_real_items(REPO / cfg["manifests"]["val"], features_root, REPO)
    test_items = build_real_items(REPO / cfg["manifests"]["test"], features_root, REPO)

    aug_type = cfg.get("aug_type", "none")
    train_items = list(train_real)
    if "generative" in aug_type:
        synth = build_synth_proxy_items(
            train_real,
            ratio=int(cfg.get("synth_ratio", 3)),
            sigma=float(cfg.get("synth_sigma", 0.05)),
            seed=seed,
        )
        train_items.extend(synth)
        print(f"[ast-probe] mixed in {len(synth)} synth-proxy features")

    print(f"[ast-probe] train={len(train_items)}  val={len(val_items)}  test={len(test_items)}")

    train_ds = CryEmbeddingDataset(train_items)
    val_ds = CryEmbeddingDataset(val_items)
    test_ds = CryEmbeddingDataset(test_items)
    bs = cfg.get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0)

    # Class weights are over the (real + synth) train pool, same convention as the main matrix
    items_for_w = [type("X", (), {"label": it["label"], "label_str": LABEL_LIST[it["label"]]})() for it in train_items]
    class_w = class_weights(items_for_w, scheme=cfg.get("class_weight_scheme", "inverse_sqrt"))
    criterion = make_loss(cfg, class_w.to(device))

    model = LinearProbe(in_dim=cfg.get("in_dim", 768), num_classes=len(LABEL_LIST), dropout=cfg.get("dropout", 0.1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3), weight_decay=cfg.get("weight_decay", 1e-4))
    epochs = cfg.get("epochs", 30)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    out_dir = REPO / cfg.get("out_dir", f"results/ast_probe_seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_macro_f1 = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n = 0
        for batch in train_loader:
            emb = batch["emb"].to(device)
            lbl = batch["label"].to(device) if torch.is_tensor(batch["label"]) else torch.as_tensor(batch["label"]).to(device)
            logits = model(emb)
            loss = criterion(logits, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * emb.size(0)
            n += emb.size(0)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "train_loss": running_loss / max(1, n), "val": val_metrics})

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    print(
        f"[ast-probe] test macroF1={test_metrics['macro_f1']:.4f}  acc={test_metrics['accuracy']:.4f}  "
        f"bp_rec={test_metrics['per_class']['belly_pain']['recall']:.3f}  "
        f"burp_rec={test_metrics['per_class']['burping']['recall']:.3f}  ece={test_metrics['ece']:.3f}"
    )
    result = {"config": cfg, "best_val_macro_f1": best_val_macro_f1, "test": test_metrics, "history": history}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.out is not None:
        cfg["out_dir"] = str(args.out)
    train(cfg)


if __name__ == "__main__":
    main()
