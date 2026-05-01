"""Train the baseline CNN classifier on log-mel spectrograms.

Run:
  python -m src.training.train_classifier --config configs/baseline_cnn.yaml

Honors three augmentation arms via config:
  aug_type: none | classical | generative | classical+generative
  synth_manifest: optional path to data/manifests/synth_train.csv (used when
                  aug_type contains "generative")

Always reports per-class metrics. Saves a JSON result blob per epoch and a
final checkpoint.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.audio.augment import ClassicalAug, SpecAugConfig
from src.audio.dataset import CryDataset, LABELS, class_weights, read_manifest
from src.audio.features import FeatureConfig
from src.eval.fairness import safety_summary
from src.eval.metrics import evaluate
from src.models.classifier import CryCNN, count_params

REPO = Path(__file__).resolve().parents[2]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_loaders(cfg: dict, seed: int) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    feat_cfg = FeatureConfig(**cfg.get("features", {}))
    aug_type = cfg.get("aug_type", "none")
    use_classical = "classical" in aug_type
    use_generative = "generative" in aug_type

    classical = ClassicalAug(SpecAugConfig(**cfg.get("specaug", {})), seed=seed) if use_classical else None
    extra_manifests = []
    if use_generative:
        synth = cfg.get("synth_manifest")
        if synth and Path(synth).exists():
            extra_manifests.append(synth)
        elif synth:
            print(f"[warn] synth_manifest {synth} not found; running without it")

    train_ds = CryDataset(
        manifest=REPO / cfg["manifests"]["train"],
        repo_root=REPO,
        feature_cfg=feat_cfg,
        transform=classical,
        extra_manifests=extra_manifests,
    )
    val_ds = CryDataset(
        manifest=REPO / cfg["manifests"]["val"],
        repo_root=REPO,
        feature_cfg=feat_cfg,
    )
    test_ds = CryDataset(
        manifest=REPO / cfg["manifests"]["test"],
        repo_root=REPO,
        feature_cfg=feat_cfg,
    )

    bs = cfg.get("batch_size", 32)
    nw = cfg.get("num_workers", 0)

    sampler = None
    if cfg.get("balanced_sampler", False):
        # Weighted sampler so every batch contains rare classes proportionally.
        items = train_ds.items
        counts = np.array([sum(1 for it in items if it.label == i) for i in range(len(LABELS))])
        weights_per_class = 1.0 / np.clip(counts, 1, None)
        weights = np.array([weights_per_class[it.label] for it in items])
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.float64),
            num_samples=len(items),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=nw,
        drop_last=False,
    )
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)

    weights = class_weights(train_ds.items, scheme=cfg.get("class_weight_scheme", "inverse_sqrt"))
    return train_loader, val_loader, test_loader, weights


def train(cfg: dict) -> dict:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    device = select_device(cfg.get("device", "auto"))

    train_loader, val_loader, test_loader, class_w = make_loaders(cfg, seed)
    print(f"[data] train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  test={len(test_loader.dataset)}")
    print(f"[data] class_weights={class_w.tolist()}")

    model = CryCNN(
        num_classes=len(LABELS),
        base_channels=cfg.get("base_channels", 32),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)
    print(f"[model] CryCNN  params={count_params(model):,}")

    use_class_weight = cfg.get("use_class_weighted_loss", True)
    criterion = nn.CrossEntropyLoss(
        weight=class_w.to(device) if use_class_weight else None,
        label_smoothing=cfg.get("label_smoothing", 0.0),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.get("epochs", 30)
    )

    out_dir = REPO / cfg.get("out_dir", f"results/baseline_seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val_macro_f1 = -1.0
    best_path = out_dir / "best.pt"

    epochs = cfg.get("epochs", 30)
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_seen = 0
        for batch in train_loader:
            spec = batch["spec"].to(device)
            label = batch["label"].to(device)
            logits = model(spec)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += loss.item() * spec.size(0)
            n_seen += spec.size(0)
        scheduler.step()
        train_loss = running_loss / max(1, n_seen)

        val_metrics = evaluate(model, val_loader, device=device)
        elapsed = time.time() - t0
        print(
            f"[epoch {epoch:3d}] loss={train_loss:.4f}  "
            f"val_macroF1={val_metrics['macro_f1']:.4f}  "
            f"val_belly_pain_recall={val_metrics['per_class']['belly_pain']['recall']:.3f}  "
            f"val_ece={val_metrics['ece']:.3f}  "
            f"({elapsed:.1f}s)"
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "val": val_metrics})
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, best_path)

    # Final test eval with best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader, device=device)
    print(f"[test] macroF1={test_metrics['macro_f1']:.4f}  acc={test_metrics['accuracy']:.4f}  ece={test_metrics['ece']:.3f}")
    print(f"[test] safety={safety_summary(test_metrics)}")

    result = {
        "config": cfg,
        "best_val_macro_f1": best_val_macro_f1,
        "test": test_metrics,
        "safety": safety_summary(test_metrics),
        "history": history,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.out is not None:
        cfg["out_dir"] = str(args.out)
    train(cfg)


if __name__ == "__main__":
    main()
