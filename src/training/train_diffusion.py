"""Train the class-conditional DDPM on log-mel spectrograms.

Run:
  python -m src.training.train_diffusion --config configs/cond_ddpm.yaml

Trains on the train split only. Uses the same FeatureConfig as the classifier
so synthetic samples are immediately consumable by the classifier without any
re-normalization.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.audio.dataset import CryDataset, LABELS
from src.audio.features import FeatureConfig
from src.models.diffusion import CondUNet, DDPM, DDPMConfig, count_params

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


def train(cfg: dict) -> dict:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    device = select_device(cfg.get("device", "auto"))

    feat_cfg = FeatureConfig(**cfg.get("features", {}))
    train_ds = CryDataset(
        manifest=REPO / cfg["manifests"]["train"],
        repo_root=REPO,
        feature_cfg=feat_cfg,
    )
    print(f"[data] train={len(train_ds)}")

    bs = cfg.get("batch_size", 32)
    nw = cfg.get("num_workers", 0)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, drop_last=False)

    unet = CondUNet(
        num_classes=len(LABELS),
        base_ch=cfg.get("base_ch", 64),
        ch_mults=tuple(cfg.get("ch_mults", [1, 2, 4])),
        dropout=cfg.get("dropout", 0.1),
    )
    ddpm = DDPM(unet, DDPMConfig(**cfg.get("ddpm", {}))).to(device)
    print(f"[model] CondUNet+DDPM params={count_params(ddpm):,}")

    optimizer = torch.optim.AdamW(
        ddpm.parameters(), lr=cfg.get("lr", 2e-4), weight_decay=cfg.get("weight_decay", 0.0)
    )

    out_dir = REPO / cfg.get("out_dir", f"results/ddpm_seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []

    epochs = cfg.get("epochs", 50)
    log_every = cfg.get("log_every", 10)
    best_path = out_dir / "best.pt"
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        ddpm.train()
        t0 = time.time()
        running = 0.0
        n = 0
        for batch in train_loader:
            x0 = batch["spec"].to(device)
            y = batch["label"].to(device)
            loss = ddpm.loss(x0, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * x0.size(0)
            n += x0.size(0)
        avg = running / max(1, n)
        elapsed = time.time() - t0
        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            print(f"[ddpm epoch {epoch:3d}] loss={avg:.4f}  ({elapsed:.1f}s)")
        history.append({"epoch": epoch, "loss": avg, "t": elapsed})
        if avg < best_loss:
            best_loss = avg
            torch.save({"model": ddpm.state_dict(), "cfg": cfg, "epoch": epoch}, best_path)

    result = {"config": cfg, "best_loss": best_loss, "history": history}
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
