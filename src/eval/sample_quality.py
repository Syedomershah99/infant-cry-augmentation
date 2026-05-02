"""Sample-quality checks for synthetic spectrograms.

Two cheap proxies that don't require a vocoder or external embedding:
  1. Class consistency: train a held-out probe classifier on REAL train data,
     then evaluate it on synthetic samples. If synthetic samples carry the
     intended class signal, the probe should classify them above chance.
  2. Mean / std / range stats per class to spot mode collapse or distribution
     shift between real and synthetic.

Run:
  python -m src.eval.sample_quality \
      --probe_ckpt results/baseline_seed0/best.pt \
      --synth_manifest data/manifests/synth_train.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.audio.dataset import CryDataset, IDX_TO_LABEL, LABELS, LABEL_TO_IDX
from src.audio.features import FeatureConfig
from src.eval.metrics import per_class_prf
from src.models.classifier import CryCNN

REPO = Path(__file__).resolve().parents[2]


def load_classifier(ckpt_path: Path, device: str) -> CryCNN:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = CryCNN(
        num_classes=len(LABELS),
        base_channels=cfg.get("base_channels", 32),
        dropout=cfg.get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def class_consistency(model: CryCNN, synth_manifest: Path, device: str) -> dict:
    """Run probe classifier on synth samples, report per-class precision/recall/F1."""
    feat_cfg = FeatureConfig()
    ds = CryDataset(
        manifest=synth_manifest,
        repo_root=REPO,
        feature_cfg=feat_cfg,
        extra_manifests=[synth_manifest],  # forces is_synthetic=True path; manifest is itself synthetic
    )
    # The manifest above has is_synthetic=False rows by default; instead, build the dataset
    # directly with an empty primary and just pass synth as extra:
    ds = CryDataset(
        manifest=synth_manifest,  # any manifest pointer works; we only care about extras for synth flag
        repo_root=REPO,
        feature_cfg=feat_cfg,
    )
    # Items are loaded as REAL (since dataset parses synth_train.csv as primary).
    # That's wrong for .pt sources. Instead, construct items manually.

    rows: list[dict] = []
    with synth_manifest.open() as f:
        for row in csv.DictReader(f):
            if row["label"] in LABEL_TO_IDX:
                rows.append(row)

    y_true: list[int] = []
    y_pred: list[int] = []
    stats_per_class: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    with torch.no_grad():
        for r in rows:
            spec = torch.load(REPO / r["filepath"], map_location=device, weights_only=False)
            if spec.dim() == 2:
                spec = spec.unsqueeze(0)
            spec = spec.unsqueeze(0).to(device).float()
            logits = model(spec)
            pred = int(logits.argmax(dim=-1).item())
            true = LABEL_TO_IDX[r["label"]]
            y_true.append(true)
            y_pred.append(pred)
            stats_per_class[r["label"]].append(
                (spec.mean().item(), spec.std().item(), spec.min().item(), spec.max().item())
            )

    pc = per_class_prf(np.array(y_true), np.array(y_pred))
    summary_stats = {
        cls: {
            "n": len(stats),
            "mean": float(np.mean([s[0] for s in stats])),
            "std": float(np.mean([s[1] for s in stats])),
            "min": float(np.mean([s[2] for s in stats])),
            "max": float(np.mean([s[3] for s in stats])),
        }
        for cls, stats in stats_per_class.items()
    }
    return {"per_class_metrics_on_synth": pc, "spec_stats": summary_stats, "n_total": len(y_true)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--probe_ckpt", type=Path, required=True)
    p.add_argument("--synth_manifest", type=Path, default=REPO / "data" / "manifests" / "synth_train.csv")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    model = load_classifier(args.probe_ckpt, args.device)
    out = class_consistency(model, args.synth_manifest, args.device)

    print(f"[probe] tested {out['n_total']} synthetic samples")
    print("[probe] per-class metrics (on synthetic; intended labels treated as ground truth):")
    for label in LABELS:
        m = out["per_class_metrics_on_synth"].get(label)
        if not m or m["support"] == 0:
            continue
        print(
            f"  {label:11s}  n={m['support']:3d}  recall={m['recall']:.3f}  "
            f"precision={m['precision']:.3f}  f1={m['f1']:.3f}"
        )
    print("[probe] spec value distribution (per intended class):")
    for label, st in out["spec_stats"].items():
        print(f"  {label:11s}  n={st['n']:3d}  mean={st['mean']:+.2f}  std={st['std']:.2f}  range=[{st['min']:+.2f}, {st['max']:+.2f}]")


if __name__ == "__main__":
    main()
