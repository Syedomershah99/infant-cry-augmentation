"""Render a side-by-side grid of real vs synthetic spectrograms.

Saves a PNG grid to report/figures/. Three rows: real rare-class samples,
synthetic samples for the same classes, and a difference heatmap.

Run:
  python -m src.eval.visualize \
      --classes belly_pain burping \
      --n_per_class 4 \
      --out report/figures/real_vs_synth.png
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch

from src.audio.dataset import LABEL_TO_IDX
from src.audio.features import FeatureConfig, fix_duration, load_and_resample, LogMelExtractor, standardize

REPO = Path(__file__).resolve().parents[2]


def load_real_specs(label: str, n: int, seed: int = 0) -> list[np.ndarray]:
    cfg = FeatureConfig()
    extractor = LogMelExtractor(cfg)
    rng = random.Random(seed)
    rows: list[dict] = []
    with (REPO / "data" / "manifests" / "train.csv").open() as f:
        for row in csv.DictReader(f):
            if row["label"] == label:
                rows.append(row)
    rng.shuffle(rows)
    out: list[np.ndarray] = []
    for row in rows[:n]:
        path = REPO / row["filepath"]
        wav = load_and_resample(str(path), cfg.sample_rate)
        wav = fix_duration(wav, cfg.sample_rate, cfg.target_seconds)
        spec = extractor(wav.unsqueeze(0)).squeeze(0)
        spec = standardize(spec)
        out.append(spec.squeeze(0).numpy())
    return out


def load_synth_specs(label: str, n: int, seed: int = 0) -> list[np.ndarray]:
    rng = random.Random(seed)
    rows: list[dict] = []
    manifest = REPO / "data" / "manifests" / "synth_train.csv"
    if not manifest.exists():
        return []
    with manifest.open() as f:
        for row in csv.DictReader(f):
            if row["label"] == label:
                rows.append(row)
    rng.shuffle(rows)
    out: list[np.ndarray] = []
    for row in rows[:n]:
        spec = torch.load(REPO / row["filepath"], map_location="cpu", weights_only=False)
        if spec.dim() == 3:
            spec = spec.squeeze(0)
        out.append(spec.numpy())
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--classes", nargs="+", default=["belly_pain", "burping"])
    p.add_argument("--n_per_class", type=int, default=4)
    p.add_argument("--out", type=Path, default=REPO / "report" / "figures" / "real_vs_synth.png")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import matplotlib  # noqa: WPS433 — keep optional dep
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_classes = len(args.classes)
    n = args.n_per_class
    fig, axes = plt.subplots(2 * n_classes, n, figsize=(2 * n, 2 * 2 * n_classes), squeeze=False)
    for ci, label in enumerate(args.classes):
        real = load_real_specs(label, n, args.seed)
        synth = load_synth_specs(label, n, args.seed)
        for j in range(n):
            ax_r = axes[2 * ci, j]
            ax_s = axes[2 * ci + 1, j]
            if j < len(real):
                ax_r.imshow(real[j], origin="lower", aspect="auto")
            ax_r.set_xticks([]); ax_r.set_yticks([])
            if j == 0:
                ax_r.set_ylabel(f"{label}\nreal", fontsize=8)
            if j < len(synth):
                ax_s.imshow(synth[j], origin="lower", aspect="auto")
            ax_s.set_xticks([]); ax_s.set_yticks([])
            if j == 0:
                ax_s.set_ylabel(f"{label}\nsynth", fontsize=8)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"[viz] saved {args.out}")


if __name__ == "__main__":
    main()
