"""Plot the synth-to-real ratio sweep: macro-F1 and burping recall vs ratio.

Includes a ratio=0 anchor reused from the no-augmentation baseline matrix.

Run:
  python -m src.eval.plot_ratio --out report/figures/ratio_sweep.png
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def load(path: Path, ratio_override: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            if ratio_override is not None:
                r["synth_ratio"] = str(ratio_override)
            rows.append(r)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ratio_csv", type=Path, default=REPO / "results" / "matrix_ratio_sweep.csv")
    p.add_argument("--baseline_csv", type=Path, default=REPO / "results" / "matrix_part1.csv",
                   help="anchor ratio=0 point from a no-augmentation cell")
    p.add_argument("--out", type=Path, default=REPO / "report" / "figures" / "ratio_sweep.png")
    args = p.parse_args()

    rows: list[dict] = []
    # Anchor: none/weighted_ce rows reused as ratio=0 generative-equivalent
    with args.baseline_csv.open() as f:
        for r in csv.DictReader(f):
            if r.get("aug_type") == "none" and r.get("recipe") == "weighted_ce":
                r["synth_ratio"] = "0"
                rows.append(r)
    rows.extend(load(args.ratio_csv))

    by_ratio: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        try:
            ratio = int(r["synth_ratio"])
        except (KeyError, ValueError):
            continue
        for m in ("macro_f1", "accuracy", "ece", "belly_pain_recall", "burping_recall"):
            try:
                by_ratio[ratio][m].append(float(r[m]))
            except (KeyError, ValueError):
                pass

    ratios = sorted(by_ratio.keys())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric, ylabel in [
        (axes[0], "macro_f1", "Test macro-F1"),
        (axes[1], "burping_recall", "Burping recall"),
    ]:
        means = [np.mean(by_ratio[r][metric]) for r in ratios]
        mins = [np.min(by_ratio[r][metric]) for r in ratios]
        maxs = [np.max(by_ratio[r][metric]) for r in ratios]
        means = np.array(means)
        yerr = [means - np.array(mins), np.array(maxs) - means]
        ax.errorbar(ratios, means, yerr=yerr, marker="o", capsize=3, color="#d95f02")
        ax.set_xticks(ratios)
        ax.set_xticklabels([f"{r}×" for r in ratios])
        ax.set_xlabel("Synth-to-real ratio (rare classes only)")
        ax.set_ylabel(ylabel + " (mean over 3 seeds, range bars)")
        ax.set_ylim(-0.05, 1.05 if metric == "burping_recall" else 0.5)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Synth-to-real ratio sweep (generative + weighted_ce)", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"[ratio-plot] wrote {args.out}")


if __name__ == "__main__":
    main()
