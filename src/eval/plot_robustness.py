"""Plot the noise-robustness curves: baseline vs generative-augmented.

Reads multiple `results/robustness_*.csv` files, groups by (arm-label, snr), and
plots mean+/-range across seeds.

Run:
  python -m src.eval.plot_robustness \
      --inputs results/robustness_none_weighted_ce.csv:baseline \
               results/robustness_generative_weighted_ce.csv:generative \
      --out report/figures/robustness.png
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def load(path: Path, label: str) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            r["arm"] = label
            rows.append(r)
    return rows


def snr_sort_key(s: str):
    if s.lower() == "clean":
        return float("inf")
    return float(s)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="path.csv:label pairs")
    p.add_argument("--out", type=Path, default=REPO / "report" / "figures" / "robustness.png")
    args = p.parse_args()

    all_rows: list[dict] = []
    for spec in args.inputs:
        path_str, label = spec.split(":")
        all_rows.extend(load(Path(path_str), label))

    snrs = sorted({r["snr_db"] for r in all_rows}, key=snr_sort_key)
    labels = list(dict.fromkeys(r["arm"] for r in all_rows))

    def gather(metric: str) -> dict[tuple[str, str], list[float]]:
        out: dict[tuple[str, str], list[float]] = defaultdict(list)
        for r in all_rows:
            try:
                out[(r["arm"], r["snr_db"])].append(float(r[metric]))
            except (KeyError, ValueError):
                pass
        return out

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for metric_name, ax, ylabel in [
        ("macro_f1", axes[0], "Test macro-F1"),
        ("burping_recall", axes[1], "Burping recall"),
    ]:
        data = gather(metric_name)
        x_positions = np.arange(len(snrs))
        for label in labels:
            means = []
            mins = []
            maxs = []
            for snr in snrs:
                vals = data.get((label, snr), [])
                means.append(np.mean(vals) if vals else np.nan)
                mins.append(np.min(vals) if vals else np.nan)
                maxs.append(np.max(vals) if vals else np.nan)
            means = np.array(means)
            yerr = [means - np.array(mins), np.array(maxs) - means]
            ax.errorbar(x_positions, means, yerr=yerr, marker="o", capsize=3, label=label)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["clean" if s.lower() == "clean" else f"{s} dB" for s in snrs])
        ax.set_xlabel("SNR (additive Gaussian noise in mel space)")
        ax.set_ylabel(ylabel + " (mean over 3 seeds, range bars)")
        ax.set_ylim(-0.05, 1.05 if metric_name == "burping_recall" else 0.5)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Noise robustness: baseline vs generative augmentation", fontsize=11)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=130)
    plt.close(fig)
    print(f"[robustness-plot] wrote {args.out}")


if __name__ == "__main__":
    main()
