"""Generate the final report figures from results/matrix*.csv.

Produces:
  report/figures/macro_f1_by_arm.png       per-arm bar chart (mean +/- range)
  report/figures/fnr_by_class.png          per-class FNR grouped by arm
  report/figures/ece_by_arm.png            ECE distribution per arm
  report/figures/confusion_baseline.png    confusion matrix from a baseline cell
  report/figures/confusion_focal.png       confusion matrix from a focal cell

Run:
  python -m src.eval.figures
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LABELS = ["belly_pain", "burping", "discomfort", "hungry", "tired"]


def load_matrix(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        if not p.exists():
            print(f"[figures] skipping missing {p}")
            continue
        with p.open() as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


def _aggregate_metric(rows: list[dict], metric: str) -> dict[tuple[str, str], list[float]]:
    """key = (aug_type, recipe), value = list of metric values across seeds."""
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        try:
            v = float(r[metric])
        except (KeyError, ValueError):
            continue
        out[(r["aug_type"], r.get("recipe", "weighted_ce"))].append(v)
    return out


def plot_macro_f1(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = _aggregate_metric(rows, "macro_f1")
    arms = sorted({k[0] for k in agg.keys()})
    recipes = sorted({k[1] for k in agg.keys()})
    width = 0.8 / max(1, len(recipes))
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, recipe in enumerate(recipes):
        means = [np.mean(agg.get((a, recipe), [np.nan])) for a in arms]
        ranges_lo = [np.min(agg.get((a, recipe), [np.nan])) for a in arms]
        ranges_hi = [np.max(agg.get((a, recipe), [np.nan])) for a in arms]
        yerr = [
            [m - lo for m, lo in zip(means, ranges_lo)],
            [hi - m for m, hi in zip(means, ranges_hi)],
        ]
        ax.bar(x + i * width, means, width, yerr=yerr, capsize=2, label=recipe)
    ax.set_xticks(x + width * (len(recipes) - 1) / 2)
    ax.set_xticklabels(arms, rotation=15, ha="right")
    ax.set_ylabel("Test macro-F1 (mean over seeds, range bars)")
    ax.set_title("Test macro-F1 by augmentation arm and optimizer recipe")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[figures] wrote {out}")


def plot_ece(rows: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = _aggregate_metric(rows, "ece")
    arms = sorted({k[0] for k in agg.keys()})
    recipes = sorted({k[1] for k in agg.keys()})
    width = 0.8 / max(1, len(recipes))
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, recipe in enumerate(recipes):
        means = [np.mean(agg.get((a, recipe), [np.nan])) for a in arms]
        ax.bar(x + i * width, means, width, label=recipe)
    ax.set_xticks(x + width * (len(recipes) - 1) / 2)
    ax.set_xticklabels(arms, rotation=15, ha="right")
    ax.set_ylabel("Expected Calibration Error (mean over seeds)")
    ax.set_title("Calibration (ECE) by arm and recipe (lower is better)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[figures] wrote {out}")


def plot_fnr_by_class(rows: list[dict], out: Path, recipe_filter: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if recipe_filter is None or r.get("recipe") == recipe_filter]
    arms = sorted({r["aug_type"] for r in rows})
    fnr_by_arm: dict[str, dict[str, list[float]]] = {a: {l: [] for l in LABELS} for a in arms}
    for r in rows:
        a = r["aug_type"]
        for l in LABELS:
            try:
                v = float(r[f"{l}_fnr"]) if f"{l}_fnr" in r else 1.0 - float(r[f"{l}_recall"])
            except (KeyError, ValueError):
                continue
            fnr_by_arm[a][l].append(v)
    width = 0.8 / max(1, len(LABELS))
    x = np.arange(len(arms))
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, l in enumerate(LABELS):
        means = [np.mean(fnr_by_arm[a][l]) if fnr_by_arm[a][l] else np.nan for a in arms]
        ax.bar(x + i * width, means, width, label=l)
    ax.set_xticks(x + width * (len(LABELS) - 1) / 2)
    ax.set_xticklabels(arms, rotation=15, ha="right")
    ax.set_ylabel("False-negative rate (mean over seeds)")
    title = "Per-class FNR by augmentation arm"
    if recipe_filter:
        title += f"  (recipe = {recipe_filter})"
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[figures] wrote {out}")


def plot_confusion(result_json: Path, out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not result_json.exists():
        print(f"[figures] skip confusion (missing) {result_json}")
        return
    with result_json.open() as f:
        res = json.load(f)
    cm = np.array(res["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=45, ha="right")
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[figures] wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--matrix",
        nargs="+",
        type=Path,
        default=[REPO / "results" / "matrix.csv", REPO / "results" / "matrix_part1.csv", REPO / "results" / "matrix_part2.csv"],
    )
    p.add_argument("--out", type=Path, default=REPO / "report" / "figures")
    args = p.parse_args()

    rows = load_matrix(args.matrix)
    if not rows:
        raise SystemExit("no matrix rows found; run experiments.run_matrix first")
    print(f"[figures] {len(rows)} rows from {len(args.matrix)} matrix files")

    plot_macro_f1(rows, args.out / "macro_f1_by_arm.png")
    plot_ece(rows, args.out / "ece_by_arm.png")
    plot_fnr_by_class(rows, args.out / "fnr_by_class.png")
    for recipe in {r.get("recipe") for r in rows} - {None}:
        plot_fnr_by_class(rows, args.out / f"fnr_by_class_{recipe}.png", recipe_filter=recipe)


if __name__ == "__main__":
    main()
