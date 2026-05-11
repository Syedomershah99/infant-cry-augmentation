"""Bootstrap 95% confidence intervals from per-seed result.json files.

We have 3 seeds per cell in the matrix and 3 seeds per (ckpt, SNR) in the
robustness eval. A 3-sample bootstrap is genuinely thin -- we use it as a
descriptive interval to acknowledge seed variance rather than a hypothesis
test -- and report it that way in the paper.

We resample the 3 per-seed scalar values with replacement N=10000 times,
take the percentile interval [2.5, 97.5] on the resampled mean, and report
mean (CI_low, CI_high).

Run:
  python -m src.eval.bootstrap
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

METRICS = [
    "macro_f1",
    "accuracy",
    "ece",
    "belly_pain_recall",
    "burping_recall",
    "discomfort_recall",
    "hungry_recall",
    "tired_recall",
]


def bootstrap_ci(values: list[float], n_boot: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI on the mean. Returns (mean, lo, hi)."""
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def aggregate_matrix(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    """Group by group_keys, bootstrap each metric, return summary rows."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = tuple(r.get(k, "") for k in group_keys)
        groups[key].append(r)
    out: list[dict] = []
    for key, cells in groups.items():
        row = {k: v for k, v in zip(group_keys, key)}
        row["n_seeds"] = len(cells)
        for m in METRICS:
            vals: list[float] = []
            for c in cells:
                try:
                    vals.append(float(c[m]))
                except (KeyError, ValueError):
                    pass
            mean, lo, hi = bootstrap_ci(vals)
            row[f"{m}_mean"] = round(mean, 4)
            row[f"{m}_ci_lo"] = round(lo, 4)
            row[f"{m}_ci_hi"] = round(hi, 4)
        out.append(row)
    return out


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=Path, default=REPO / "results" / "bootstrap")
    args = p.parse_args()

    summaries: dict[str, list[dict]] = {}

    # 1. Main matrix (4 aug x 3 recipes x 3 seeds = 36 cells)
    matrix_rows = load(REPO / "results" / "matrix_part1.csv") + load(REPO / "results" / "matrix_part2.csv")
    summaries["matrix"] = aggregate_matrix(matrix_rows, ("aug_type", "recipe"))

    # 2. Ratio sweep (3 ratios x 3 seeds = 9 cells)
    ratio_rows = load(REPO / "results" / "matrix_ratio_sweep.csv")
    summaries["ratio_sweep"] = aggregate_matrix(ratio_rows, ("aug_type", "recipe", "synth_ratio"))

    # 3. Noise robustness (2 arms x 5 SNRs x 3 seeds = 30 cells)
    robustness_rows = (
        load(REPO / "results" / "robustness_none_weighted_ce.csv")
        + load(REPO / "results" / "robustness_generative_weighted_ce.csv")
    )
    # Each row has cell + snr_db; we want arm-level grouping by cell-name prefix
    for r in robustness_rows:
        cell = r.get("cell", "")
        if cell.startswith("generative_"):
            r["arm"] = "generative+weighted_ce"
        elif cell.startswith("none_"):
            r["arm"] = "baseline (none+weighted_ce)"
        else:
            r["arm"] = "unknown"
    summaries["robustness"] = aggregate_matrix(robustness_rows, ("arm", "snr_db"))

    for name, rows in summaries.items():
        out_csv = args.out_dir / f"{name}_bootstrap.csv"
        write_csv(rows, out_csv)
        print(f"[bootstrap] wrote {out_csv} ({len(rows)} rows)")

    # Print the headline matrix table for the paper
    print("\n[bootstrap] Main matrix (mean [95% CI]):")
    print(f"  {'arm':22s} {'recipe':12s} {'macroF1':>20s} {'acc':>20s} {'burping_rec':>22s}")
    for r in sorted(summaries["matrix"], key=lambda x: (x["aug_type"], x["recipe"])):
        f1 = f"{r['macro_f1_mean']:.3f} [{r['macro_f1_ci_lo']:.3f}, {r['macro_f1_ci_hi']:.3f}]"
        acc = f"{r['accuracy_mean']:.3f} [{r['accuracy_ci_lo']:.3f}, {r['accuracy_ci_hi']:.3f}]"
        burp = f"{r['burping_recall_mean']:.3f} [{r['burping_recall_ci_lo']:.3f}, {r['burping_recall_ci_hi']:.3f}]"
        print(f"  {r['aug_type']:22s} {r['recipe']:12s} {f1:>20s} {acc:>20s} {burp:>22s}")


if __name__ == "__main__":
    main()
