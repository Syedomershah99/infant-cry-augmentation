"""Aggregate matrix CSVs into a clean per-cell mean+/-std summary.

Run:
  python -m src.eval.aggregate
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parents[2]


METRICS = [
    "macro_f1", "accuracy", "ece",
    "belly_pain_recall", "burping_recall", "discomfort_recall",
    "hungry_recall", "tired_recall",
]


def load(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        if not p.exists():
            continue
        with p.open() as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    """Group by (aug_type, recipe, synth_ratio); compute mean+std for each metric."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["aug_type"], r.get("recipe", "weighted_ce"), r.get("synth_ratio", "0"))
        groups[key].append(r)

    out: list[dict] = []
    for (aug, recipe, ratio), cells in groups.items():
        row = {"aug_type": aug, "recipe": recipe, "synth_ratio": ratio, "n_seeds": len(cells)}
        for m in METRICS:
            vals: list[float] = []
            for c in cells:
                try:
                    vals.append(float(c[m]))
                except (KeyError, ValueError):
                    pass
            if not vals:
                row[f"{m}_mean"] = ""
                row[f"{m}_std"] = ""
                continue
            row[f"{m}_mean"] = round(mean(vals), 4)
            row[f"{m}_std"] = round(stdev(vals), 4) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[REPO / "results" / "matrix.csv", REPO / "results" / "matrix_part1.csv", REPO / "results" / "matrix_part2.csv"],
    )
    p.add_argument("--out_csv", type=Path, default=REPO / "results" / "matrix_summary.csv")
    p.add_argument("--out_json", type=Path, default=REPO / "results" / "matrix_summary.json")
    args = p.parse_args()

    rows = load(args.inputs)
    if not rows:
        raise SystemExit("no input rows")
    print(f"[aggregate] {len(rows)} rows")
    summary = aggregate(rows)
    if not summary:
        raise SystemExit("empty summary")
    fieldnames = list(summary[0].keys())
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary:
            w.writerow(r)
    args.out_json.write_text(json.dumps(summary, indent=2))
    print(f"[aggregate] wrote {args.out_csv} and {args.out_json}")
    # Print headline table to stdout
    print("\n[aggregate] headline (mean over seeds):")
    print(f"  {'arm':22s} {'recipe':12s} {'ratio':>5s} {'macroF1':>8s} {'acc':>6s} {'ECE':>6s} {'bp_rec':>7s} {'burp_rec':>8s}")
    for r in sorted(summary, key=lambda x: (x["aug_type"], x["recipe"], int(str(x["synth_ratio"]) or 0))):
        print(
            f"  {r['aug_type']:22s} {r['recipe']:12s} {str(r['synth_ratio']):>5s} "
            f"{r['macro_f1_mean']:>8.3f} {r['accuracy_mean']:>6.3f} {r['ece_mean']:>6.3f} "
            f"{r['belly_pain_recall_mean']:>7.3f} {r['burping_recall_mean']:>8.3f}"
        )


if __name__ == "__main__":
    main()
