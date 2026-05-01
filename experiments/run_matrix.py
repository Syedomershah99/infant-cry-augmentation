"""Run the augmentation x seed experiment matrix and emit a tidy CSV.

Iterates over the cross product of:
  aug_type x synth_ratio (rare-class only) x seed

For each cell, runs the classifier training driver with the right config + flags,
parses the resulting `result.json`, and writes a row of the headline metrics.

Cross-source robustness eval (test condition) is added later when a second
public corpus is integrated; the placeholder column is included now so the
schema is stable.

Run:
  python -m experiments.run_matrix --out results/matrix.csv
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Cell:
    aug_type: str       # none | classical | generative | classical+generative
    synth_ratio: int    # multiplier for rare-class synth (used to subset synth manifest)
    seed: int
    epochs: int
    out_dir: Path

    @property
    def name(self) -> str:
        return f"{self.aug_type.replace('+','-')}_r{self.synth_ratio}_s{self.seed}"


CONFIG_BY_AUG = {
    "none": "configs/baseline_cnn.yaml",
    "classical": "configs/baseline_classical.yaml",
    "generative": "configs/baseline_generative.yaml",
    "classical+generative": "configs/baseline_classical_generative.yaml",
}


def write_subset_synth_manifest(src: Path, dst: Path, rare_classes: list[str], ratio_x: int) -> None:
    """Pick the first N synthetic rows per rare class, where N = ratio_x * (real-class count).

    Real-class counts come from data/manifests/train.csv (a separate read kept here
    so the manifest builder stays the only source of truth for splits).
    """
    train_csv = REPO / "data" / "manifests" / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"missing {train_csv}")
    real_counts: dict[str, int] = {c: 0 for c in rare_classes}
    with train_csv.open() as f:
        for row in csv.DictReader(f):
            if row["label"] in real_counts:
                real_counts[row["label"]] += 1
    target = {c: real_counts[c] * ratio_x for c in rare_classes}
    print(f"[matrix] rare class real counts: {real_counts}")
    print(f"[matrix] target synth-per-class: {target}")

    if not src.exists():
        print(f"[matrix] WARN: synth manifest {src} not found; writing empty subset")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("")
        return
    with src.open() as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys() if rows else []

    by_class: dict[str, list[dict]] = {c: [] for c in rare_classes}
    for r in rows:
        if r["label"] in by_class:
            by_class[r["label"]].append(r)
    out_rows: list[dict] = []
    for c in rare_classes:
        out_rows.extend(by_class[c][: target[c]])
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="") as f:
        if not fieldnames:
            return
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"[matrix] wrote {len(out_rows)} synth rows -> {dst}")


def make_cell_config(cell: Cell, base_config_path: Path, synth_manifest: Path | None) -> Path:
    cfg = yaml.safe_load(base_config_path.read_text())
    cfg["seed"] = cell.seed
    cfg["epochs"] = cell.epochs
    cfg["out_dir"] = str(cell.out_dir)
    if "generative" in cell.aug_type and synth_manifest is not None:
        cfg["synth_manifest"] = str(synth_manifest)
    cell.out_dir.mkdir(parents=True, exist_ok=True)
    out_cfg = cell.out_dir / "config.yaml"
    out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_cfg


def run_cell(cell: Cell) -> dict:
    base_cfg = REPO / CONFIG_BY_AUG[cell.aug_type]
    if not base_cfg.exists():
        raise FileNotFoundError(base_cfg)
    synth_manifest = None
    if "generative" in cell.aug_type:
        # Subset the synth manifest to ratio_x * real-class count
        synth_src = REPO / "data" / "manifests" / "synth_train.csv"
        synth_dst = cell.out_dir / "synth_subset.csv"
        write_subset_synth_manifest(synth_src, synth_dst, ["belly_pain", "burping"], cell.synth_ratio)
        synth_manifest = synth_dst
    cfg_path = make_cell_config(cell, base_cfg, synth_manifest)

    cmd = [
        sys.executable,
        "-m",
        "src.training.train_classifier",
        "--config",
        str(cfg_path),
    ]
    print(f"[matrix] {cell.name}: launching")
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if proc.returncode != 0:
        (cell.out_dir / "stderr.log").write_text(proc.stderr)
        (cell.out_dir / "stdout.log").write_text(proc.stdout)
        return {
            "cell": cell.name,
            "aug_type": cell.aug_type,
            "synth_ratio": cell.synth_ratio,
            "seed": cell.seed,
            "status": f"failed_rc{proc.returncode}",
        }
    (cell.out_dir / "stdout.log").write_text(proc.stdout)
    result_path = cell.out_dir / "result.json"
    if not result_path.exists():
        return {"cell": cell.name, "status": "no_result_json"}
    res = json.loads(result_path.read_text())
    test = res["test"]
    pc = test["per_class"]
    return {
        "cell": cell.name,
        "aug_type": cell.aug_type,
        "synth_ratio": cell.synth_ratio,
        "seed": cell.seed,
        "macro_f1": round(test["macro_f1"], 4),
        "accuracy": round(test["accuracy"], 4),
        "ece": round(test["ece"], 4),
        "belly_pain_recall": round(pc["belly_pain"]["recall"], 4),
        "belly_pain_fnr": round(pc["belly_pain"]["fnr"], 4),
        "burping_recall": round(pc["burping"]["recall"], 4),
        "burping_fnr": round(pc["burping"]["fnr"], 4),
        "discomfort_recall": round(pc["discomfort"]["recall"], 4),
        "hungry_recall": round(pc["hungry"]["recall"], 4),
        "tired_recall": round(pc["tired"]["recall"], 4),
        "best_val_macro_f1": round(res["best_val_macro_f1"], 4),
        "status": "ok",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=REPO / "results" / "matrix.csv")
    p.add_argument("--aug_types", nargs="+", default=["none", "classical", "generative", "classical+generative"])
    p.add_argument("--ratios", nargs="+", type=int, default=[0, 1, 5, 10],
                   help="Synth-to-real ratios for rare classes; ignored for non-generative arms.")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--results_root", type=Path, default=REPO / "results" / "matrix")
    p.add_argument("--clean", action="store_true")
    args = p.parse_args()

    if args.clean and args.results_root.exists():
        shutil.rmtree(args.results_root)
    args.results_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for aug in args.aug_types:
        # Non-generative arms have no ratio dimension; collapse to one value.
        ratios = args.ratios if "generative" in aug else [0]
        for ratio, seed in itertools.product(ratios, args.seeds):
            cell = Cell(
                aug_type=aug,
                synth_ratio=ratio,
                seed=seed,
                epochs=args.epochs,
                out_dir=args.results_root / f"{aug.replace('+','-')}_r{ratio}_s{seed}",
            )
            row = run_cell(cell)
            print(f"[matrix] {row}")
            rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[matrix] wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
