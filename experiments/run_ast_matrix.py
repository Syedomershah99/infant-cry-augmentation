"""Run a small AST linear-probe matrix: 4 aug arms x 3 seeds = 12 cells.

Same conventions as ``experiments.run_matrix``, but consumes cached AST
embeddings instead of waveform-derived log-mel spectrograms. The synth
augmentation is approximated in AST embedding space (see
``src.training.train_ast_probe.build_synth_proxy_items``), so this section
of the paper reports a *backbone-comparison* result, not a clean replication
of the DDPM pipeline through AST.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Cell:
    aug_type: str
    seed: int
    epochs: int
    out_dir: Path

    @property
    def name(self) -> str:
        return f"{self.aug_type.replace('+','-')}_weighted_ce_s{self.seed}"


def make_cell_config(cell: Cell, base_config_path: Path) -> Path:
    cfg = yaml.safe_load(base_config_path.read_text())
    cfg["seed"] = cell.seed
    cfg["epochs"] = cell.epochs
    cfg["aug_type"] = cell.aug_type
    cfg["out_dir"] = str(cell.out_dir)
    cell.out_dir.mkdir(parents=True, exist_ok=True)
    out_cfg = cell.out_dir / "config.yaml"
    out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_cfg


def run_cell(cell: Cell, base_config: Path) -> dict:
    cfg_path = make_cell_config(cell, base_config)
    cmd = [sys.executable, "-m", "src.training.train_ast_probe", "--config", str(cfg_path)]
    print(f"[ast-matrix] {cell.name}: launching")
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    (cell.out_dir / "stdout.log").write_text(proc.stdout)
    (cell.out_dir / "stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        return {"cell": cell.name, "aug_type": cell.aug_type, "seed": cell.seed, "status": f"failed_rc{proc.returncode}"}
    result = json.loads((cell.out_dir / "result.json").read_text())
    test = result["test"]
    pc = test["per_class"]
    return {
        "cell": cell.name,
        "aug_type": cell.aug_type,
        "seed": cell.seed,
        "macro_f1": round(test["macro_f1"], 4),
        "accuracy": round(test["accuracy"], 4),
        "ece": round(test["ece"], 4),
        "belly_pain_recall": round(pc["belly_pain"]["recall"], 4),
        "burping_recall": round(pc["burping"]["recall"], 4),
        "discomfort_recall": round(pc["discomfort"]["recall"], 4),
        "hungry_recall": round(pc["hungry"]["recall"], 4),
        "tired_recall": round(pc["tired"]["recall"], 4),
        "best_val_macro_f1": round(result["best_val_macro_f1"], 4),
        "status": "ok",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base_config", type=Path, default=REPO / "configs" / "ast_baseline.yaml")
    p.add_argument("--aug_types", nargs="+", default=["none", "classical", "generative", "classical+generative"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--out", type=Path, default=REPO / "results" / "ast_matrix.csv")
    p.add_argument("--results_root", type=Path, default=REPO / "results" / "ast_probe")
    args = p.parse_args()

    args.results_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for aug, seed in itertools.product(args.aug_types, args.seeds):
        cell = Cell(
            aug_type=aug,
            seed=seed,
            epochs=args.epochs,
            out_dir=args.results_root / f"{aug.replace('+','-')}_weighted_ce_s{seed}",
        )
        row = run_cell(cell, args.base_config)
        print(f"[ast-matrix] {row}")
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[ast-matrix] wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
