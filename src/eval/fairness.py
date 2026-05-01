"""Fairness slices: per-class FNR (especially belly_pain) and per-source gaps."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .metrics import LABELS, per_class_prf


def fnr_by_class(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pc = per_class_prf(y_true, y_pred)
    return {l: pc[l]["fnr"] for l in LABELS}


def per_source_metrics(y_true: np.ndarray, y_pred: np.ndarray, sources: list[str]) -> dict[str, dict]:
    """Compute per-class P/R/F1/FNR within each source slice."""
    by_source: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for t, p, s in zip(y_true, y_pred, sources):
        by_source[s].append((int(t), int(p)))
    out: dict[str, dict] = {}
    for src, pairs in by_source.items():
        ts = np.array([t for t, _ in pairs])
        ps = np.array([p for _, p in pairs])
        out[src] = per_class_prf(ts, ps)
    return out


def cross_source_gap(per_source: dict[str, dict], primary: str, cross: str) -> dict[str, float]:
    """F1 gap (primary - cross) per class, used as a robustness proxy for population shift."""
    if primary not in per_source or cross not in per_source:
        return {}
    return {
        l: per_source[primary][l]["f1"] - per_source[cross][l]["f1"]
        for l in LABELS
    }


def safety_summary(eval_result: dict) -> dict:
    """Pull the ethics-facing safety numbers out of an `evaluate` result."""
    pc = eval_result["per_class"]
    return {
        "belly_pain_recall": pc["belly_pain"]["recall"],
        "belly_pain_fnr": pc["belly_pain"]["fnr"],
        "burping_recall": pc["burping"]["recall"],
        "burping_fnr": pc["burping"]["fnr"],
        "macro_f1": eval_result["macro_f1"],
        "ece": eval_result["ece"],
    }
