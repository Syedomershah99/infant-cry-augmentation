"""Classification metrics with explicit per-class reporting.

Aggregate metrics never appear in this project without a per-class breakdown.
The functions here all return dicts so every result table can be written
straight to JSON or CSV without a second formatting pass.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

LABELS = ["belly_pain", "burping", "discomfort", "hungry", "tired"]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_prf(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, n_classes=len(LABELS))
    out = {}
    for i, label in enumerate(LABELS):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(cm[i, :].sum())
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        out[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "fnr": fnr,
        }
    return out


def macro_f1(per_class: dict) -> float:
    return float(np.mean([per_class[l]["f1"] for l in LABELS]))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Standard ECE over the predicted-class confidence."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def evaluate(model: torch.nn.Module, loader, device: str = "cpu") -> dict:
    model.eval()
    all_logits = []
    all_labels = []
    all_sources = []
    with torch.no_grad():
        for batch in loader:
            spec = batch["spec"].to(device)
            logits = model(spec)
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu())
            if "source" in batch:
                # batch["source"] is a list[str] when default collate is used
                if isinstance(batch["source"], list):
                    all_sources.extend(batch["source"])
                else:
                    all_sources.extend(list(batch["source"]))
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = probs.argmax(axis=1)

    pc = per_class_prf(labels, preds)
    cm = confusion_matrix(labels, preds, n_classes=len(LABELS))
    return {
        "macro_f1": macro_f1(pc),
        "accuracy": accuracy(labels, preds),
        "ece": expected_calibration_error(probs, labels),
        "per_class": pc,
        "confusion_matrix": cm.tolist(),
        "support_total": int(len(labels)),
        "n_correct": int((preds == labels).sum()),
        "source_counts": dict(Counter(all_sources)) if all_sources else {},
    }
