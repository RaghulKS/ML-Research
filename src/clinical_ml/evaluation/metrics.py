"""Threshold-free and thresholded binary metrics.

AUROC and AUPRC are the primary comparison metrics. Thresholded metrics are
reported at 0.5 and, separately, at a threshold chosen on validation data.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_auroc(y, p) -> float | None:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, p))


def safe_auprc(y, p) -> float | None:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return None
    return float(average_precision_score(y, p))


def threshold_metrics(y, p, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else None
    specificity = float(tn / (tn + fp)) if (tn + fp) else None
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def calibration_bins(y, p, n_bins: int = 5) -> list[dict]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        count = int(mask.sum())
        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "n": count,
                "mean_predicted": float(p[mask].mean()) if count else None,
                "fraction_positive": float(y[mask].mean()) if count else None,
            }
        )
    return rows


def compute_metrics(y, p, threshold: float = 0.5, n_bins: int = 5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    metrics = {
        "n": int(len(y)),
        "n_positive": int(y.sum()) if len(y) else 0,
        "auroc": safe_auroc(y, p),
        "auprc": safe_auprc(y, p),
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
        "calibration_bins": calibration_bins(y, p, n_bins=n_bins) if len(y) else [],
    }
    metrics.update(threshold_metrics(y, p, threshold))
    if metrics["auroc"] is None:
        metrics["undefined_reason"] = "AUROC and AUPRC are undefined when only one class is present."
    return metrics


def best_f1_threshold(y, p) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5
    best_t = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        score = threshold_metrics(y, p, float(threshold))["f1"]
        if score > best_f1:
            best_f1 = score
            best_t = float(threshold)
    return best_t


def bootstrap_ci(y, p, metric: str = "auroc", n_boot: int = 200, seed: int = 0) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    fn = safe_auroc if metric == "auroc" else safe_auprc
    stats = []
    n = len(y)
    if n == 0 or len(np.unique(y)) < 2:
        return {"metric": metric, "low": None, "high": None, "n_used": 0, "n_boot": n_boot}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        value = fn(y[idx], p[idx])
        if value is not None:
            stats.append(value)
    if len(stats) < max(20, n_boot // 5):
        return {"metric": metric, "low": None, "high": None, "n_used": len(stats), "n_boot": n_boot}
    low, high = np.quantile(stats, [0.025, 0.975])
    return {
        "metric": metric,
        "low": float(low),
        "high": float(high),
        "n_used": len(stats),
        "n_boot": n_boot,
    }
