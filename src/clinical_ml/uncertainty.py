"""Validation-set probability calibration.

Isotonic regression is fit on validation predictions only. A change in Brier
score is a calibration diagnostic. It is not proof of out-of-distribution
detection.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from clinical_ml.evaluation.metrics import compute_metrics


def fit_isotonic(y_val, p_val):
    y = np.asarray(y_val).astype(int)
    p = np.asarray(p_val, dtype=float)
    if len(y) < 8 or len(np.unique(y)) < 2:
        return None
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(p, y)
    return model


def apply_isotonic(model, p) -> np.ndarray | None:
    if model is None:
        return None
    return np.clip(np.asarray(model.predict(np.asarray(p, dtype=float)), dtype=float), 0.0, 1.0)


def calibration_comparison(y_val, p_val, y_test, p_test) -> dict:
    raw = compute_metrics(y_test, p_test)
    iso = fit_isotonic(y_val, p_val)
    if iso is None:
        return {
            "method": "isotonic_regression",
            "fit": False,
            "reason": "Validation split was too small or single-class for isotonic regression.",
            "brier_raw": raw["brier"],
            "brier_calibrated": None,
        }
    calibrated = apply_isotonic(iso, p_test)
    cal_metrics = compute_metrics(y_test, calibrated)
    return {
        "method": "isotonic_regression",
        "fit": True,
        "fit_split": "validation",
        "brier_raw": raw["brier"],
        "brier_calibrated": cal_metrics["brier"],
        "auroc_calibrated": cal_metrics["auroc"],
        "auprc_calibrated": cal_metrics["auprc"],
        "note": "Calibration was not used to choose the classifier.",
    }
