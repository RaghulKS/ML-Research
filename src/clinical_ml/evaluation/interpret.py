"""Descriptive importance. These are not causal effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def correlation_screen(x: pd.DataFrame, y, threshold: float = 0.98) -> list[dict]:
    """Flag features that are nearly identical to the label.

    A flag is a leakage warning to investigate. It is not automatic proof of
    leakage, and it is not a reason to claim a causal driver.
    """
    y = np.asarray(y, dtype=float)
    flags = []
    if len(y) < 3 or np.unique(y).size < 2:
        return flags
    for column in x.columns:
        values = pd.to_numeric(x[column], errors="coerce")
        mask = values.notna()
        if int(mask.sum()) < 3 or values[mask].nunique() < 2:
            continue
        corr = np.corrcoef(values[mask].to_numpy(dtype=float), y[mask.to_numpy()])[0, 1]
        if np.isfinite(corr) and abs(corr) >= threshold:
            flags.append({"feature": column, "abs_correlation": float(abs(corr)), "threshold": threshold})
    return flags


def permutation_importance_table(model, x: pd.DataFrame, y, seed: int, n_repeats: int = 5) -> pd.DataFrame:
    result = permutation_importance(
        model,
        x,
        y,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="average_precision",
    )
    table = pd.DataFrame(
        {
            "feature": list(x.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return table.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def impurity_importance_table(model, feature_names: list[str]) -> pd.DataFrame | None:
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if not hasattr(estimator, "feature_importances_"):
        return None
    table = pd.DataFrame({"feature": feature_names, "impurity_importance": estimator.feature_importances_})
    return table.sort_values("impurity_importance", ascending=False).reset_index(drop=True)


def try_shap_summary(model, x: pd.DataFrame, max_rows: int = 40) -> dict:
    """Optional SHAP values. Absence of the package is reported, not invented."""
    try:
        import shap
    except ImportError:
        return {"available": False, "reason": "shap is not installed"}
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
    sample = x.head(max_rows)
    try:
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(sample)
    except Exception as exc:  # shap can fail on wrapped pipelines; keep the run alive
        return {"available": False, "reason": f"shap failed: {exc.__class__.__name__}"}
    array = np.asarray(values)
    if array.ndim == 3:
        array = array[:, :, -1]
    mean_abs = np.abs(array).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:15]
    return {
        "available": True,
        "top_features": [
            {"feature": str(sample.columns[i]), "mean_abs_shap": float(mean_abs[i])} for i in order
        ],
    }
