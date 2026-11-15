"""Logistic regression, random forest, and gradient boosting.

Hyperparameters are chosen on the validation split. The test split is scored
once after that choice. If XGBoost is not importable, a scikit-learn gradient
boosting model is fit with the same depth grid and the result is labeled as a
fallback rather than as XGBoost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clinical_ml.evaluation.metrics import compute_metrics


@dataclass
class BaselineSpec:
    name: str
    family: str
    params: dict
    fallback: str | None = None


def _logistic(c_value: float, seed: int, max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=max_iter,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def _forest(max_depth: int, seed: int, n_estimators: int, min_samples_leaf: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=2,
                ),
            ),
        ]
    )


def _xgboost(max_depth: int, seed: int, n_estimators: int, learning_rate: float, subsample: float, colsample: float, scale_pos_weight: float):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=seed,
                    n_jobs=2,
                    scale_pos_weight=scale_pos_weight,
                    verbosity=0,
                ),
            ),
        ]
    )


def _boosting_fallback(max_depth: int, seed: int, n_estimators: int, learning_rate: float) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=seed,
                ),
            ),
        ]
    )


def balanced_sample_weight(y) -> np.ndarray:
    y = np.asarray(y).astype(int)
    n = len(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_pos = n_pos if n_pos > 0 else 1
    n_neg = n_neg if n_neg > 0 else 1
    weights = np.where(y == 1, 0.5 * n / n_pos, 0.5 * n / n_neg)
    return weights.astype(float)


def selection_score(metrics: dict) -> float:
    if metrics.get("auprc") is not None:
        return float(metrics["auprc"])
    if metrics.get("auroc") is not None:
        return float(metrics["auroc"])
    brier = metrics.get("brier")
    return -float(brier) if brier is not None else -1e9


def predict_proba(model, x) -> np.ndarray:
    probabilities = model.predict_proba(x)[:, 1]
    return np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)


def fit_selected_baseline(family: str, config: dict, x_train, y_train, x_val, y_val, seed: int) -> dict:
    """Fit the small validation grid for one model family."""
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"{family} training split contains only one class.")
    y_train = np.asarray(y_train).astype(int)
    candidates = []
    scale = float((y_train == 0).sum() / max(int((y_train == 1).sum()), 1))
    if family == "logistic_regression":
        for c_value in config.get("C_grid", [1.0]):
            model = _logistic(float(c_value), seed, int(config.get("max_iter", 1000)))
            model.fit(x_train, y_train)
            metrics = compute_metrics(y_val, predict_proba(model, x_val))
            candidates.append((selection_score(metrics), {"C": float(c_value)}, model, metrics))
        fallback = None
    elif family == "random_forest":
        for depth in config.get("max_depth_grid", [4]):
            model = _forest(
                int(depth),
                seed,
                int(config.get("n_estimators", 200)),
                int(config.get("min_samples_leaf", 3)),
            )
            model.fit(x_train, y_train)
            metrics = compute_metrics(y_val, predict_proba(model, x_val))
            candidates.append((selection_score(metrics), {"max_depth": int(depth)}, model, metrics))
        fallback = None
    elif family == "xgboost":
        depth_grid = config.get("max_depth_grid", [3])
        probe = _xgboost(int(depth_grid[0]), seed, 2, 0.1, 1.0, 1.0, scale)
        use_fallback = probe is None
        for depth in depth_grid:
            if use_fallback:
                model = _boosting_fallback(
                    int(depth),
                    seed,
                    int(config.get("n_estimators", 80)),
                    float(config.get("learning_rate", 0.08)),
                )
                weight = balanced_sample_weight(y_train)
                model.fit(x_train, y_train, model__sample_weight=weight)
            else:
                model = _xgboost(
                    int(depth),
                    seed,
                    int(config.get("n_estimators", 80)),
                    float(config.get("learning_rate", 0.08)),
                    float(config.get("subsample", 0.9)),
                    float(config.get("colsample_bytree", 0.9)),
                    scale,
                )
                model.fit(x_train, y_train)
            metrics = compute_metrics(y_val, predict_proba(model, x_val))
            candidates.append((selection_score(metrics), {"max_depth": int(depth)}, model, metrics))
        fallback = "sklearn.GradientBoostingClassifier" if use_fallback else None
    else:
        raise ValueError(f"Unknown baseline family: {family}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, params, model, val_metrics = candidates[0]
    spec = BaselineSpec(
        name="gradient_boosting_fallback" if fallback else family,
        family=family,
        params={**params, **{k: v for k, v in config.items() if not str(k).endswith("_grid")}},
        fallback=fallback,
    )
    return {
        "spec": spec,
        "model": model,
        "validation_selection_score": score,
        "validation_metrics": val_metrics,
        "scale_pos_weight": scale if family == "xgboost" and not fallback else None,
    }
