"""Patient-disjoint splits. No patient appears in more than one split."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from clinical_ml.errors import LeakageError


def patient_disjoint_split(
    labeled: pd.DataFrame,
    seed: int,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> tuple[dict[str, list[str]], dict]:
    if labeled.empty:
        raise ValueError("Cannot split an empty labeled cohort.")
    ids = labeled["patient_id"].astype(str).to_numpy()
    y = labeled["label"].to_numpy()
    stratify = y if _can_stratify(y, test_size) else None
    warning = None
    if stratify is None:
        warning = "Stratification was skipped because a class had too few patients."
    ids_trainval, ids_test, y_trainval, _y_test = train_test_split(
        ids, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    relative_val = val_size / max(1e-6, (1.0 - test_size))
    stratify_inner = y_trainval if _can_stratify(y_trainval, relative_val) else None
    if stratify_inner is None and warning is None:
        warning = "Validation stratification was skipped because a class had too few patients."
    ids_train, ids_val = train_test_split(
        ids_trainval,
        test_size=relative_val,
        random_state=seed,
        stratify=stratify_inner,
    )
    splits = {
        "train": sorted(map(str, ids_train.tolist())),
        "validation": sorted(map(str, ids_val.tolist())),
        "test": sorted(map(str, ids_test.tolist())),
    }
    assert_patient_disjoint(splits)
    info = {"warning": warning, "seed": seed, "test_size": test_size, "val_size": val_size}
    return splits, info


def _can_stratify(y, test_size: float) -> bool:
    counts = pd.Series(y).value_counts()
    if len(counts) < 2:
        return False
    if int(counts.min()) < 2:
        return False
    # Both the held-out piece and the remainder need a member of each class.
    n = len(y)
    holdout = max(int(round(n * test_size)), 1)
    return holdout >= len(counts) and (n - holdout) >= len(counts)


def assert_patient_disjoint(splits: dict[str, list[str]]) -> None:
    seen: dict[str, str] = {}
    for name, ids in splits.items():
        for pid in ids:
            if pid in seen:
                raise LeakageError(f"Patient {pid} is in both {seen[pid]} and {name}.")
            seen[pid] = name


def assert_no_temporal_leakage(feature_frame: pd.DataFrame, labeled: pd.DataFrame) -> None:
    merged = feature_frame.merge(
        labeled[["patient_id", "anchor_time"]], on="patient_id", how="left", suffixes=("", "_label")
    )
    if "audit_max_history_time" not in merged.columns:
        raise LeakageError("Feature frame has no audit_max_history_time column.")
    leaked = merged["audit_max_history_time"] > merged["anchor_time"] + 1e-6
    if leaked.fillna(False).any():
        sample = merged.loc[leaked, "patient_id"].head(5).tolist()
        raise LeakageError(f"History extends past the anchor for patients {sample}.")
