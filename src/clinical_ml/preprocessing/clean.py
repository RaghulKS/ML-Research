"""Row-level cleaning that does not change clinical meaning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from clinical_ml.data.schemas import EVENT_COLUMNS, validate_events


def clean_events(events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop broken identifiers and exact duplicate rows, then sort timelines.

    Clinical range checks are intentionally not done here. Those belong to
    harmonization, after units are converted.
    """
    validate_events(events)
    frame = events.copy()
    report: dict = {"n_input": int(len(frame))}
    pid = frame["patient_id"].astype(str).str.strip()
    missing_pid = pid.eq("") | pid.str.lower().isin(["nan", "none", "null", "<na>"])
    report["dropped_missing_patient_id"] = int(missing_pid.sum())
    frame = frame.loc[~missing_pid].copy()
    frame["patient_id"] = pid.loc[~missing_pid].to_numpy()

    times = pd.to_numeric(frame["event_time"], errors="coerce")
    bad_time = ~np.isfinite(times.to_numpy(dtype=float, na_value=np.nan))
    report["dropped_nonfinite_event_time"] = int(bad_time.sum())
    frame = frame.loc[~bad_time].copy()
    frame["event_time"] = times.loc[~bad_time].astype(float).to_numpy()

    duplicate = frame.duplicated(keep="first")
    report["dropped_duplicate_rows"] = int(duplicate.sum())
    frame = frame.loc[~duplicate].copy()

    enc = frame["encounter_id"].astype(str).str.strip()
    missing_enc = enc.eq("") | enc.str.lower().isin(["nan", "none", "null", "<na>"])
    report["assigned_missing_encounter_id"] = int(missing_enc.sum())
    if missing_enc.any():
        synthesized = (
            "UNASSIGNED::"
            + frame.loc[missing_enc, "patient_id"].astype(str)
            + "::"
            + frame.loc[missing_enc, "event_time"].round(6).astype(str)
        )
        frame.loc[missing_enc, "encounter_id"] = synthesized
    frame["encounter_id"] = frame["encounter_id"].astype(str)

    frame["event_value"] = pd.to_numeric(frame["event_value"], errors="coerce")
    frame["treatment_indicator"] = frame["treatment_indicator"].fillna(False).astype(bool)
    frame["administration_confirmed"] = frame["administration_confirmed"].fillna(False).astype(bool)
    frame["missingness_indicator"] = frame["missingness_indicator"].fillna(False).astype(bool)
    # A lab or vital with no number is missing even if the source left the flag false.
    measurable = frame["event_type"].isin(["lab", "vital"]) & frame["event_value"].isna()
    frame.loc[measurable, "missingness_indicator"] = True

    frame = frame.sort_values(
        ["patient_id", "event_time", "encounter_id", "event_type", "event_code"],
        kind="mergesort",
    ).reset_index(drop=True)
    report["n_output"] = int(len(frame))
    report["n_patients"] = int(frame["patient_id"].nunique())
    report["n_encounters"] = int(frame.groupby(["patient_id", "encounter_id"]).ngroups) if len(frame) else 0
    # Preserve column order for the contract, then any extras.
    extras = [c for c in frame.columns if c not in EVENT_COLUMNS]
    frame = frame[EVENT_COLUMNS + extras]
    return frame, report


def assert_time_ordered(events: pd.DataFrame) -> None:
    if events.empty:
        return
    ordered = events.sort_values(["patient_id", "event_time"], kind="mergesort")
    if not ordered["event_time"].to_numpy().shape == events["event_time"].to_numpy().shape:
        raise AssertionError("Unexpected timeline shape")
    grouped = events.groupby("patient_id", sort=False)["event_time"]
    for _, times in grouped:
        values = times.to_numpy()
        if len(values) and np.any(values[1:] < values[:-1]):
            raise AssertionError("Events are not in chronological order within a patient")
