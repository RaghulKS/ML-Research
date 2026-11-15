"""Discharge-time readmission labels.

The positive label is a new encounter after the index anchor. It is not the
heart-failure code that defined the cohort. Events after the anchor are not
features; they are used only to determine the label and follow-up.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from clinical_ml.cohorts.definitions import CohortDefinition

IMPLEMENTATION_ID = "readmission_v1"
EPSILON = 1e-6


def build_readmission_labels(
    events: pd.DataFrame,
    patients: pd.DataFrame,
    encounters: pd.DataFrame,
    cohort_ids: list[str] | set[str],
    definition: CohortDefinition,
    horizon_days: float,
    task_name: str,
) -> dict:
    from clinical_ml.cohorts.extraction import cohort_mask

    cohort_ids = set(cohort_ids)
    dx = events[events["event_type"].eq("diagnosis") & events["patient_id"].isin(cohort_ids)].copy()
    matched = cohort_mask(dx, definition) if len(dx) else pd.Series(False, index=dx.index)
    dx = dx.loc[matched]
    enc = encounters.copy()
    if dx.empty or enc.empty:
        return _empty_result(task_name, definition, horizon_days, len(cohort_ids))
    dx = dx.merge(
        enc[["patient_id", "encounter_id", "start_time", "end_time", "time_origin"]],
        on=["patient_id", "encounter_id"],
        how="left",
        suffixes=("", "_enc"),
    )
    # Prefer the encounter clock. Drop rows whose encounter clock disagrees.
    if "time_origin_enc" in dx.columns:
        same_clock = dx["time_origin"].eq(dx["time_origin_enc"]) | dx["time_origin_enc"].isna()
        dx = dx.loc[same_clock].copy()
    dx = dx[dx["start_time"].notna() & dx["end_time"].notna()].copy()
    if dx.empty:
        return _empty_result(task_name, definition, horizon_days, len(cohort_ids))
    dx = dx.sort_values(["patient_id", "start_time", "encounter_id"], kind="mergesort")
    index_rows = dx.drop_duplicates("patient_id", keep="first")
    patient_follow = patients.set_index("patient_id") if len(patients) else pd.DataFrame()
    max_event_time = events.groupby("patient_id")["event_time"].max()
    labeled_rows = []
    exclusions = []
    for row in index_rows.itertuples(index=False):
        pid = row.patient_id
        anchor = float(row.end_time)
        origin = row.time_origin
        index_start = float(row.start_time)
        future = enc[
            (enc["patient_id"] == pid)
            & (enc["encounter_id"] != row.encounter_id)
            & (enc["time_origin"] == origin)
            & (enc["start_time"] > anchor + EPSILON)
        ]
        in_horizon = future[future["start_time"] <= anchor + horizon_days + EPSILON]
        followup = _followup_end(pid, origin, anchor, patient_follow, max_event_time, enc)
        history = events[
            (events["patient_id"] == pid)
            & (events["time_origin"] == origin)
            & (events["event_time"] <= anchor + EPSILON)
        ]
        if history.empty:
            exclusions.append({"patient_id": pid, "reason": "no_history_events"})
            continue
        if len(in_horizon):
            label = 1
        elif followup is not None and followup >= anchor + horizon_days - EPSILON:
            label = 0
        else:
            exclusions.append({"patient_id": pid, "reason": "censored_insufficient_followup"})
            continue
        labeled_rows.append(
            {
                "patient_id": pid,
                "index_encounter_id": row.encounter_id,
                "index_start": index_start,
                "anchor_time": anchor,
                "time_origin": origin,
                "label": int(label),
                "followup_end_time": followup,
                "horizon_days": float(horizon_days),
                "cohort": definition.condition,
                "phenotype": definition.phenotype,
                "cohort_version": definition.version,
                "task": task_name,
                "implementation": IMPLEMENTATION_ID,
                "source_dataset": row.source_dataset if hasattr(row, "source_dataset") else "",
            }
        )
    labeled = pd.DataFrame(labeled_rows)
    excluded = pd.DataFrame(exclusions)
    n_censored = int((excluded["reason"] == "censored_insufficient_followup").sum()) if len(excluded) else 0
    summary = {
        "task": task_name,
        "implementation": IMPLEMENTATION_ID,
        "cohort_version": definition.version,
        "horizon_days": float(horizon_days),
        "n_cohort_patients": len(cohort_ids),
        "n_labeled": int(len(labeled)),
        "n_positive": int((labeled["label"] == 1).sum()) if len(labeled) else 0,
        "n_negative": int((labeled["label"] == 0).sum()) if len(labeled) else 0,
        "n_censored_insufficient_followup": n_censored,
        "n_excluded_no_history": int((excluded["reason"] == "no_history_events").sum()) if len(excluded) else 0,
        "positive_definition": (
            "New encounter with start strictly after the index anchor and at most "
            f"{horizon_days} days later. Any diagnosis counts. This is not a repeat "
            "of the cohort-defining code."
        ),
        "negative_definition": (
            "No new encounter in the horizon, and observation continues through the horizon."
        ),
        "censor_definition": "Observation ends before the horizon without a qualifying encounter.",
        "anchor_definition": (
            "Anchor is the index encounter end: discharge time when an admission table "
            "provides it, otherwise the last in-encounter event time."
        ),
    }
    return {"labeled": labeled, "exclusions": excluded, "summary": summary}


def _followup_end(pid, origin, anchor, patients, max_event_time, encounters) -> float | None:
    candidates = []
    if pid in max_event_time.index and pd.notna(max_event_time.loc[pid]):
        candidates.append(float(max_event_time.loc[pid]))
    if len(patients) and pid in patients.index:
        row = patients.loc[pid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        same_clock = str(row.get("time_origin", "")) == str(origin)
        if same_clock and pd.notna(row.get("followup_end_time", np.nan)):
            candidates.append(float(row["followup_end_time"]))
    enc = encounters[(encounters["patient_id"] == pid) & (encounters["time_origin"] == origin)]
    if len(enc) and enc["end_time"].notna().any():
        candidates.append(float(enc["end_time"].max()))
    if not candidates:
        return None
    return float(np.nanmax(candidates))


def _empty_result(task_name, definition, horizon_days, n_cohort) -> dict:
    return {
        "labeled": pd.DataFrame(),
        "exclusions": pd.DataFrame(columns=["patient_id", "reason"]),
        "summary": {
            "task": task_name,
            "implementation": IMPLEMENTATION_ID,
            "cohort_version": definition.version,
            "horizon_days": float(horizon_days),
            "n_cohort_patients": n_cohort,
            "n_labeled": 0,
            "n_positive": 0,
            "n_negative": 0,
            "n_censored_insufficient_followup": 0,
            "n_excluded_no_history": 0,
        },
    }
