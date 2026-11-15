"""Cohort membership from the unified event table."""

from __future__ import annotations

import pandas as pd

from clinical_ml.cohorts.definitions import CohortDefinition, match_mask


def cohort_mask(events: pd.DataFrame, definition: CohortDefinition) -> pd.Series:
    mask = pd.Series(False, index=events.index)
    dx = events["event_type"].eq("diagnosis")
    if not dx.any():
        return mask
    matched = match_mask(events.loc[dx, "event_code"], events.loc[dx, "coding_system"], definition.rules)
    mask.loc[dx] = matched.to_numpy()
    return mask


def cohort_patient_ids(events: pd.DataFrame, definition: CohortDefinition) -> list[str]:
    mask = cohort_mask(events, definition)
    if not mask.any():
        return []
    return sorted(events.loc[mask, "patient_id"].astype(str).unique().tolist())


def cohort_summary(events: pd.DataFrame, definition: CohortDefinition) -> dict:
    mask = cohort_mask(events, definition)
    matched = events.loc[mask]
    n_encounters = int(matched.groupby(["patient_id", "encounter_id"]).ngroups) if len(matched) else 0
    return {
        "cohort_version": definition.version,
        "condition": definition.condition,
        "phenotype": definition.phenotype,
        "description": definition.description,
        "n_patients": int(matched["patient_id"].nunique()) if len(matched) else 0,
        "n_encounters": n_encounters,
        "n_diagnosis_rows": int(len(matched)),
    }
