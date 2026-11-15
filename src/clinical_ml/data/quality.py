"""Data-quality summaries. These counts are not model-performance metrics."""

from __future__ import annotations

import pandas as pd

from clinical_ml.cohorts.definitions import DEFINITIONS, match_mask


def quality_report(events: pd.DataFrame, patients: pd.DataFrame | None = None) -> dict:
    if events.empty:
        return {"n_rows": 0, "n_patients": 0, "n_encounters": 0}
    duplicate_rate = float(events.duplicated().mean())
    missingness = {}
    for column in ["patient_id", "encounter_id", "event_time", "event_code", "event_value", "event_unit"]:
        if column not in events.columns:
            continue
        series = events[column]
        if column == "event_value":
            missingness[column] = float(pd.to_numeric(series, errors="coerce").isna().mean())
        else:
            text = series.astype(str).str.strip()
            missingness[column] = float(text.isin(["", "nan", "None", "NaN"]).mean())
    per_patient = events.groupby("patient_id").size()
    type_counts = events["event_type"].value_counts().to_dict()
    origins = events["time_origin"].value_counts().to_dict()
    mixed = int((events.groupby("patient_id")["time_origin"].nunique() > 1).sum())
    dx = events[events["event_type"] == "diagnosis"]
    top_dx = dx["event_code"].value_counts().head(15).to_dict() if len(dx) else {}
    coverage = _coverage(events)
    declared = dx["coding_system"].fillna("").astype(str) if len(dx) else pd.Series(dtype=str)
    code = dx["event_code"].fillna("").astype(str) if len(dx) else pd.Series(dtype=str)
    mismatch = 0
    if len(dx):
        letter = code.str[:1].str.isalpha()
        digit = code.str[:1].str.isdigit()
        mismatch = int(((declared.eq("ICD9") & letter) | (declared.eq("ICD10") & digit)).sum())
    report = {
        "n_rows": int(len(events)),
        "n_patients": int(events["patient_id"].nunique()),
        "n_encounters": int(events.groupby(["patient_id", "encounter_id"]).ngroups),
        "duplicate_rate": duplicate_rate,
        "missingness": missingness,
        "event_type_counts": {str(k): int(v) for k, v in type_counts.items()},
        "time_origin_counts": {str(k): int(v) for k, v in origins.items()},
        "patients_with_mixed_time_origins": mixed,
        "event_time_min": float(events["event_time"].min()),
        "event_time_max": float(events["event_time"].max()),
        "sequence_length": {
            "mean": float(per_patient.mean()),
            "median": float(per_patient.median()),
            "p90": float(per_patient.quantile(0.9)),
            "max": int(per_patient.max()),
            "min": int(per_patient.min()),
        },
        "top_diagnosis_codes": {str(k): int(v) for k, v in top_dx.items()},
        "coverage": coverage,
        "diagnosis_system_form_mismatches": mismatch,
        "n_patients_table": int(patients["patient_id"].nunique()) if patients is not None and len(patients) else 0,
    }
    return report


def _coverage(events: pd.DataFrame) -> dict:
    patients = set(events["patient_id"].unique())
    n = max(len(patients), 1)

    def fraction(event_type: str) -> float:
        have = set(events.loc[events["event_type"].eq(event_type), "patient_id"].unique())
        return float(len(have) / n)

    return {
        "patients_with_diagnosis": fraction("diagnosis"),
        "patients_with_medication_order": fraction("medication_order"),
        "patients_with_medication_admin": fraction("medication_admin"),
        "patients_with_lab": fraction("lab"),
        "patients_with_vital": fraction("vital"),
        "patients_with_procedure": fraction("procedure"),
    }


def coding_overlap_report(events: pd.DataFrame) -> dict:
    """How often ICD-9 and ICD-10 for the same condition share an encounter.

    Co-occurrence can be dual recording or a mapped code. It is not evidence of
    two independent diseases, and it is not a prevalence estimate.
    """
    dx = events[events["event_type"].eq("diagnosis")].copy()
    report = {
        "interpretation": (
            "Same-encounter ICD-9 and ICD-10 matches can be dual entry or a crosswalk, "
            "not two separate diseases. Run this on a real extract before interpreting overlap."
        )
    }
    if dx.empty:
        report["conditions"] = {}
        return report
    conditions = {}
    for (condition, phenotype), defn in DEFINITIONS.items():
        if phenotype != "historical":
            continue
        mask = match_mask(dx["event_code"], dx["coding_system"], defn.rules)
        matched = dx.loc[mask, ["patient_id", "encounter_id", "coding_system"]]
        if matched.empty:
            conditions[condition] = {
                "encounters_with_condition": 0,
                "encounters_with_both_systems": 0,
                "both_system_fraction": None,
            }
            continue
        both = 0
        encounters = 0
        for _, group in matched.groupby(["patient_id", "encounter_id"], sort=False):
            encounters += 1
            systems = set(group["coding_system"])
            if "ICD9" in systems and "ICD10" in systems:
                both += 1
        conditions[condition] = {
            "encounters_with_condition": encounters,
            "encounters_with_both_systems": both,
            "both_system_fraction": float(both / encounters) if encounters else None,
        }
    report["conditions"] = conditions
    return report
