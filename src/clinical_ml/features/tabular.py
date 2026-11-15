"""Pre-anchor tabular features for classical models.

Every aggregate uses events at or before the prediction anchor. Medication
order flags are not administration flags.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from clinical_ml.cohorts.definitions import CodeRule, code_matches

FEATURE_DX = {
    "has_diabetes": (
        CodeRule("ICD10", "E08", "prefix"),
        CodeRule("ICD10", "E09", "prefix"),
        CodeRule("ICD10", "E10", "prefix"),
        CodeRule("ICD10", "E11", "prefix"),
        CodeRule("ICD10", "E13", "prefix"),
        CodeRule("ICD9", "250", "prefix"),
    ),
    "has_hypertension": (
        CodeRule("ICD10", "I10", "exact"),
        CodeRule("ICD10", "I11", "prefix"),
        CodeRule("ICD10", "I12", "prefix"),
        CodeRule("ICD10", "I13", "prefix"),
        CodeRule("ICD10", "I15", "prefix"),
        CodeRule("ICD9", "401", "prefix"),
        CodeRule("ICD9", "402", "prefix"),
        CodeRule("ICD9", "403", "prefix"),
        CodeRule("ICD9", "404", "prefix"),
        CodeRule("ICD9", "405", "prefix"),
    ),
    "has_ckd": (CodeRule("ICD10", "N18", "prefix"), CodeRule("ICD9", "585", "prefix")),
    "has_copd": (
        CodeRule("ICD10", "J44", "prefix"),
        CodeRule("ICD9", "496", "exact"),
        CodeRule("ICD9", "4912", "prefix"),
    ),
    "has_atrial_fibrillation": (CodeRule("ICD10", "I48", "prefix"), CodeRule("ICD9", "4273", "prefix")),
    "has_stroke_history": (
        CodeRule("ICD10", "I60", "prefix"),
        CodeRule("ICD10", "I61", "prefix"),
        CodeRule("ICD10", "I63", "prefix"),
        CodeRule("ICD10", "I64", "exact"),
        CodeRule("ICD9", "430", "exact"),
        CodeRule("ICD9", "431", "exact"),
        CodeRule("ICD9", "433", "prefix"),
        CodeRule("ICD9", "434", "prefix"),
        CodeRule("ICD9", "436", "exact"),
    ),
    "has_mi_history": (
        CodeRule("ICD10", "I21", "prefix"),
        CodeRule("ICD10", "I22", "prefix"),
        CodeRule("ICD9", "410", "prefix"),
    ),
}

MED_CLASSES = {
    "diuretic": {"MED:LOOP_DIURETIC", "FUROSEMIDE", "LASIX", "BUMETANIDE", "HCTZ"},
    "acei_arb": {"MED:ACEI", "MED:ARB", "LISINOPRIL", "LOSARTAN", "ENALAPRIL", "VALSARTAN"},
    "beta_blocker": {"MED:BETA_BLOCKER", "METOPROLOL", "CARVEDILOL", "BISOPROLOL"},
    "anticoagulant": {"MED:ANTICOAGULANT", "APIXABAN", "WARFARIN", "HEPARIN"},
    "statin": {"MED:STATIN", "ATORVASTATIN", "ROSUVASTATIN"},
}

LABS = ("CREAT", "SODIUM", "POTASSIUM", "HGB", "BNP")
VITALS = ("SBP", "DBP", "HR")

FEATURE_COLUMNS = [
    "age",
    "sex_male",
    "sex_missing",
    "n_history_events",
    "n_encounters",
    "n_prior_encounters",
    "n_emergency_prior",
    "admission_type_missing",
    "n_diagnosis_events",
    "n_unique_diagnoses",
    "n_med_orders",
    "n_med_admins",
    "n_procedures",
    "n_lab_events",
    "n_vital_events",
    "n_treatment_events",
    "has_diabetes",
    "has_hypertension",
    "has_ckd",
    "has_copd",
    "has_atrial_fibrillation",
    "has_stroke_history",
    "has_mi_history",
    "med_diuretic_order",
    "med_diuretic_admin",
    "med_acei_arb_order",
    "med_acei_arb_admin",
    "med_beta_blocker_order",
    "med_beta_blocker_admin",
    "med_anticoagulant_order",
    "med_anticoagulant_admin",
    "med_statin_order",
    "med_statin_admin",
    "creat_last",
    "creat_mean",
    "creat_delta",
    "creat_count",
    "creat_missing",
    "sodium_last",
    "sodium_missing",
    "potassium_last",
    "potassium_missing",
    "hgb_last",
    "hgb_missing",
    "bnp_last",
    "bnp_missing",
    "sbp_last",
    "sbp_delta",
    "sbp_missing",
    "dbp_last",
    "dbp_missing",
    "hr_last",
    "hr_missing",
    "days_since_previous_encounter",
    "events_last_30d",
    "events_last_180d",
    "index_los_days",
]


def build_feature_frame(
    events: pd.DataFrame,
    patients: pd.DataFrame,
    encounters: pd.DataFrame,
    labeled: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per labeled patient from pre-anchor information only."""
    if labeled.empty:
        return pd.DataFrame(columns=["patient_id", "audit_max_history_time", "audit_n_events_after_anchor", *FEATURE_COLUMNS])
    patient_lookup = patients.set_index("patient_id") if len(patients) else pd.DataFrame()
    event_groups = {pid: group for pid, group in events.groupby("patient_id", sort=False)} if len(events) else {}
    encounter_groups = (
        {pid: group for pid, group in encounters.groupby("patient_id", sort=False)} if len(encounters) else {}
    )
    rows = []
    for record in labeled.itertuples(index=False):
        pid = record.patient_id
        anchor = float(record.anchor_time)
        origin = record.time_origin
        group = event_groups.get(pid, events.iloc[0:0])
        same = group["time_origin"].eq(origin) if len(group) else pd.Series(dtype=bool)
        history = group.loc[same & (group["event_time"] <= anchor + 1e-6)] if len(group) else group
        future_n = int((same & (group["event_time"] > anchor + 1e-6)).sum()) if len(group) else 0
        row = _features_for_history(
            history,
            encounter_groups.get(pid, encounters.iloc[0:0]),
            patient_lookup.loc[pid] if len(patient_lookup) and pid in patient_lookup.index else None,
            record,
        )
        row["patient_id"] = pid
        row["audit_max_history_time"] = float(history["event_time"].max()) if len(history) else np.nan
        row["audit_n_events_after_anchor"] = future_n
        rows.append(row)
    return pd.DataFrame(rows)


def _features_for_history(history: pd.DataFrame, encounters: pd.DataFrame, patient, record) -> dict:
    row = {name: np.nan for name in FEATURE_COLUMNS}
    age, sex = _demographics(patient)
    row["age"] = age
    sex_male, sex_missing = _sex_flags(sex)
    row["sex_male"] = sex_male
    row["sex_missing"] = sex_missing
    row["n_history_events"] = int(len(history))
    row["index_los_days"] = float(record.anchor_time) - float(record.index_start)
    if len(history):
        row["n_encounters"] = int(history.groupby("encounter_id").ngroups)
        row["n_diagnosis_events"] = int(history["event_type"].eq("diagnosis").sum())
        diagnoses = history.loc[history["event_type"].eq("diagnosis"), "event_code"]
        row["n_unique_diagnoses"] = int(diagnoses.nunique())
        row["n_med_orders"] = int(history["event_type"].eq("medication_order").sum())
        row["n_med_admins"] = int(history["event_type"].eq("medication_admin").sum())
        row["n_procedures"] = int(history["event_type"].eq("procedure").sum())
        row["n_lab_events"] = int(history["event_type"].eq("lab").sum())
        row["n_vital_events"] = int(history["event_type"].eq("vital").sum())
        row["n_treatment_events"] = int(history["treatment_indicator"].astype(bool).sum())
        for name, rules in FEATURE_DX.items():
            row[name] = int(_any_diagnosis(history, rules))
        for class_name, codes in MED_CLASSES.items():
            row[f"med_{class_name}_order"] = int(_has_med(history, codes, "medication_order"))
            row[f"med_{class_name}_admin"] = int(_has_med(history, codes, "medication_admin"))
        for analyte in LABS:
            stats = _numeric_stats(history, "lab", analyte)
            prefix = analyte.lower()
            row[f"{prefix}_last"] = stats["last"]
            row[f"{prefix}_missing"] = stats["missing"]
            if analyte == "CREAT":
                row["creat_mean"] = stats["mean"]
                row["creat_delta"] = stats["delta"]
                row["creat_count"] = stats["count"]
        for analyte in VITALS:
            stats = _numeric_stats(history, "vital", analyte)
            prefix = analyte.lower()
            row[f"{prefix}_last"] = stats["last"]
            row[f"{prefix}_missing"] = stats["missing"]
            if analyte == "SBP":
                row["sbp_delta"] = stats["delta"]
        anchor = float(record.anchor_time)
        row["events_last_30d"] = int((history["event_time"] >= anchor - 30).sum())
        row["events_last_180d"] = int((history["event_time"] >= anchor - 180).sum())
    else:
        for name in FEATURE_DX:
            row[name] = 0
        for class_name in MED_CLASSES:
            row[f"med_{class_name}_order"] = 0
            row[f"med_{class_name}_admin"] = 0
    prior = _prior_encounters(encounters, record)
    row["n_prior_encounters"] = prior["n_prior"]
    row["n_emergency_prior"] = prior["n_emergency"]
    row["admission_type_missing"] = prior["type_missing"]
    row["days_since_previous_encounter"] = prior["days_since"]
    return row


def _demographics(patient):
    if patient is None:
        return np.nan, ""
    if isinstance(patient, pd.DataFrame):
        patient = patient.iloc[0]
    age = pd.to_numeric(pd.Series([patient.get("age") if hasattr(patient, "get") else patient["age"]]), errors="coerce").iloc[0]
    sex = patient["sex"] if "sex" in patient else ""
    return age, sex


def _sex_flags(sex) -> tuple[float, float]:
    text = "" if sex is None or (isinstance(sex, float) and np.isnan(sex)) else str(sex).strip().lower()
    if text in {"m", "male"}:
        return 1.0, 0.0
    if text in {"f", "female"}:
        return 0.0, 0.0
    return 0.0, 1.0


def _any_diagnosis(history: pd.DataFrame, rules) -> bool:
    dx = history[history["event_type"].eq("diagnosis")]
    for code, system in zip(dx["event_code"], dx["coding_system"]):
        if code_matches(str(code), None if pd.isna(system) else str(system), rules):
            return True
    return False


def _has_med(history: pd.DataFrame, codes: set[str], event_type: str) -> bool:
    subset = history[history["event_type"].eq(event_type)]
    if subset.empty:
        return False
    return bool(subset["event_code"].astype(str).str.upper().isin(codes).any())


def _numeric_stats(history: pd.DataFrame, event_type: str, code: str) -> dict:
    subset = history[(history["event_type"].eq(event_type)) & (history["event_code"].astype(str).eq(code))]
    values = pd.to_numeric(subset["event_value"], errors="coerce")
    kept = subset.loc[values.notna()].copy()
    kept["_value"] = values.loc[values.notna()].to_numpy()
    if kept.empty:
        return {"last": np.nan, "mean": np.nan, "delta": np.nan, "count": 0, "missing": 1.0}
    kept = kept.sort_values("event_time", kind="mergesort")
    series = kept["_value"].to_numpy(dtype=float)
    delta = float(series[-1] - series[0]) if len(series) >= 2 else np.nan
    return {
        "last": float(series[-1]),
        "mean": float(series.mean()),
        "delta": delta,
        "count": int(len(series)),
        "missing": 0.0,
    }


def _prior_encounters(encounters: pd.DataFrame, record) -> dict:
    if encounters is None or len(encounters) == 0:
        return {"n_prior": 0, "n_emergency": np.nan, "type_missing": 1.0, "days_since": np.nan}
    origin = record.time_origin
    index_start = float(record.index_start)
    same = encounters["time_origin"].eq(origin)
    typed = encounters.loc[same, "admission_type"].astype(str).str.strip()
    type_missing = float(typed.isin(["", "nan", "None"]).all())
    prior = encounters.loc[same & (encounters["end_time"] < index_start - 1e-6)]
    n_prior = int(prior["encounter_id"].nunique()) if len(prior) else 0
    if type_missing:
        n_emergency = np.nan
    else:
        kinds = prior["admission_type"].astype(str).str.upper()
        n_emergency = int(kinds.isin(["EMERGENCY", "URGENT", "ED"]).sum()) if len(prior) else 0
    if len(prior) and prior["end_time"].notna().any():
        days_since = index_start - float(prior["end_time"].max())
    else:
        days_since = np.nan
    return {"n_prior": n_prior, "n_emergency": n_emergency, "type_missing": type_missing, "days_since": days_since}


def matrix_for_ids(feature_frame: pd.DataFrame, ids: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Return model matrix columns and labels aligned to ``ids`` order in the feature frame."""
    subset = feature_frame[feature_frame["patient_id"].isin(ids)].copy()
    if "label" not in subset.columns:
        raise KeyError("Feature frame must contain label before matrix_for_ids.")
    x = subset[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y = subset["label"].astype(int)
    return x, y


def drop_all_nan_columns(train_x: pd.DataFrame, others: list[pd.DataFrame]) -> tuple[pd.DataFrame, list[pd.DataFrame], list[str]]:
    keep = [column for column in train_x.columns if train_x[column].notna().any()]
    dropped = [column for column in train_x.columns if column not in keep]
    return train_x[keep], [frame[keep] for frame in others], dropped
