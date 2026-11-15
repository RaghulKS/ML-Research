"""Unified longitudinal event schema.

Absolute clocks from different extracts are not interchangeable. Every event
carries ``time_origin``. Within-patient gaps are valid only inside one origin.
"""

from __future__ import annotations

import math
import re

import pandas as pd

EVENT_COLUMNS = [
    "patient_id",
    "encounter_id",
    "event_time",
    "time_origin",
    "event_type",
    "event_code",
    "raw_code",
    "event_value",
    "event_unit",
    "source_dataset",
    "event_description",
    "treatment_indicator",
    "administration_confirmed",
    "provenance",
    "missingness_indicator",
    "coding_system",
]

PATIENT_COLUMNS = [
    "patient_id",
    "age",
    "sex",
    "followup_end_time",
    "time_origin",
    "source_dataset",
]

ENCOUNTER_COLUMNS = [
    "patient_id",
    "encounter_id",
    "start_time",
    "end_time",
    "time_origin",
    "source_dataset",
    "admission_type",
    "end_source",
]

EVENT_TYPES = {
    "diagnosis",
    "medication_order",
    "medication_admin",
    "lab",
    "vital",
    "procedure",
}

# Medication orders are never stored as administrations. Confirmed
# administrations require an administration source or an explicit flag.
TREATMENT_TYPES = {"medication_order", "medication_admin", "procedure"}


def empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def validate_events(events: pd.DataFrame) -> None:
    missing = [c for c in EVENT_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"Event table is missing columns: {missing}")


def normalize_identifier_series(series: pd.Series) -> pd.Series:
    """Cast identifiers to strings without turning integer ids into '123.0'."""
    if pd.api.types.is_numeric_dtype(series):
        num = pd.to_numeric(series, errors="coerce")
        out = pd.Series("", index=series.index, dtype="object")
        finite = num.notna()
        integral = finite & (num.round() == num)
        out.loc[integral] = num.loc[integral].round().astype("int64").astype(str)
        fractional = finite & ~integral
        out.loc[fractional] = num.loc[fractional].astype(str)
        return out
    out = series.astype("object").where(series.notna(), "")
    out = out.map(lambda v: "" if v is None else str(v).strip())
    out = out.mask(out.str.lower().isin(["", "nan", "none", "null", "<na>", "nat"]), "")
    out = out.str.replace(r"^(\d+)\.0$", r"\1", regex=True)
    return out


_UNIT_MICRO = str.maketrans({"µ": "u", "μ": "u"})


def normalize_unit(unit) -> str:
    if unit is None or (isinstance(unit, float) and math.isnan(unit)):
        return ""
    text = str(unit).strip().translate(_UNIT_MICRO)
    if text.lower() in {"", "nan", "none", "null"}:
        return ""
    return text


def normalize_unit_key(unit) -> str:
    return normalize_unit(unit).lower()


_NON_ALNUM = re.compile(r"[^A-Z0-9]")
_CODE_SPLIT = re.compile(r"[,;|]")


def normalize_icd(code) -> str:
    if code is None or (isinstance(code, float) and math.isnan(code)):
        return ""
    text = str(code).strip().upper()
    if text in {"", "NAN", "NONE", "NULL"}:
        return ""
    return _NON_ALNUM.sub("", text)


def split_code_list(raw) -> list[str]:
    """Split a diagnosis field that may contain multiple codes."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    text = str(raw).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return []
    codes = []
    for part in _CODE_SPLIT.split(text):
        norm = normalize_icd(part)
        if norm:
            codes.append(norm)
    return codes


def raw_code_tokens(raw) -> list[str]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return []
    text = str(raw).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return []
    return [part.strip() for part in _CODE_SPLIT.split(text) if part.strip()]


def paired_codes(raw) -> list[tuple[str, str]]:
    """Pair each raw token with its normalized ICD code, dropping empty tokens."""
    pairs = []
    for raw_token in raw_code_tokens(raw):
        normalized = normalize_icd(raw_token)
        if normalized:
            pairs.append((raw_token, normalized))
    return pairs


def parse_icd_system(declared, normalized_code: str) -> str:
    text = "" if declared is None else str(declared).strip().upper()
    text = text.replace("-", "").replace("_", "").replace(" ", "")
    if text in {"9", "ICD9"}:
        return "ICD9"
    if text in {"10", "ICD10"}:
        return "ICD10"
    if not normalized_code:
        return ""
    if normalized_code[0].isalpha():
        return "ICD10"
    if normalized_code[0].isdigit():
        return "ICD9"
    return ""


def infer_system_series(codes: pd.Series) -> pd.Series:
    first = codes.fillna("").astype(str).str[:1]
    system = pd.Series("", index=codes.index, dtype="object")
    system = system.mask(first.str.isalpha(), "ICD10")
    system = system.mask(first.str.isdigit(), "ICD9")
    return system
