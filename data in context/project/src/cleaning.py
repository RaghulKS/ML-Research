"""Data cleaning module: load, inspect, clean, derive variables."""
import pandas as pd
import numpy as np

from config import RAW_CSV, CLEANED_CSV, DATETIME_COLS, DISEASE_FLAGS


def load_raw():
    """Load the raw CSV with sensible dtypes."""
    df = pd.read_csv(RAW_CSV, low_memory=False)
    return df


def inspect_dtypes(df):
    """Return a dataframe describing dtypes and non-null counts."""
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "null": df.isna().sum(),
        "n_unique": df.nunique(dropna=True),
    })
    return summary


def clean(df):
    """Perform full cleaning pipeline and return cleaned dataframe."""
    df = df.copy()

    # --- Datetime conversion ---
    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # --- Numeric coercion for derived flags / measures ---
    for col in DISEASE_FLAGS + [
        "hospital_expire_flag",
        "hospital_length_of_stay_days",
        "icu_length_of_stay_days",
        "icu_admission",
        "anchor_age",
        "anchor_year",
        "icd_version",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Categorical conversion ---
    cat_cols = [
        "gender", "anchor_year_group", "admission_type", "admission_location",
        "discharge_location", "insurance", "language", "marital_status",
        "race", "icd_version", "first_careunit", "last_careunit",
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # --- Handle missing values ---
    # Text columns -> "UNKNOWN"
    text_fill = ["admission_location", "discharge_location", "language",
                 "marital_status", "race", "insurance", "first_careunit",
                 "last_careunit"]
    for col in text_fill:
        if col in df.columns:
            if df[col].dtype.name == "category":
                if "UNKNOWN" not in df[col].cat.categories:
                    df[col] = df[col].cat.add_categories(["UNKNOWN"])
            df[col] = df[col].fillna("UNKNOWN")

    # Numeric LOS: leave NaN (means no ICU stay) but keep as is for stats
    # icu_length_of_stay_days NaN -> 0 only when icu_admission == 0
    if "icu_length_of_stay_days" in df.columns and "icu_admission" in df.columns:
        mask = (df["icu_admission"] == 0) & (df["icu_length_of_stay_days"].isna())
        df.loc[mask, "icu_length_of_stay_days"] = 0

    # --- Remove exact duplicate rows ---
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)

    # --- Verify derived flags ---
    df = _verify_flags(df)

    # --- Create clean categorical disease group ---
    df["disease_group"] = _make_disease_group(df)
    df["disease_group"] = df["disease_group"].astype("category")

    return df, removed


def _verify_flags(df):
    """Verify and fix has_both consistency."""
    if all(c in df.columns for c in DISEASE_FLAGS):
        # has_both should be 1 when both diabetes and kidney disease are 1
        expected = ((df["has_diabetes"] == 1) & (df["has_kidney_disease"] == 1)).astype(int)
        mismatch = (df["has_both"] != expected).sum()
        if mismatch > 0:
            df["has_both"] = expected
    return df


def _make_disease_group(df):
    """Create a single categorical disease-group column."""
    def _row(r):
        d = r.get("has_diabetes", 0) == 1
        k = r.get("has_kidney_disease", 0) == 1
        if d and k:
            return "Both"
        if d:
            return "Diabetes only"
        if k:
            return "Kidney disease only"
        return "Neither"
    return df.apply(_row, axis=1)


def save_clean(df):
    df.to_csv(CLEANED_CSV, index=False)
