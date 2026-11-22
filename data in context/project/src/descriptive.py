"""Descriptive statistics module."""
import pandas as pd
import numpy as np

from config import DIRS


def descriptive_stats(df):
    tables = {}

    # Numeric summary
    num_cols = ["anchor_age", "hospital_length_of_stay_days",
                "icu_length_of_stay_days", "hospital_expire_flag",
                "icu_admission"]
    num_cols = [c for c in num_cols if c in df.columns]
    num_summary = df[num_cols].describe(percentiles=[.25, .5, .75]).T
    num_summary["median"] = df[num_cols].median()
    tables["numeric_summary"] = num_summary

    # Categorical frequency tables
    cat_cols = ["gender", "admission_type", "insurance", "race",
                "language", "marital_status", "anchor_year_group",
                "first_careunit", "last_careunit"]
    for col in cat_cols:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False)
            pct = (vc / vc.sum() * 100).round(2)
            tables[f"freq_{col}"] = pd.DataFrame({"count": vc, "pct": pct})

    # Mortality
    if "hospital_expire_flag" in df.columns:
        mort = df["hospital_expire_flag"].value_counts(dropna=False)
        tables["mortality"] = pd.DataFrame({
            "count": mort,
            "pct": (mort / mort.sum() * 100).round(2),
        })

    return tables


def save_descriptive(tables):
    for name, tbl in tables.items():
        tbl.to_csv(DIRS["tables"] / f"{name}.csv")
    # Combined summary_statistics.csv
    frames = []
    for name, tbl in tables.items():
        t = tbl.copy()
        t.insert(0, "table", name)
        t.insert(1, "category", t.index.astype(str))
        frames.append(t.reset_index(drop=True))
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(DIRS["tables"] / "summary_statistics.csv", index=False)
