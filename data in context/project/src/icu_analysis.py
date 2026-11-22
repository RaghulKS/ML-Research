"""ICU analysis module."""
import pandas as pd
import numpy as np

from config import DIRS


def icu_analysis(df):
    tables = {}

    # ICU admission rate (admission-level)
    adm = df.drop_duplicates(subset="hadm_id")
    icu_rate = adm["icu_admission"].mean()
    tables["icu_admission_rate"] = pd.DataFrame({
        "metric": ["icu_admission_rate", "n_admissions", "n_icu_admissions"],
        "value": [round(icu_rate, 4), len(adm), int(adm["icu_admission"].sum())],
    })

    # ICU LOS stats (only those with ICU stay)
    icu_stays = df[df["icu_admission"] == 1]["icu_length_of_stay_days"].dropna()
    tables["icu_los_stats"] = pd.DataFrame({
        "metric": ["mean", "median", "std", "min", "max", "n"],
        "value": [icu_stays.mean(), icu_stays.median(), icu_stays.std(),
                  icu_stays.min(), icu_stays.max(), len(icu_stays)],
    })

    # Distribution of first ICU care units
    if "first_careunit" in df.columns:
        first_cu = df[df["icu_admission"] == 1]["first_careunit"].value_counts(dropna=False)
        tables["first_careunit_distribution"] = pd.DataFrame({
            "first_careunit": first_cu.index.astype(str),
            "count": first_cu.values,
        })

    # Distribution of last ICU care units
    if "last_careunit" in df.columns:
        last_cu = df[df["icu_admission"] == 1]["last_careunit"].value_counts(dropna=False)
        tables["last_careunit_distribution"] = pd.DataFrame({
            "last_careunit": last_cu.index.astype(str),
            "count": last_cu.values,
        })

    return tables


def save_icu(tables):
    for name, tbl in tables.items():
        tbl.to_csv(DIRS["tables"] / f"{name}.csv", index=False)
