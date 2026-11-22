"""Hospital analysis module."""
import pandas as pd
import numpy as np

from config import DIRS


def hospital_analysis(df):
    tables = {}

    # Admission-level dataframe
    adm = df.drop_duplicates(subset="hadm_id").copy()

    # Overall hospital LOS
    los = adm["hospital_length_of_stay_days"].dropna()
    tables["hospital_los_overall"] = pd.DataFrame({
        "metric": ["mean", "median", "std", "min", "max", "n"],
        "value": [los.mean(), los.median(), los.std(),
                  los.min(), los.max(), len(los)],
    })

    # LOS by disease group
    if "disease_group" in adm.columns:
        los_by_grp = adm.groupby("disease_group")["hospital_length_of_stay_days"].agg(
            ["mean", "median", "std", "count"]
        ).round(2)
        tables["los_by_disease_group"] = los_by_grp

    # LOS by ICU admission
    los_icu = adm.groupby("icu_admission")["hospital_length_of_stay_days"].agg(
        ["mean", "median", "std", "count"]
    ).round(2)
    tables["los_by_icu_admission"] = los_icu

    # Mortality by disease group
    if "disease_group" in adm.columns:
        mort_grp = adm.groupby("disease_group")["hospital_expire_flag"].agg(
            ["sum", "count"]
        )
        mort_grp["mortality_rate"] = (mort_grp["sum"] / mort_grp["count"] * 100).round(2)
        mort_grp.columns = ["deaths", "n", "mortality_rate_pct"]
        tables["mortality_by_disease_group"] = mort_grp

    # Mortality by ICU admission
    mort_icu = adm.groupby("icu_admission")["hospital_expire_flag"].agg(["sum", "count"])
    mort_icu["mortality_rate"] = (mort_icu["sum"] / mort_icu["count"] * 100).round(2)
    mort_icu.columns = ["deaths", "n", "mortality_rate_pct"]
    tables["mortality_by_icu_admission"] = mort_icu

    return tables


def save_hospital(tables):
    for name, tbl in tables.items():
        tbl.to_csv(DIRS["tables"] / f"{name}.csv")
