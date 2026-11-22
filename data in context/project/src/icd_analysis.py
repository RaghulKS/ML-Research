"""ICD analysis module."""
import pandas as pd

from config import DIRS


def icd_analysis(df):
    tables = {}

    # Top 20 ICD codes overall
    top20 = df["icd_code"].value_counts().head(20)
    top20_df = pd.DataFrame({
        "icd_code": top20.index,
        "count": top20.values,
    })
    # attach long_title
    title_map = df.groupby("icd_code")["long_title"].first()
    top20_df["long_title"] = top20_df["icd_code"].map(title_map)
    tables["top20_icd"] = top20_df

    # Top diabetes codes
    diab = df[df["has_diabetes"] == 1]
    top_diab = diab["icd_code"].value_counts().head(20)
    top_diab_df = pd.DataFrame({
        "icd_code": top_diab.index,
        "count": top_diab.values,
    })
    top_diab_df["long_title"] = top_diab_df["icd_code"].map(title_map)
    tables["top_diabetes_icd"] = top_diab_df

    # Top kidney disease codes
    kid = df[df["has_kidney_disease"] == 1]
    top_kid = kid["icd_code"].value_counts().head(20)
    top_kid_df = pd.DataFrame({
        "icd_code": top_kid.index,
        "count": top_kid.values,
    })
    top_kid_df["long_title"] = top_kid_df["icd_code"].map(title_map)
    tables["top_kidney_icd"] = top_kid_df

    # Most frequent diagnosis descriptions
    top_titles = df["long_title"].value_counts().head(20)
    tables["top_titles"] = pd.DataFrame({
        "long_title": top_titles.index,
        "count": top_titles.values,
    })

    # ICD-9 vs ICD-10 distribution
    icd_dist = df["icd_version"].value_counts(dropna=False)
    tables["icd_version_distribution"] = pd.DataFrame({
        "icd_version": icd_dist.index,
        "count": icd_dist.values,
        "pct": (icd_dist.values / icd_dist.sum() * 100).round(2),
    })

    return tables


def save_icd(tables):
    for name, tbl in tables.items():
        tbl.to_csv(DIRS["tables"] / f"{name}.csv", index=False)
