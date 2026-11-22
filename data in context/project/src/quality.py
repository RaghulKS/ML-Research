"""Data quality report module."""
import pandas as pd

from config import DIRS


def quality_report(df):
    """Return a dict of data-quality tables."""
    n_rows = len(df)
    n_patients = df["subject_id"].nunique() if "subject_id" in df.columns else None
    n_admissions = df["hadm_id"].nunique() if "hadm_id" in df.columns else None
    n_dup = int(df.duplicated().sum())

    missing = df.isna().sum()
    missing_pct = (missing / n_rows * 100).round(2)
    missing_table = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct,
    })
    missing_table = missing_table[missing_table["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )

    dtype_summary = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "non_null": df.notna().sum().values,
        "n_unique": df.nunique(dropna=True).values,
    }).set_index("column")

    summary = {
        "n_rows": n_rows,
        "n_patients": n_patients,
        "n_admissions": n_admissions,
        "n_duplicates": n_dup,
        "missing_table": missing_table,
        "dtype_summary": dtype_summary,
    }
    return summary


def save_quality(summary):
    summary["missing_table"].to_csv(DIRS["tables"] / "missing_values.csv")
    summary["dtype_summary"].to_csv(DIRS["tables"] / "datatype_summary.csv")
    overview = pd.DataFrame({
        "metric": ["rows", "unique_patients", "unique_admissions", "duplicate_rows"],
        "value": [summary["n_rows"], summary["n_patients"],
                  summary["n_admissions"], summary["n_duplicates"]],
    })
    overview.to_csv(DIRS["tables"] / "data_quality_overview.csv", index=False)
