"""Cohort summary module."""
import pandas as pd

from config import DIRS


def cohort_summary(df):
    n_patients = df["subject_id"].nunique()
    n_admissions = df["hadm_id"].nunique()

    # Patient-level disease flags (max per subject collapses duplicates)
    pat = df.groupby("subject_id")[["has_diabetes", "has_kidney_disease", "has_both"]].max()
    n_diab = int((pat["has_diabetes"] == 1).sum())
    n_kid = int((pat["has_kidney_disease"] == 1).sum())
    n_both = int((pat["has_both"] == 1).sum())

    rows = [
        ("Total unique patients", n_patients, 100.0),
        ("Total admissions", n_admissions, round(n_admissions / n_patients * 100, 2)),
        ("Patients with diabetes", n_diab, round(n_diab / n_patients * 100, 2)),
        ("Patients with kidney disease", n_kid, round(n_kid / n_patients * 100, 2)),
        ("Patients with both diseases", n_both, round(n_both / n_patients * 100, 2)),
    ]
    table = pd.DataFrame(rows, columns=["metric", "count", "pct_of_patients"])
    return table


def save_cohort(table):
    table.to_csv(DIRS["tables"] / "cohort_counts.csv", index=False)
