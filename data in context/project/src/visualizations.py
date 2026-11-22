"""Visualization module — all figures use matplotlib only."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DIRS, FIG_DPI, plt_rc

plt.rcParams.update(plt_rc)

PALETTE = {
    "blue": "#4C72B0",
    "orange": "#DD8452",
    "green": "#55A868",
    "red": "#C44E52",
    "purple": "#8172B3",
    "gray": "#937860",
}


def _save(fig, name):
    path = DIRS["figures"] / f"{name}.png"
    fig.savefig(path, dpi=FIG_DPI, format="png", bbox_inches="tight")
    plt.close(fig)
    return path


def _bar(series, title, xlabel, ylabel="Count", color=None, rotation=45, top=None):
    s = series.value_counts(dropna=False)
    if top:
        s = s.head(top)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(s.index.astype(str), s.values, color=color or PALETTE["blue"])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    for i, v in enumerate(s.values):
        ax.text(i, v + s.max() * 0.01, str(v), ha="center", fontsize=8)
    fig.tight_layout()
    return fig


def _hist(series, title, xlabel, ylabel="Frequency", bins=50, color=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    data = series.dropna()
    ax.hist(data, bins=bins, color=color or PALETTE["blue"], edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def generate_all(df):
    paths = []

    # 1. Cohort counts bar chart
    pat = df.groupby("subject_id")[["has_diabetes", "has_kidney_disease", "has_both"]].max()
    counts = {
        "Diabetes": int((pat["has_diabetes"] == 1).sum()),
        "Kidney Disease": int((pat["has_kidney_disease"] == 1).sum()),
        "Both": int((pat["has_both"] == 1).sum()),
        "Neither": int(((pat["has_diabetes"] == 0) & (pat["has_kidney_disease"] == 0)).sum()),
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(counts.keys(), counts.values(),
           color=[PALETTE["blue"], PALETTE["orange"], PALETTE["red"], PALETTE["gray"]])
    ax.set_title("Cohort Counts by Disease Group")
    ax.set_ylabel("Number of Patients")
    for i, v in enumerate(counts.values()):
        ax.text(i, v + max(counts.values()) * 0.01, str(v), ha="center")
    fig.tight_layout()
    paths.append(_save(fig, "01_cohort_counts"))

    # 2. Disease overlap (Venn-style bar)
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["Diabetes only", "Kidney only", "Both", "Neither"]
    vals = [
        int(((pat["has_diabetes"] == 1) & (pat["has_kidney_disease"] == 0)).sum()),
        int(((pat["has_diabetes"] == 0) & (pat["has_kidney_disease"] == 1)).sum()),
        int((pat["has_both"] == 1).sum()),
        int(((pat["has_diabetes"] == 0) & (pat["has_kidney_disease"] == 0)).sum()),
    ]
    ax.barh(labels, vals, color=[PALETTE["blue"], PALETTE["orange"], PALETTE["red"], PALETTE["gray"]])
    ax.set_title("Disease Overlap")
    ax.set_xlabel("Number of Patients")
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, str(v), va="center")
    fig.tight_layout()
    paths.append(_save(fig, "02_disease_overlap"))

    # 3. Age histogram
    paths.append(_save(_hist(df["anchor_age"], "Age Distribution", "Age (years)"),
                       "03_age_histogram"))

    # 4. Gender distribution
    paths.append(_save(_bar(df["gender"], "Gender Distribution", "Gender",
                            color=PALETTE["purple"]), "04_gender_distribution"))

    # 5. Race distribution
    paths.append(_save(_bar(df["race"], "Race Distribution", "Race", rotation=60, top=15),
                       "05_race_distribution"))

    # 6. Insurance distribution
    paths.append(_save(_bar(df["insurance"], "Insurance Distribution", "Insurance"),
                       "06_insurance_distribution"))

    # 7. Admission type distribution
    paths.append(_save(_bar(df["admission_type"], "Admission Type Distribution",
                            "Admission Type", rotation=60),
                       "07_admission_type_distribution"))

    # 8. Hospital LOS histogram
    paths.append(_save(_hist(df.drop_duplicates(subset="hadm_id")["hospital_length_of_stay_days"],
                             "Hospital Length of Stay Distribution",
                             "LOS (days)", bins=60, color=PALETTE["green"]),
                       "08_hospital_los_histogram"))

    # 9. ICU LOS histogram
    icu_los = df.loc[df["icu_admission"] == 1, "icu_length_of_stay_days"].dropna()
    paths.append(_save(_hist(icu_los, "ICU Length of Stay Distribution",
                             "ICU LOS (days)", bins=60, color=PALETTE["orange"]),
                       "09_icu_los_histogram"))

    # 10. Mortality by disease group
    adm = df.drop_duplicates(subset="hadm_id")
    mort = adm.groupby("disease_group")["hospital_expire_flag"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(mort.index.astype(str), mort.values, color=PALETTE["red"])
    ax.set_title("Mortality Rate by Disease Group")
    ax.set_ylabel("Mortality Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(mort.values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    paths.append(_save(fig, "10_mortality_by_disease_group"))

    # 11. ICU admission rates
    icu_rate = adm.groupby("disease_group")["icu_admission"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(icu_rate.index.astype(str), icu_rate.values, color=PALETTE["blue"])
    ax.set_title("ICU Admission Rate by Disease Group")
    ax.set_ylabel("ICU Admission Rate (%)")
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(icu_rate.values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    paths.append(_save(fig, "11_icu_admission_rates"))

    # 12. Top 20 ICD codes
    top20 = df["icd_code"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top20.index[::-1], top20.values[::-1], color=PALETTE["blue"])
    ax.set_title("Top 20 ICD Codes")
    ax.set_xlabel("Count")
    fig.tight_layout()
    paths.append(_save(fig, "12_top20_icd_codes"))

    # 13. Top diabetes ICD codes
    diab = df[df["has_diabetes"] == 1]
    top_d = diab["icd_code"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top_d.index[::-1], top_d.values[::-1], color=PALETTE["orange"])
    ax.set_title("Top 20 Diabetes ICD Codes")
    ax.set_xlabel("Count")
    fig.tight_layout()
    paths.append(_save(fig, "13_top_diabetes_icd_codes"))

    # 14. Top kidney ICD codes
    kid = df[df["has_kidney_disease"] == 1]
    top_k = kid["icd_code"].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top_k.index[::-1], top_k.values[::-1], color=PALETTE["green"])
    ax.set_title("Top 20 Kidney Disease ICD Codes")
    ax.set_xlabel("Count")
    fig.tight_layout()
    paths.append(_save(fig, "14_top_kidney_icd_codes"))

    # 15. ICD9 vs ICD10 comparison
    icd_dist = df["icd_version"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar([f"ICD-{int(v)}" for v in icd_dist.index], icd_dist.values,
           color=[PALETTE["blue"], PALETTE["orange"]])
    ax.set_title("ICD-9 vs ICD-10 Distribution")
    ax.set_ylabel("Count")
    for i, v in enumerate(icd_dist.values):
        ax.text(i, v + icd_dist.max() * 0.01, str(v), ha="center")
    fig.tight_layout()
    paths.append(_save(fig, "15_icd9_vs_icd10"))

    # 16. First ICU care units
    first_cu = df[df["icu_admission"] == 1]["first_careunit"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(first_cu.index.astype(str), first_cu.values, color=PALETTE["purple"])
    ax.set_title("First ICU Care Unit Distribution")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    paths.append(_save(fig, "16_first_careunit"))

    # 17. Last ICU care units
    last_cu = df[df["icu_admission"] == 1]["last_careunit"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(last_cu.index.astype(str), last_cu.values, color=PALETTE["gray"])
    ax.set_title("Last ICU Care Unit Distribution")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    paths.append(_save(fig, "17_last_careunit"))

    # 18. Anchor year group distribution
    paths.append(_save(_bar(df["anchor_year_group"], "Anchor Year Group Distribution",
                            "Year Group", rotation=30),
                       "18_anchor_year_group"))

    # 19. Heatmap of missing values
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(missing.values.reshape(1, -1), aspect="auto", cmap="Reds")
        ax.set_xticks(range(len(missing)))
        ax.set_xticklabels(missing.index, rotation=45, ha="right")
        ax.set_yticks([0])
        ax.set_yticklabels(["Missing"])
        ax.set_title("Missing Values Heatmap")
        for i, v in enumerate(missing.values):
            ax.text(i, 0, str(v), ha="center", va="center", fontsize=8,
                    color="white" if v > missing.max() * 0.5 else "black")
        fig.colorbar(im, ax=ax, label="Missing Count")
        fig.tight_layout()
        paths.append(_save(fig, "19_missing_values_heatmap"))

    # 20. Correlation heatmap for numeric variables
    num_cols = ["anchor_age", "hospital_length_of_stay_days",
                "icu_length_of_stay_days", "hospital_expire_flag",
                "icu_admission", "has_diabetes", "has_kidney_disease", "has_both"]
    num_cols = [c for c in num_cols if c in df.columns]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="right")
    ax.set_yticks(range(len(num_cols)))
    ax.set_yticklabels(num_cols)
    ax.set_title("Correlation Heatmap (Numeric Variables)")
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    paths.append(_save(fig, "20_correlation_heatmap"))

    return paths
