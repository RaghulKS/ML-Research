"""Statistical testing module."""
import pandas as pd
import numpy as np
from scipy import stats

from config import DIRS


def run_tests(df):
    """Run chi-square, t-tests, and Mann-Whitney U where appropriate."""
    results = []
    adm = df.drop_duplicates(subset="hadm_id").copy()

    # --- Chi-square: disease group vs mortality ---
    if "disease_group" in adm.columns and "hospital_expire_flag" in adm.columns:
        ct = pd.crosstab(adm["disease_group"], adm["hospital_expire_flag"])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            results.append(("Chi-square: disease_group vs mortality",
                            f"chi2={chi2:.3f}, dof={dof}, p={p:.3g}"))

    # --- Chi-square: ICU admission vs mortality ---
    if "icu_admission" in adm.columns and "hospital_expire_flag" in adm.columns:
        ct = pd.crosstab(adm["icu_admission"], adm["hospital_expire_flag"])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            results.append(("Chi-square: ICU admission vs mortality",
                            f"chi2={chi2:.3f}, dof={dof}, p={p:.3g}"))

    # --- Chi-square: gender vs mortality ---
    if "gender" in adm.columns:
        ct = pd.crosstab(adm["gender"], adm["hospital_expire_flag"])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            results.append(("Chi-square: gender vs mortality",
                            f"chi2={chi2:.3f}, dof={dof}, p={p:.3g}"))

    # --- Chi-square: disease group vs ICU admission ---
    if "disease_group" in adm.columns and "icu_admission" in adm.columns:
        ct = pd.crosstab(adm["disease_group"], adm["icu_admission"])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            results.append(("Chi-square: disease_group vs ICU admission",
                            f"chi2={chi2:.3f}, dof={dof}, p={p:.3g}"))

    # --- T-test & Mann-Whitney: age by mortality ---
    if "anchor_age" in adm.columns and "hospital_expire_flag" in adm.columns:
        alive = adm.loc[adm["hospital_expire_flag"] == 0, "anchor_age"].dropna()
        dead = adm.loc[adm["hospital_expire_flag"] == 1, "anchor_age"].dropna()
        if len(alive) > 1 and len(dead) > 1:
            t, p_t = stats.ttest_ind(alive, dead, equal_var=False)
            u, p_u = stats.mannwhitneyu(alive, dead, alternative="two-sided")
            results.append(("t-test: age alive vs dead",
                            f"t={t:.3f}, p={p_t:.3g}"))
            results.append(("Mann-Whitney U: age alive vs dead",
                            f"U={u:.1f}, p={p_u:.3g}"))

    # --- T-test & Mann-Whitney: hospital LOS by ICU admission ---
    if "hospital_length_of_stay_days" in adm.columns and "icu_admission" in adm.columns:
        no_icu = adm.loc[adm["icu_admission"] == 0, "hospital_length_of_stay_days"].dropna()
        icu = adm.loc[adm["icu_admission"] == 1, "hospital_length_of_stay_days"].dropna()
        if len(no_icu) > 1 and len(icu) > 1:
            t, p_t = stats.ttest_ind(no_icu, icu, equal_var=False)
            u, p_u = stats.mannwhitneyu(no_icu, icu, alternative="two-sided")
            results.append(("t-test: hospital LOS no-ICU vs ICU",
                            f"t={t:.3f}, p={p_t:.3g}"))
            results.append(("Mann-Whitney U: hospital LOS no-ICU vs ICU",
                            f"U={u:.1f}, p={p_u:.3g}"))

    # --- T-test & Mann-Whitney: hospital LOS by diabetes ---
    if "hospital_length_of_stay_days" in adm.columns and "has_diabetes" in adm.columns:
        no_d = adm.loc[adm["has_diabetes"] == 0, "hospital_length_of_stay_days"].dropna()
        d = adm.loc[adm["has_diabetes"] == 1, "hospital_length_of_stay_days"].dropna()
        if len(no_d) > 1 and len(d) > 1:
            t, p_t = stats.ttest_ind(no_d, d, equal_var=False)
            u, p_u = stats.mannwhitneyu(no_d, d, alternative="two-sided")
            results.append(("t-test: hospital LOS no-diabetes vs diabetes",
                            f"t={t:.3f}, p={p_t:.3g}"))
            results.append(("Mann-Whitney U: hospital LOS no-diabetes vs diabetes",
                            f"U={u:.1f}, p={p_u:.3g}"))

    # --- T-test & Mann-Whitney: hospital LOS by kidney disease ---
    if "hospital_length_of_stay_days" in adm.columns and "has_kidney_disease" in adm.columns:
        no_k = adm.loc[adm["has_kidney_disease"] == 0, "hospital_length_of_stay_days"].dropna()
        k = adm.loc[adm["has_kidney_disease"] == 1, "hospital_length_of_stay_days"].dropna()
        if len(no_k) > 1 and len(k) > 1:
            t, p_t = stats.ttest_ind(no_k, k, equal_var=False)
            u, p_u = stats.mannwhitneyu(no_k, k, alternative="two-sided")
            results.append(("t-test: hospital LOS no-kidney vs kidney",
                            f"t={t:.3f}, p={p_t:.3g}"))
            results.append(("Mann-Whitney U: hospital LOS no-kidney vs kidney",
                            f"U={u:.1f}, p={p_u:.3g}"))

    table = pd.DataFrame(results, columns=["test", "result"])
    return table


def save_stats(table):
    table.to_csv(DIRS["tables"] / "statistical_tests.csv", index=False)
