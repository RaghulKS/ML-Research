"""
MIMIC-IV Clinical Cohort Analysis — Main Pipeline
==================================================
Run:  python main.py
"""
import sys
import time
import pandas as pd

import config
from config import ensure_dirs

import cleaning
import quality
import cohort
import descriptive
import icd_analysis
import icu_analysis
import hospital_analysis
import stats
import visualizations
import report_gen


def main():
    t0 = time.time()
    ensure_dirs()
    print("=" * 70)
    print("MIMIC-IV Clinical Cohort Analysis Pipeline")
    print("=" * 70)

    # --- 1. Data Cleaning ---
    print("\n[1/10] Loading and cleaning data ...")
    raw = cleaning.load_raw()
    print(f"  Raw rows: {len(raw):,}")
    df, n_dup_removed = cleaning.clean(raw)
    print(f"  Duplicates removed: {n_dup_removed:,}")
    print(f"  Cleaned rows: {len(df):,}")
    cleaning.save_clean(df)
    print(f"  Saved cleaned dataset -> {config.CLEANED_CSV}")

    # --- 2. Data Quality Report ---
    print("\n[2/10] Generating data quality report ...")
    q = quality.quality_report(df)
    quality.save_quality(q)
    print(f"  Rows: {q['n_rows']:,}  Patients: {q['n_patients']:,}  "
          f"Admissions: {q['n_admissions']:,}  Duplicates: {q['n_duplicates']}")

    # --- 3. Cohort Summary ---
    print("\n[3/10] Generating cohort summary ...")
    cc = cohort.cohort_summary(df)
    cohort.save_cohort(cc)
    print(cc.to_string(index=False))

    # --- 4. Descriptive Statistics ---
    print("\n[4/10] Computing descriptive statistics ...")
    desc = descriptive.descriptive_stats(df)
    descriptive.save_descriptive(desc)
    print(f"  Generated {len(desc)} descriptive tables")

    # --- 5. ICD Analysis ---
    print("\n[5/10] Performing ICD analysis ...")
    icd_tbls = icd_analysis.icd_analysis(df)
    icd_analysis.save_icd(icd_tbls)
    print(f"  Generated {len(icd_tbls)} ICD tables")

    # --- 6. ICU Analysis ---
    print("\n[6/10] Performing ICU analysis ...")
    icu_tbls = icu_analysis.icu_analysis(df)
    icu_analysis.save_icu(icu_tbls)
    print(f"  Generated {len(icu_tbls)} ICU tables")

    # --- 7. Hospital Analysis ---
    print("\n[7/10] Performing hospital analysis ...")
    hosp_tbls = hospital_analysis.hospital_analysis(df)
    hospital_analysis.save_hospital(hosp_tbls)
    print(f"  Generated {len(hosp_tbls)} hospital tables")

    # --- 8. Visualizations ---
    print("\n[8/10] Generating visualizations ...")
    fig_paths = visualizations.generate_all(df)
    print(f"  Saved {len(fig_paths)} figures to {config.DIRS['figures']}")

    # --- 9. Statistical Testing ---
    print("\n[9/10] Running statistical tests ...")
    stat_tbl = stats.run_tests(df)
    stats.save_stats(stat_tbl)
    print(stat_tbl.to_string(index=False))

    # --- 10. Report ---
    print("\n[10/10] Generating Markdown report ...")
    report_path = report_gen.generate_report()
    print(f"  Report saved -> {report_path}")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"  Cleaned data:  {config.CLEANED_CSV}")
    print(f"  Tables:        {config.DIRS['tables']}")
    print(f"  Figures:       {config.DIRS['figures']}")
    print(f"  Report:        {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
