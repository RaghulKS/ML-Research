"""
Simplified Patient Count Report
================================
Counts unique patients in each ICD code category and writes
a short, easy-to-understand report explaining every number.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from config import CLEANED_CSV, DIRS


def main():
    df = pd.read_csv(CLEANED_CSV, low_memory=False)

    # Ensure icd_code is string for prefix matching
    df["icd_code"] = df["icd_code"].astype(str).str.strip().str.upper()

    # ── Helper: count unique patients matching an ICD predicate ──
    def count_patients(mask, label):
        n = df.loc[mask, "subject_id"].nunique()
        return label, n

    results = []

    # ── ICD-10 Diabetes Mellitus: E11 prefix ──
    is_icd10 = df["icd_version"] == 10
    is_icd9  = df["icd_version"] == 9

    mask_e11 = is_icd10 & df["icd_code"].str.startswith("E11")
    results.append(count_patients(mask_e11,
        "Patients with ICD-10 E11 (Type 2 Diabetes Mellitus) codes"))

    # ── ICD-10 Kidney diseases: N17, N18, N19 prefixes ──
    mask_n17_19 = is_icd10 & df["icd_code"].str.match(r"^(N17|N18|N19)")
    results.append(count_patients(mask_n17_19,
        "Patients with ICD-10 N17/N18/N19 (Kidney disease) codes"))

    # ── ICD-9 Kidney disease: 584, 585 prefixes ──
    mask_584_585 = is_icd9 & df["icd_code"].str.match(r"^(584|585)")
    results.append(count_patients(mask_584_585,
        "Patients with ICD-9 584/585 (Acute/Chronic kidney disease) codes"))

    # ── ICD-9 Diabetes Mellitus: 250 prefix ──
    mask_250 = is_icd9 & df["icd_code"].str.startswith("250")
    results.append(count_patients(mask_250,
        "Patients with ICD-9 250 (Diabetes Mellitus) codes"))

    # ── Disease flag columns (already derived in the dataset) ──
    pat = df.groupby("subject_id")[["has_diabetes", "has_kidney_disease", "has_both"]].max()
    results.append(("Patients flagged has_diabetes = 1 (any ICD version)", int((pat["has_diabetes"] == 1).sum())))
    results.append(("Patients flagged has_kidney_disease = 1 (any ICD version)", int((pat["has_kidney_disease"] == 1).sum())))
    results.append(("Patients flagged has_both = 1 (diabetes AND kidney disease)", int((pat["has_both"] == 1).sum())))
    results.append(("Total unique patients in dataset", int(pat.shape[0])))

    # ── Build the report ──
    lines = []
    lines.append("# Simplified Patient Count Report\n")
    lines.append("## Patient Counts by ICD Code Category\n")
    lines.append("| Category | Unique Patients |")
    lines.append("|----------|----------------:|")
    for label, n in results:
        lines.append(f"| {label} | {n:,} |")
    lines.append("")

    # ── Explanatory paragraphs ──
    lines.append("## How Each Number Was Found\n")

    lines.append(
        f"**Total unique patients ({results[-1][1]:,}):** "
        f"The cleaned dataset contains one row per diagnosis code per hospital "
        f"admission.  To count individual patients we grouped all rows by "
        f"`subject_id` (the unique patient identifier in MIMIC-IV) and counted "
        f"the number of distinct groups.  This gives the total number of unique "
        f"human beings in the cohort.\n"
    )

    lines.append(
        f"**ICD-10 E11 – Type 2 Diabetes Mellitus ({results[0][1]:,} patients):** "
        f"We filtered rows where `icd_version` equals 10 (ICD-10) and the "
        f"`icd_code` column starts with the prefix 'E11'.  ICD-10 uses the E11 "
        f"block for non-insulin-dependent (Type 2) diabetes mellitus, with "
        f"sub-codes like E11.0, E11.1, E11.22, etc.  We then counted the number "
        f"of distinct `subject_id` values among those rows so that a patient "
        f"with multiple E11 sub-codes is counted only once.\n"
    )

    lines.append(
        f"**ICD-10 N17/N18/N19 – Kidney diseases ({results[1][1]:,} patients):** "
        f"We filtered rows where `icd_version` equals 10 and `icd_code` starts "
        f"with N17 (acute kidney failure), N18 (chronic kidney disease), or "
        f"N19 (unspecified kidney failure).  These three ICD-10 blocks together "
        f"cover the major kidney-disease diagnoses.  Distinct `subject_id` "
        f"values were counted so each patient appears once regardless of how "
        f"many N17/N18/N19 sub-codes they have.\n"
    )

    lines.append(
        f"**ICD-9 584/585 – Kidney disease ({results[2][1]:,} patients):** "
        f"Older records in MIMIC-IV use ICD-9-CM codes.  Code 584 covers acute "
        f"renal failure and 585 covers chronic kidney disease.  We filtered "
        f"rows where `icd_version` equals 9 and `icd_code` starts with '584' "
        f"or '585', then counted distinct patients.\n"
    )

    lines.append(
        f"**ICD-9 250 – Diabetes Mellitus ({results[3][1]:,} patients):** "
        f"In ICD-9-CM the 250 code block represents diabetes mellitus (with "
        f"sub-codes 250.00, 250.01, 250.40, etc. distinguishing type and "
        f"complications).  We filtered rows where `icd_version` equals 9 and "
        f"`icd_code` starts with '250', then counted distinct `subject_id` "
        f"values.\n"
    )

    lines.append(
        f"**has_diabetes flag ({results[4][1]:,} patients):** "
        f"The dataset ships with a pre-derived binary column called "
        f"`has_diabetes`.  Because a patient can appear on many rows (one per "
        f"diagnosis), we grouped by `subject_id` and took the maximum value of "
        f"`has_diabetes` for each patient.  If any row for that patient had a "
        f"1, the patient is counted as having diabetes.  This flag may capture "
        f"both ICD-9 250xx and ICD-10 E11xx codes (and possibly E10/E13/E14 "
        f"depending on how it was originally derived).\n"
    )

    lines.append(
        f"**has_kidney_disease flag ({results[5][1]:,} patients):** "
        f"Similarly, we grouped by `subject_id` and took the maximum of the "
        f"`has_kidney_disease` column.  Any patient with at least one row "
        f"flagged 1 is counted.  This flag may include ICD-9 codes 584/585 "
        f"and ICD-10 codes N17/N18/N19 (and possibly others like N20 or "
        f"586 depending on the original derivation logic).\n"
    )

    lines.append(
        f"**has_both flag ({results[6][1]:,} patients):** "
        f"We grouped by `subject_id` and took the maximum of the `has_both` "
        f"column.  This flag is 1 only when a patient has both diabetes and "
        f"kidney disease.  During cleaning we verified that `has_both` equals 1 "
        f"exactly when both `has_diabetes` and `has_kidney_disease` are 1 for "
        f"the same patient.\n"
    )

    lines.append(
        "**Why ICD-code counts and flag counts may differ:** "
        "The pre-derived `has_diabetes` and `has_kidney_disease` flags may "
        "include additional ICD codes beyond the specific prefixes listed above "
        "(e.g., E10 for Type 1 diabetes, N20 for kidney stones, or 586 for "
        "unspecified renal failure).  Therefore the flag-based counts can be "
        "higher than the specific-code-prefix counts.  The prefix-based counts "
        "are the most conservative — they include only patients who have at "
        "least one diagnosis code starting with the exact prefix specified.\n"
    )

    report_text = "\n".join(lines)
    out_path = DIRS["report"] / "simple_patient_counts.md"
    out_path.write_text(report_text, encoding="utf-8")
    print(f"Report saved -> {out_path}\n")
    print(report_text)


if __name__ == "__main__":
    main()
