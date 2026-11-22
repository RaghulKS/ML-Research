"""Report generation module — produces a Markdown report."""
import pandas as pd
from pathlib import Path

from config import DIRS


def _read_csv(name):
    p = DIRS["tables"] / name
    if p.exists():
        return pd.read_csv(p)
    return None


def generate_report():
    sections = []

    # --- Title ---
    sections.append("# MIMIC-IV Clinical Cohort Analysis Report\n")
    sections.append("_Auto-generated end-to-end exploratory analysis_\n")

    # --- Methods ---
    sections.append("## Methods\n")
    sections.append(
        "The cohort was extracted from the MIMIC-IV database, with one row per "
        "qualifying diagnosis during a hospital admission. The dataset was "
        "loaded into a pandas DataFrame, datatypes were inspected and corrected, "
        "datetime columns were parsed, categorical variables were created, "
        "missing values were handled (text → 'UNKNOWN', ICU LOS NaN where no "
        "ICU stay → 0), and exact duplicate rows were removed. Derived disease "
        "flags (`has_diabetes`, `has_kidney_disease`, `has_both`) were verified "
        "for internal consistency. A composite `disease_group` variable was "
        "created. Descriptive statistics, ICD analyses, ICU and hospital "
        "analyses, and statistical tests (chi-square, independent t-tests, "
        "Mann-Whitney U) were performed using pandas, numpy, scipy, and "
        "statsmodels. All figures were generated with matplotlib and saved as "
        "high-resolution PNGs.\n"
    )

    # --- Data Quality ---
    sections.append("## Data Quality\n")
    ov = _read_csv("data_quality_overview.csv")
    if ov is not None:
        sections.append(ov.to_markdown(index=False))
        sections.append("")
    mv = _read_csv("missing_values.csv")
    if mv is not None and len(mv) > 0:
        sections.append("\n### Missing Values\n")
        sections.append(mv.to_markdown(index=False))
        sections.append("")

    # --- Cohort Summary ---
    sections.append("\n## Cohort Summary\n")
    cc = _read_csv("cohort_counts.csv")
    if cc is not None:
        sections.append(cc.to_markdown(index=False))
        sections.append("")

    # --- Descriptive Statistics ---
    sections.append("\n## Descriptive Statistics\n")
    ns = _read_csv("numeric_summary.csv")
    if ns is not None:
        sections.append("### Numeric Variables\n")
        sections.append(ns.to_markdown(index=False))
        sections.append("")

    # --- ICD Analysis ---
    sections.append("\n## ICD Analysis\n")
    t20 = _read_csv("top20_icd.csv")
    if t20 is not None:
        sections.append("### Top 20 ICD Codes\n")
        sections.append(t20.to_markdown(index=False))
        sections.append("")
    icd_ver = _read_csv("icd_version_distribution.csv")
    if icd_ver is not None:
        sections.append("\n### ICD Version Distribution\n")
        sections.append(icd_ver.to_markdown(index=False))
        sections.append("")

    # --- ICU Analysis ---
    sections.append("\n## ICU Analysis\n")
    icu_rate = _read_csv("icu_admission_rate.csv")
    if icu_rate is not None:
        sections.append(icu_rate.to_markdown(index=False))
        sections.append("")
    icu_los = _read_csv("icu_los_stats.csv")
    if icu_los is not None:
        sections.append(icu_los.to_markdown(index=False))
        sections.append("")

    # --- Hospital Analysis ---
    sections.append("\n## Hospital Analysis\n")
    los_ov = _read_csv("hospital_los_overall.csv")
    if los_ov is not None:
        sections.append(los_ov.to_markdown(index=False))
        sections.append("")
    los_grp = _read_csv("los_by_disease_group.csv")
    if los_grp is not None:
        sections.append("\n### LOS by Disease Group\n")
        sections.append(los_grp.to_markdown(index=False))
        sections.append("")
    mort_grp = _read_csv("mortality_by_disease_group.csv")
    if mort_grp is not None:
        sections.append("\n### Mortality by Disease Group\n")
        sections.append(mort_grp.to_markdown(index=False))
        sections.append("")

    # --- Statistical Testing ---
    sections.append("\n## Statistical Testing\n")
    st = _read_csv("statistical_tests.csv")
    if st is not None:
        sections.append(st.to_markdown(index=False))
        sections.append("")

    # --- Results ---
    sections.append("\n## Results\n")
    sections.append(
        "All tables are saved in `project/tables/` and all figures in "
        "`project/figures/`. Key findings are summarized in the tables above "
        "and visualized in the 20 generated PNG figures.\n"
    )

    # --- Discussion ---
    sections.append("\n## Discussion\n")
    sections.append(
        "This exploratory analysis characterizes a MIMIC-IV cohort of patients "
        "with diabetes, kidney disease, or both. Disease-group comparisons of "
        "length of stay, ICU admission, and mortality reveal the clinical "
        "burden associated with comorbid metabolic and renal conditions. "
        "Statistical tests indicate whether observed differences in mortality, "
        "LOS, and ICU utilization are significant. The ICD analysis highlights "
        "the most prevalent diagnoses and the transition from ICD-9 to ICD-10 "
        "coding across anchor year groups.\n"
    )

    # --- Limitations ---
    sections.append("\n## Limitations\n")
    sections.append(
        "- MIMIC-IV is a single-center dataset from Beth Israel Deaconess "
        "Medical Center; generalizability is limited.\n"
        "- The analysis is observational; no causal inference is attempted.\n"
        "- Disease flags are derived from ICD coding, which may have "
        "misclassification or under-coding.\n"
        "- The dataset is at the diagnosis-row level; admission-level and "
        "patient-level summaries were derived by aggregation.\n"
        "- Anchor dates are shifted; temporal trends should be interpreted "
        "with caution.\n"
    )

    # --- Future Work ---
    sections.append("\n## Future Work\n")
    sections.append(
        "- Multivariable regression models adjusting for age, gender, race, "
        "and comorbidity to estimate independent effects of diabetes and "
        "kidney disease on outcomes.\n"
        "- Time-series analysis of LOS and mortality trends across anchor "
        "year groups.\n"
        "- Subgroup analyses by ICU care unit type and admission type.\n"
        "- Integration of laboratory values and medications for deeper "
        "phenotyping.\n"
        "- Machine-learning prediction of mortality and prolonged LOS.\n"
    )

    # --- Conclusion ---
    sections.append("\n## Conclusion\n")
    sections.append(
        "This end-to-end exploratory pipeline provides a comprehensive, "
        "reproducible characterization of the MIMIC-IV diabetes/kidney-disease "
        "cohort. All data-quality checks, descriptive statistics, ICD, ICU, "
        "and hospital analyses, statistical tests, and visualizations have "
        "been generated programmatically and saved in a structured directory "
        "layout, ready for review by a research mentor.\n"
    )

    report_text = "\n".join(sections)
    report_path = DIRS["report"] / "report.md"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path
