"""SQL renderings of the same rules the Python matcher uses.

These strings are phenotype documentation. Nothing in this module connects to
BigQuery, MIMIC-IV, CRDW, Amarel, or any other database.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from clinical_ml.cohorts.definitions import DEFINITIONS, CohortDefinition

HEADER = """-- AUTO-GENERATED from clinical_ml.cohorts.definitions.
-- Reproducible phenotype logic only. This statement was not executed against
-- MIMIC-IV, BigQuery, CRDW, Amarel, or any other clinical system.
-- Historical phenotypes are narrower than expanded phenotypes. Do not replace
-- previously reported historical counts with expanded counts.
-- CRDW is a hypertension-selected extract. Its counts are not general-population
-- prevalence and are not directly comparable to MIMIC-IV counts.
"""


def _predicate(defn: CohortDefinition, code_sql: str, system_sql: str) -> str:
    parts = []
    for rule in defn.rules:
        system = rule.system
        if rule.match == "exact":
            test = f"{code_sql} = '{rule.normalized}'"
        else:
            test = f"{code_sql} LIKE '{rule.normalized}%'"
        parts.append(f"({system_sql} = '{system}' AND {code_sql} != '' AND {test})")
    return "\n   OR ".join(parts)


def render_mimic_sql(defn: CohortDefinition) -> str:
    code_sql = "REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '')"
    # icd_version is numeric in MIMIC. Map it before comparing to ICD9/ICD10.
    system_sql = "CASE WHEN icd_version = 9 THEN 'ICD9' WHEN icd_version = 10 THEN 'ICD10' ELSE '' END"
    where = _predicate(defn, code_sql, system_sql)
    return (
        HEADER
        + f"-- Cohort: {defn.version}\n"
        + f"-- {defn.description}\n"
        + "SELECT DISTINCT subject_id AS patient_id\n"
        + "FROM diagnoses_icd\n"
        + f"WHERE {where};\n"
    )


def render_crdw_sql(defn: CohortDefinition) -> str:
    code_sql = "icd"
    system_sql = "coding_system"
    where = _predicate(defn, code_sql, system_sql)
    return (
        HEADER
        + f"-- Cohort: {defn.version}\n"
        + f"-- {defn.description}\n"
        + "-- CRDW lists may be comma-separated. A LIKE against the raw list\n"
        + "-- column can match a fragment of a neighboring code. Split first.\n"
        + "-- Dialect sketch: BigQuery. Replace `comorbidities` with the extract table.\n"
        + "WITH tokens AS (\n"
        + "  SELECT PAT_MRN_ID_ENCRYPT AS patient_id,\n"
        + "         PAT_ENC_CSN_ID_ENCRYPT AS encounter_id,\n"
        + "         'ICD10' AS coding_system,\n"
        + "         code AS raw_code\n"
        + "  FROM comorbidities,\n"
        + "  UNNEST(SPLIT(IFNULL(CURRENT_ICD10_LIST, ''), ',')) AS code\n"
        + "  UNION ALL\n"
        + "  SELECT PAT_MRN_ID_ENCRYPT,\n"
        + "         PAT_ENC_CSN_ID_ENCRYPT,\n"
        + "         'ICD9',\n"
        + "         code\n"
        + "  FROM comorbidities,\n"
        + "  UNNEST(SPLIT(IFNULL(CURRENT_ICD9_LIST, ''), ',')) AS code\n"
        + "), normalized AS (\n"
        + "  SELECT patient_id, encounter_id, coding_system,\n"
        + "         REGEXP_REPLACE(UPPER(TRIM(raw_code)), r'[^A-Z0-9]', '') AS icd\n"
        + "  FROM tokens\n"
        + ")\n"
        + "SELECT DISTINCT patient_id\n"
        + "FROM normalized\n"
        + f"WHERE {where};\n"
    )


def write_sql_catalog(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for (condition, phenotype), defn in sorted(DEFINITIONS.items()):
        mimic_path = directory / f"mimic_{condition}_{phenotype}.sql"
        crdw_path = directory / f"crdw_{condition}_{phenotype}.sql"
        mimic_path.write_text(render_mimic_sql(defn), encoding="utf-8")
        crdw_path.write_text(render_crdw_sql(defn), encoding="utf-8")
        written.extend([mimic_path, crdw_path])
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write SQL phenotype files from the Python rules.")
    parser.add_argument("--write", type=Path, default=Path("sql"))
    args = parser.parse_args(argv)
    paths = write_sql_catalog(args.write)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
