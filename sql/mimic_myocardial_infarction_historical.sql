-- AUTO-GENERATED from clinical_ml.cohorts.definitions.
-- Reproducible phenotype logic only. This statement was not executed against
-- MIMIC-IV, BigQuery, CRDW, Amarel, or any other clinical system.
-- Historical phenotypes are narrower than expanded phenotypes. Do not replace
-- previously reported historical counts with expanded counts.
-- CRDW is a hypertension-selected extract. Its counts are not general-population
-- prevalence and are not directly comparable to MIMIC-IV counts.
-- Cohort: mi_historical_v1
-- Historical myocardial-infarction list: exact ICD-10 I21.9 and ICD-9 prefix 410.9. This is intentionally narrow.
SELECT DISTINCT subject_id AS patient_id
FROM diagnoses_icd
WHERE (CASE WHEN icd_version = 9 THEN 'ICD9' WHEN icd_version = 10 THEN 'ICD10' ELSE '' END = 'ICD10' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') != '' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') = 'I219')
   OR (CASE WHEN icd_version = 9 THEN 'ICD9' WHEN icd_version = 10 THEN 'ICD10' ELSE '' END = 'ICD9' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') != '' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') LIKE '4109%');
