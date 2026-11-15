-- AUTO-GENERATED from clinical_ml.cohorts.definitions.
-- Reproducible phenotype logic only. This statement was not executed against
-- MIMIC-IV, BigQuery, CRDW, Amarel, or any other clinical system.
-- Historical phenotypes are narrower than expanded phenotypes. Do not replace
-- previously reported historical counts with expanded counts.
-- CRDW is a hypertension-selected extract. Its counts are not general-population
-- prevalence and are not directly comparable to MIMIC-IV counts.
-- Cohort: heart_failure_historical_v1
-- Historical heart-failure list: ICD-10 prefix I50 and ICD-9 prefix 428. This is the definition behind the previously reported counts.
SELECT DISTINCT subject_id AS patient_id
FROM diagnoses_icd
WHERE (CASE WHEN icd_version = 9 THEN 'ICD9' WHEN icd_version = 10 THEN 'ICD10' ELSE '' END = 'ICD10' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') != '' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') LIKE 'I50%')
   OR (CASE WHEN icd_version = 9 THEN 'ICD9' WHEN icd_version = 10 THEN 'ICD10' ELSE '' END = 'ICD9' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') != '' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') LIKE '428%');
