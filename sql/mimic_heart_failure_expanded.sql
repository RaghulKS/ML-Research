-- AUTO-GENERATED from clinical_ml.cohorts.definitions.
-- Reproducible phenotype logic only. This statement was not executed against
-- MIMIC-IV, BigQuery, CRDW, Amarel, or any other clinical system.
-- Historical phenotypes are narrower than expanded phenotypes. Do not replace
-- previously reported historical counts with expanded counts.
-- CRDW is a hypertension-selected extract. Its counts are not general-population
-- prevalence and are not directly comparable to MIMIC-IV counts.
-- Cohort: heart_failure_expanded_v1
-- Broader heart-failure research phenotype. Includes the historical I50/428 family plus selected hypertensive and rheumatic heart-failure codes. Not a replacement for historical counts.
SELECT DISTINCT subject_id AS patient_id
FROM diagnoses_icd
WHERE (CASE WHEN icd_version = 9 THEN 'ICD9' WHEN icd_version = 10 THEN 'ICD10' ELSE '' END = 'ICD10' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') != '' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') LIKE 'I50%')
   OR (CASE WHEN icd_version = 9 THEN 'ICD9' WHEN icd_version = 10 THEN 'ICD10' ELSE '' END = 'ICD9' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') != '' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') LIKE '428%')
   OR (CASE WHEN icd_version = 9 THEN 'ICD9' WHEN icd_version = 10 THEN 'ICD10' ELSE '' END = 'ICD10' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') != '' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') = 'I110')
   OR (CASE WHEN icd_version = 9 THEN 'ICD9' WHEN icd_version = 10 THEN 'ICD10' ELSE '' END = 'ICD10' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') != '' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') = 'I130')
   OR (CASE WHEN icd_version = 9 THEN 'ICD9' WHEN icd_version = 10 THEN 'ICD10' ELSE '' END = 'ICD10' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') != '' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') = 'I132')
   OR (CASE WHEN icd_version = 9 THEN 'ICD9' WHEN icd_version = 10 THEN 'ICD10' ELSE '' END = 'ICD10' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') != '' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') = 'I0981')
   OR (CASE WHEN icd_version = 9 THEN 'ICD9' WHEN icd_version = 10 THEN 'ICD10' ELSE '' END = 'ICD9' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') != '' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') = '40201')
   OR (CASE WHEN icd_version = 9 THEN 'ICD9' WHEN icd_version = 10 THEN 'ICD10' ELSE '' END = 'ICD9' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') != '' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') = '40211')
   OR (CASE WHEN icd_version = 9 THEN 'ICD9' WHEN icd_version = 10 THEN 'ICD10' ELSE '' END = 'ICD9' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') != '' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') = '40291')
   OR (CASE WHEN icd_version = 9 THEN 'ICD9' WHEN icd_version = 10 THEN 'ICD10' ELSE '' END = 'ICD9' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') != '' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') = '40401')
   OR (CASE WHEN icd_version = 9 THEN 'ICD9' WHEN icd_version = 10 THEN 'ICD10' ELSE '' END = 'ICD9' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') != '' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') = '40403')
   OR (CASE WHEN icd_version = 9 THEN 'ICD9' WHEN icd_version = 10 THEN 'ICD10' ELSE '' END = 'ICD9' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') != '' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') = '40411')
   OR (CASE WHEN icd_version = 9 THEN 'ICD9' WHEN icd_version = 10 THEN 'ICD10' ELSE '' END = 'ICD9' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') != '' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') = '40413')
   OR (CASE WHEN icd_version = 9 THEN 'ICD9' WHEN icd_version = 10 THEN 'ICD10' ELSE '' END = 'ICD9' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') != '' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') = '40491')
   OR (CASE WHEN icd_version = 9 THEN 'ICD9' WHEN icd_version = 10 THEN 'ICD10' ELSE '' END = 'ICD9' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') != '' AND REGEXP_REPLACE(UPPER(icd_code), r'[^A-Z0-9]', '') = '40493');
