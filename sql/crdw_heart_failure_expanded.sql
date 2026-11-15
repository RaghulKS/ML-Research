-- AUTO-GENERATED from clinical_ml.cohorts.definitions.
-- Reproducible phenotype logic only. This statement was not executed against
-- MIMIC-IV, BigQuery, CRDW, Amarel, or any other clinical system.
-- Historical phenotypes are narrower than expanded phenotypes. Do not replace
-- previously reported historical counts with expanded counts.
-- CRDW is a hypertension-selected extract. Its counts are not general-population
-- prevalence and are not directly comparable to MIMIC-IV counts.
-- Cohort: heart_failure_expanded_v1
-- Broader heart-failure research phenotype. Includes the historical I50/428 family plus selected hypertensive and rheumatic heart-failure codes. Not a replacement for historical counts.
-- CRDW lists may be comma-separated. A LIKE against the raw list
-- column can match a fragment of a neighboring code. Split first.
-- Dialect sketch: BigQuery. Replace `comorbidities` with the extract table.
WITH tokens AS (
  SELECT PAT_MRN_ID_ENCRYPT AS patient_id,
         PAT_ENC_CSN_ID_ENCRYPT AS encounter_id,
         'ICD10' AS coding_system,
         code AS raw_code
  FROM comorbidities,
  UNNEST(SPLIT(IFNULL(CURRENT_ICD10_LIST, ''), ',')) AS code
  UNION ALL
  SELECT PAT_MRN_ID_ENCRYPT,
         PAT_ENC_CSN_ID_ENCRYPT,
         'ICD9',
         code
  FROM comorbidities,
  UNNEST(SPLIT(IFNULL(CURRENT_ICD9_LIST, ''), ',')) AS code
), normalized AS (
  SELECT patient_id, encounter_id, coding_system,
         REGEXP_REPLACE(UPPER(TRIM(raw_code)), r'[^A-Z0-9]', '') AS icd
  FROM tokens
)
SELECT DISTINCT patient_id
FROM normalized
WHERE (coding_system = 'ICD10' AND icd != '' AND icd LIKE 'I50%')
   OR (coding_system = 'ICD9' AND icd != '' AND icd LIKE '428%')
   OR (coding_system = 'ICD10' AND icd != '' AND icd = 'I110')
   OR (coding_system = 'ICD10' AND icd != '' AND icd = 'I130')
   OR (coding_system = 'ICD10' AND icd != '' AND icd = 'I132')
   OR (coding_system = 'ICD10' AND icd != '' AND icd = 'I0981')
   OR (coding_system = 'ICD9' AND icd != '' AND icd = '40201')
   OR (coding_system = 'ICD9' AND icd != '' AND icd = '40211')
   OR (coding_system = 'ICD9' AND icd != '' AND icd = '40291')
   OR (coding_system = 'ICD9' AND icd != '' AND icd = '40401')
   OR (coding_system = 'ICD9' AND icd != '' AND icd = '40403')
   OR (coding_system = 'ICD9' AND icd != '' AND icd = '40411')
   OR (coding_system = 'ICD9' AND icd != '' AND icd = '40413')
   OR (coding_system = 'ICD9' AND icd != '' AND icd = '40491')
   OR (coding_system = 'ICD9' AND icd != '' AND icd = '40493');
