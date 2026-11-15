-- AUTO-GENERATED from clinical_ml.cohorts.definitions.
-- Reproducible phenotype logic only. This statement was not executed against
-- MIMIC-IV, BigQuery, CRDW, Amarel, or any other clinical system.
-- Historical phenotypes are narrower than expanded phenotypes. Do not replace
-- previously reported historical counts with expanded counts.
-- CRDW is a hypertension-selected extract. Its counts are not general-population
-- prevalence and are not directly comparable to MIMIC-IV counts.
-- Cohort: stroke_expanded_v1
-- Broader stroke research phenotype including hemorrhagic and unspecified stroke. Not a replacement for historical counts.
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
WHERE (coding_system = 'ICD10' AND icd != '' AND icd LIKE 'I63%')
   OR (coding_system = 'ICD9' AND icd != '' AND icd LIKE '434%')
   OR (coding_system = 'ICD10' AND icd != '' AND icd LIKE 'I60%')
   OR (coding_system = 'ICD10' AND icd != '' AND icd LIKE 'I61%')
   OR (coding_system = 'ICD10' AND icd != '' AND icd = 'I64')
   OR (coding_system = 'ICD9' AND icd != '' AND icd = '430')
   OR (coding_system = 'ICD9' AND icd != '' AND icd = '431')
   OR (coding_system = 'ICD9' AND icd != '' AND icd LIKE '433%')
   OR (coding_system = 'ICD9' AND icd != '' AND icd = '436');
