# Milestone 1 — Cohort development and preprocessing

## Research goal

Build a reusable, leakage-resistant path from MIMIC-like and CRDW-like tables
to patient timelines and a heart-failure prediction table. Keep the historical
code lists reproducible, and keep stroke and myocardial infarction available
as secondary cohorts.

## Work implemented

- Source adapters for MIMIC-like and CRDW-like directories. Each file is
  optional. CSV reading is chunked and retries Latin-1 and Windows encodings
  if UTF-8 fails. Diagnosis fields are read as strings.
- Historical and expanded phenotypes, with SQL generated from the same rule
  objects.
- A unified event table: patient, encounter, time, time origin, event type,
  code, raw code, value, unit, source, description, treatment flag,
  administration flag, provenance, missingness, and coding system.
- Cleaning for missing ids, non-finite times, and exact duplicates, plus
  chronological sorting inside each patient.
- Encounter tables. MIMIC discharge times are preserved. CRDW stays use the
  observed event span.
- Quality reports: counts, duplicate rate, missingness, time origins,
  sequence-length distribution, coverage by event type, and same-encounter
  ICD-9/ICD-10 overlap.
- Unit and code harmonization with an explicit map and physiologic quality
  bounds applied after conversion.
- The 30-day readmission label, including censoring when follow-up is shorter
  than the horizon.

## Important files

- `src/clinical_ml/data/adapters.py`
- `src/clinical_ml/data/schemas.py`
- `src/clinical_ml/cohorts/definitions.py`
- `src/clinical_ml/cohorts/sql.py`
- `src/clinical_ml/preprocessing/labels.py`
- `src/clinical_ml/preprocessing/harmonize.py`
- `src/clinical_ml/data/quality.py`
- `configs/harmonization/default.yaml`
- `sql/`

## Experiments actually executed

No authorized MIMIC-IV or CRDW file was in the workspace. The adapters were
executed on deterministic synthetic hospitals written in the MIMIC-like and
CRDW-like layouts. Details and counts are in `experiment_results.md`.

On that synthetic MIMIC-like extract the historical heart-failure cohort had
145 patients and the expanded definition had 156. The gap is the intended
`I11.0` / `402.91` cases. Stroke and MI historical counts were smaller than
their expanded counts for the same reason. CRDW-like medication rows with
`ORDER_STATUS=ordered` produced zero confirmed administrations. MIMIC-like
EMAR rows produced administrations. One creatinine recorded as 500 mg/dL was
set to missing, and one value with unit `furlongs` was set to missing.

The ICD-9/ICD-10 overlap report runs. In the synthetic files the overlap is
about half of condition-positive encounters because the generator dual-codes
every other patient. That fraction is a test of the report, not a measurement
of Rutgers coding practice.

## Validation

Unit tests cover historical versus expanded matching, comma-separated codes,
declared coding system, patient deduplication, encounter counts, Latin-1
reading, sort order, order versus administration, unit conversion, censoring,
and the refusal to call synthetic provenance a real clinical result.

## Limitations

Real extracts may use different column names, local drug codes, and a
follow-up definition this pipeline has not seen. The column aliases cover the
fields named in the project notes, and unknown columns will need an alias
rather than a silent guess. MIMIC diagnosis timing remains admission-level.

## Next step

Run `scripts/validate_datasets.py` and `scripts/build_cohorts.py` on the
authorized MIMIC-IV v3.1 tables and the PR0077 comorbidities file, and compare
historical patient counts with the earlier BigQuery and CRDW tallies before
any expanded phenotype is used in a paper.
