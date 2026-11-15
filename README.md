# Longitudinal clinical machine learning

This repository is a research codebase for modeling heart-failure care as a
patient timeline. The scientific goal is to predict a later outcome from
information available at a defined anchor, then test whether a model trained
in one hospital-style extract can be fine-tuned in another.

Three milestones are implemented:

1. Cohort construction and longitudinal preprocessing for heart failure, with
   stroke and myocardial infarction retained as secondary phenotypes.
2. Logistic regression, random forest, and XGBoost baselines.
3. A GRU, a compact Transformer, treatment-aware inputs, and a MIMIC-style to
   CRDW-style transfer workflow.

No MIMIC-IV or Rutgers CRDW extract is stored in this repository. The checked-in
metrics are **synthetic engineering validation**. They are not estimates of
clinical performance.

## What the prediction task is

Historical heart failure is ICD-10 prefix `I50` and ICD-9 prefix `428`.
That definition is kept separate from a broader expanded phenotype.

The default task is 30-day all-cause readmission after the first heart-failure
encounter:

- Anchor: encounter end (discharge time when an admission table provides it).
- Features and sequences: events at or before the anchor only.
- Positive label: a new encounter starts after the anchor and within 30 days.
- Negative label: no such encounter, and observation continues through day 30.
- Censored: observation ends before day 30 without a qualifying encounter.
  Those patients are excluded from training and metrics.

The label is a later encounter. It is not a repeat of the heart-failure code
that defined the cohort. Medication orders stay orders unless the source is an
administration table or an explicit administration flag.

## Layout

```text
src/clinical_ml/     cohort, timeline, model, and transfer code
configs/             phenotypes, harmonization, and experiment settings
sql/                 SQL rendered from the same rules as the Python matcher
scripts/             command-line entry points
tests/               unit tests and a small end-to-end smoke test
docs/research/       methods, milestone notes, and measured results
artifacts/           local outputs (synthetic extracts and model binaries are ignored)
```

## Setup

Python 3.10 or newer. On this machine the verified interpreter is `python3`.

CPU-only PyTorch is enough for the synthetic demonstration. These are the
install commands used in this workspace:

```bash
python3 -m pip install numpy pandas pyarrow "scikit-learn>=1.3" xgboost PyYAML joblib pytest
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`requirements.txt` lists the same packages. On a machine without a GPU, install
PyTorch from the CPU index above so pip does not fetch a CUDA wheel.

If `xgboost` cannot be imported, the boosting baseline falls back to
`sklearn.GradientBoostingClassifier` and the result is labeled
`gradient_boosting_fallback`. SHAP is optional. Permutation importance runs
without it.

## Synthetic demonstration

This is the command to run after a fresh clone. It generates two deterministic
hospitals, builds cohorts, trains every required model, checks leakage and
reload, and writes metrics.

```bash
python3 scripts/run_smoke_tests.py --output artifacts/experiments/synthetic_smoke
```

The summary is printed and stored in
`artifacts/experiments/synthetic_smoke/results.json`.
Every record is labeled `SYNTHETIC ENGINEERING VALIDATION`.

## Commands on a local extract

Place MIMIC-like tables or CRDW-like tables in a directory you pass on the
command line. CSV and Parquet are both accepted. Diagnosis codes should be
stored as strings. Files that are absent are skipped, so a diagnoses-only
extract can still build a cohort.

MIMIC-like file stems: `patients`, `admissions`, `diagnoses_icd`,
`prescriptions`, `emar`, `labevents`, `chartevents`, `procedures_icd`.

CRDW-like file stems: `diagnoses` (or `comorbidities`), `medications`, `labs`,
`vitals`, `procedures`, `patients`.

Diagnosis lists may be comma-separated. CRDW day counts are kept as relative
days. MIMIC timestamps are kept as days since the Unix epoch. Those clocks are
not subtracted from each other.

```bash
python3 scripts/validate_datasets.py \
  --source-dir data/raw/mimic --source mimic \
  --output artifacts/reports/mimic_quality.json

python3 scripts/build_cohorts.py \
  --source-dir data/raw/mimic --source mimic \
  --cohort heart_failure --phenotype historical \
  --output artifacts/cohorts/hf_historical_mimic.csv

python3 scripts/train_baselines.py \
  --source-dir data/raw/mimic --source mimic \
  --config configs/experiments/baseline_hf_readmission.yaml \
  --output artifacts/experiments/baselines_mimic

python3 scripts/train_longitudinal.py \
  --source-dir data/raw/mimic --source mimic \
  --config configs/experiments/longitudinal_hf_readmission.yaml \
  --output artifacts/experiments/longitudinal_mimic

python3 scripts/run_transfer.py \
  --source-dir data/raw/mimic --source-name mimic \
  --target-dir data/raw/crdw --target-name crdw \
  --config configs/experiments/transfer_mimic_to_crdw.yaml \
  --output artifacts/experiments/transfer_mimic_crdw
```

Pass `--dataset-type "REAL CLINICAL RESULT"` only for an authorized real
extract that does not carry synthetic provenance. A directory written by the
synthetic generator cannot be relabeled as real. If the flag is omitted, a
real extract is reported as `IMPLEMENTED BUT NOT CLINICALLY EVALUATED` until
you set the category yourself.

There is no Amarel path, SSH target, or remote warehouse client in these
commands. The SQL files in `sql/` document the same phenotype rules and were
not executed.

Regenerate the SQL after a phenotype change:

```bash
python3 -m clinical_ml.cohorts.sql --write sql
```

## Tests

```bash
PYTHONPATH=src python3 -m pytest
```

## Reading the results

Start with `docs/research/research_progress_report.md` and
`docs/research/experiment_results.md`.

Feature importance is a description of the fitted predictor. It is not a
causal estimate, and a medication-order coefficient is not evidence that the
drug was given or that it changed the outcome.

## Limitations

- Real MIMIC-IV and CRDW files were not available in this workspace, so no
  clinical performance number is claimed.
- CRDW in the research plan is a hypertension-selected extract. Its counts are
  not general-population prevalence and are not directly comparable to MIMIC-IV.
- The historical myocardial-infarction rule is narrow: exact ICD-10 `I21.9`
  and ICD-9 prefix `410.9`. `I21.90` and `I21.4` are outside that historical
  rule. The expanded phenotype is separate.
- MIMIC diagnosis rows have no event time in the source table. This pipeline
  timestamps them at admission. That is a timing limitation.
- CRDW encounters use the first and last observed event as the stay bounds
  unless a discharge field is supplied.
- Follow-up end must be on the same clock as the events. A mismatched clock is
  discarded rather than compared.
- Cross-hospital metrics require the same outcome and comparable
  ascertainment. The synthetic domains satisfy that by construction. Real
  MIMIC-IV and CRDW do not until that is shown.
