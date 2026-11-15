# Methodology

This note describes the methods implemented in the repository. It does not
report clinical findings. Measured synthetic checks are in
`experiment_results.md`.

## Cohort construction

Two phenotype layers are maintained.

Historical phenotypes reproduce the previously supplied code lists. They are
the definitions that should be used if a later extract is compared with the
earlier MIMIC-IV and CRDW counts.

| Condition | Historical rule |
| --- | --- |
| Heart failure | ICD-10 prefix `I50`; ICD-9 prefix `428` |
| Stroke | ICD-10 prefix `I63`; ICD-9 prefix `434` |
| Myocardial infarction | ICD-10 exact `I21.9`; ICD-9 prefix `410.9` |

Codes are uppercased and stripped of dots and other non-alphanumeric
characters before matching. `I21.9` therefore becomes `I219` and does not
match `I2190` (`I21.90`). `410.91` becomes `41091` and does match prefix
`4109`.

Expanded phenotypes add codes and are versioned separately
(`heart_failure_expanded_v1`, and the stroke and MI analogues). Heart-failure
additions are limited to codes that specify heart failure, including `I11.0`,
`I13.0`, `I13.2`, `I09.81`, and the ICD-9 `402`/`404` codes whose fifth digits
indicate heart failure. Prefixes such as `402` or `I13` are not used, because
they also cover disease without heart failure. `I13.1` and `402.90` stay out.

A declared ICD version overrides the letter/digit guess. A code declared as
ICD-9 is not allowed to satisfy an ICD-10 rule.

Comma-, semicolon-, and pipe-separated diagnosis fields are split before
matching. Empty tokens are dropped. The same rules are rendered to SQL under
`sql/`. Those statements were not executed against BigQuery or CRDW.

Patients are counted once. Encounters are `(patient_id, encounter_id)` pairs.
Rows with a missing patient id or a non-finite time are dropped. Exact
duplicate rows are dropped. A missing encounter id becomes a per-time
placeholder so unrelated events are not collapsed into one stay.

## Clocks

Every event has a `time_origin`.

- MIMIC-like timestamps become days since the Unix epoch
  (`calendar_days_since_unix_epoch`).
- CRDW-like day counters stay in `relative_days`.

Gaps are computed inside one patient and one origin. A follow-up timestamp on
a different origin is discarded. Calendar days are not subtracted from
relative day counts.

MIMIC diagnosis tables do not carry a time. Matching diagnoses are placed at
the admission time. The readmission anchor is the discharge time from the
admission table when that table exists. CRDW encounters, which have no
discharge column in the expected extract, use the earliest and latest event
times in the encounter. A vital or lab on the intended last day is what makes
that end observable.

## Prediction task

Task id: `hf_30day_all_cause_readmission`.
Implementation id: `readmission_v1`.

Default cohort: historical heart failure.
Index encounter: earliest encounter containing a matching diagnosis.
Anchor: that encounter's end.
Observation window: events with the same time origin and `event_time <= anchor`.
Horizon: 30 days.

Positive: another encounter starts strictly after the anchor and at most 30
days later. The later diagnosis can be anything.
Negative: no such encounter, and the observed end of follow-up is at least
30 days after the anchor.
Censored: follow-up ends earlier and no qualifying encounter was seen. Censored
patients are removed before the split. They are counted in the task summary
and are not given an imputed label.

Follow-up end is the latest of the patient's event times, encounter end times,
and an optional follow-up field on the same clock. The optional field is how a
censored patient can be recognized when no later event exists. It is not
filled in with a study end date for every patient.

## Feature engineering

Tabular features use only the pre-anchor window:

- age and sex, with a missingness indicator for sex
- encounter counts, prior-encounter count, and days since the previous encounter
- diagnosis counts and comorbidity flags (diabetes, hypertension, CKD, COPD,
  atrial fibrillation, prior stroke, prior MI)
- medication-order counts and medication-administration counts, kept separate
- order and administration flags for diuretic, ACE inhibitor/ARB, beta blocker,
  anticoagulant, and statin classes
- last, and where useful mean or change, for creatinine, sodium, potassium,
  hemoglobin, BNP, systolic pressure, diastolic pressure, and heart rate
- event counts in the prior 30 and 180 days
- index length of stay

Laboratory and vital values are harmonized to canonical units before
aggregation. Unknown units and values outside configured quality bounds become
missing. The bounds are quality filters, not reference ranges. Creatinine is
stored as mg/dL (`1 mg/dL = 88.4 umol/L`). Hemoglobin is stored as g/dL.

Median imputation and, for logistic regression, scaling are fit on the
training split only. Columns that are entirely missing in training are dropped
and the dropped list is saved. The validation split selects hyperparameters.
The test split is scored once.

Comorbidity flags are feature definitions. They are not the cohort definitions.
In particular, the MI history flag uses the broader `I21`/`I22`/`410` family,
while the historical MI cohort remains `I21.9` and `410.9*`.

## Models

Baselines:

- logistic regression, `class_weight="balanced"`, `C` in `{0.1, 1, 3}`
- random forest, 200 trees, depth in `{4, 6}`, `class_weight="balanced_subsample"`
- XGBoost, 80 trees, depth in `{2, 3}`, `scale_pos_weight` from the training
  class counts

Selection score is validation AUPRC.

Sequences are the most recent pre-anchor events, truncated to a configured
length (32 in the synthetic demonstration). Each event has a code embedding,
an event-type embedding, a treatment embedding, and a numeric projection of
the standardized value, a missingness bit, the log time gap, and the log time
to the anchor. Age and sex enter the classification head as static inputs.
Code vocabularies and numeric means and standard deviations are fit on
training histories only. Unseen codes become `<UNK>`.

The GRU uses the hidden state at the true sequence length, not the padded
tail. The Transformer is a 2-layer encoder with 4 heads and width 48, with a
padding mask and a masked mean pool. Both are trained with class-weighted
binary cross-entropy. The checkpoint with the best validation AUPRC is
reloaded.

Treatment-aware modeling means medication-order, medication-administration,
and procedure events keep a learned treatment embedding. The ablation removes
those events and refits the vocabulary. The cohort, label, split, and epoch
budget stay the same. A difference between the two runs is an observational
association in the fitted predictor.

## Evaluation

Primary metrics are AUROC and AUPRC. Also reported: precision, recall, F1,
sensitivity, specificity, the confusion matrix at threshold 0.5, a
validation-chosen F1 threshold, Brier score, and equal-width calibration bins.
Baseline metrics on the test set have percentile bootstrap intervals (200
resamples). Resamples with one class are skipped.

Isotonic regression is fit on validation probabilities and applied to test
probabilities. Monte Carlo dropout (8 stochastic passes) reports the mean
predictive standard deviation for the neural models. Neither procedure is
treated as out-of-distribution detection.

Permutation importance uses validation average precision. It is descriptive.
A correlation screen flags any training feature with absolute correlation at
least 0.98 with the label so that a copied outcome can be seen.

## Transfer

The workflow requires the same task name, implementation, cohort version, and
horizon. If those differ, the run stops.

1. Train on source patients.
2. Evaluate on held-out source patients.
3. Encode the target test patients with the source vocabulary and source
   normalizer, and evaluate the frozen model.
4. Add target training codes that the source vocabulary lacks, initialize the
   new embedding rows from the unknown-token vector, and fine-tune all
   parameters.
5. Repeat from the source checkpoint with only the classification head
   trainable.
6. Train a separate target model from scratch with a vocabulary fit on target
   training patients.

Harmonization is a configured code and unit map. Shared names are not assumed
to share a meaning unless the map says so. MIMIC item identifiers in the
default map are common public identifiers and still need to be checked against
a local `d_labitems` / `d_items` dictionary before a clinical run.

## Splits and leakage controls

Splits are patient-level and stratified when both classes are large enough.
A patient cannot appear in two splits. Training, vocabulary fitting,
imputation, scaling, normalizers, and hyperparameter choice use the training
or validation split as specified above. Test labels are used only for the
final metrics.

`audit_max_history_time` is stored with the feature table and checked against
the anchor. Sequence construction raises if an event time is after the anchor.

## What this design does not do

It does not impute a medication administration from an order. It does not
interpret CRDW hypertension-extract counts as population prevalence. It does
not claim that a feature importance ranking is a treatment effect. It does not
train a reinforcement-learning policy. The state and action representation
that would be required later is described in `milestone_3.md`.
