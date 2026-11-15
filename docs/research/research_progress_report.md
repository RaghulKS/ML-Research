# Research progress report

Undergraduate research with Professor Branimir Ljubic, Rutgers University.

Subject: longitudinal modeling of heart failure, with stroke and myocardial
infarction kept as secondary cohorts, and a path from MIMIC-IV-style training
to a CRDW-style target.

This report uses Week 1, Week 2, and Week 3 as milestone labels from the
project plan. The software described here was built and executed in one
implementation pass. It should not be read as a diary of three calendar weeks.
No authorized MIMIC-IV or Rutgers CRDW extract was mounted in the workspace,
so every trained model below is a synthetic engineering check.

## Week 1 / Milestone 1 — Cohort development and data preprocessing

### Research goal

Represent patients as timelines rather than as isolated diagnosis rows, and
make the earlier heart-failure, stroke, and myocardial-infarction code lists
reproducible.

### Work implemented

I separated the historical code lists from a broader research phenotype.
Historical heart failure remains ICD-10 `I50*` and ICD-9 `428*`. Historical
stroke remains `I63*` and `434*`. Historical myocardial infarction remains
exact ICD-10 `I21.9` and ICD-9 `410.9*`. The expanded lists are versioned
apart from those rules so a later count cannot silently change meaning.
`I21.90` does not match historical `I21.9`. `402.90` and `I13.1`, which do
not specify heart failure, do not enter the expanded heart-failure list.

MIMIC-like and CRDW-like readers accept the files that happen to be present.
Comma-separated CRDW diagnosis fields are split before matching. Medication
orders are not treated as administrations. CRDW day offsets and MIMIC
timestamps keep different time origins and are not subtracted. The prediction
table defines an anchor at the end of the first heart-failure encounter and
builds a 30-day all-cause readmission label. Patients whose observation ends
before 30 days, and who have no readmission, are censored rather than labeled
negative.

### Technical methods

Python is the executable path. SQL for the same rules is generated into `sql/`
and was not run against BigQuery or an internal database. Quality reports
cover counts, missingness, duplicate rows, sequence length, event-type
coverage, and same-encounter ICD-9/ICD-10 overlap.

### Important files

`src/clinical_ml/cohorts/definitions.py`, `src/clinical_ml/data/adapters.py`,
`src/clinical_ml/preprocessing/labels.py`, `configs/harmonization/default.yaml`.

### Experiments actually executed

Synthetic MIMIC-like extract: 180 patients, 3,895 events after cleaning, 145
historical heart-failure patients, 156 under the expanded definition.
Synthetic CRDW-like extract: 150 patients, 121 historical heart-failure
patients, 130 expanded. Of the 145 historical MIMIC-like patients, 124 had a
usable 30-day label (54 positive, 70 negative) and 21 were censored.

The overlap report returned a dual-coding fraction near one half. The
generator plants dual coding, so this confirms the report runs. It says
nothing about the PR0077 extract.

### Validation

Tests cover code matching, comma-separated fields, deduplication, clocks,
order versus administration, and censoring. The synthetic cleaner dropped the
injected blank id, the bad timestamp, and the duplicate row.

### Limitations

Real column layouts and real follow-up definitions may still need aliases.
MIMIC diagnoses are timestamped at admission because the diagnosis table has
no event time. CRDW encounter ends are the last observed event unless a
discharge field is added.

### Next steps

Compare historical counts on the real MIMIC-IV v3.1 diagnosis table and the
PR0077 comorbidities file with the earlier tallies before modeling.

## Week 2 / Milestone 2 — Baseline machine learning

### Research goal

Establish logistic regression, random forest, and XGBoost baselines for
30-day readmission, with patient-level splits and no post-anchor features.

### Work implemented

Features come from demographics, prior utilization, comorbidity flags,
medication orders, medication administrations, labs, and vitals at or before
discharge. Imputation and scaling are fit on training patients. The validation
split picks the small hyperparameter grid. Test labels are used once.

### Technical methods

AUROC and AUPRC are the ranking metrics. Bootstrap intervals use 200 test-set
resamples. Permutation importance is computed on validation data. A screen
flags features that are nearly copies of the label.

### Important files

`src/clinical_ml/features/tabular.py`, `src/clinical_ml/models/baselines.py`,
`scripts/train_baselines.py`.

### Experiments actually executed

On the synthetic MIMIC-like test set of 25 patients (11 positive), test AUROC
was 0.578 for logistic regression, 0.519 for random forest, and 0.513 for
XGBoost. The 95% bootstrap intervals all cover a wide range around chance
(logistic AUROC interval 0.346–0.792). AUPRC was 0.576, 0.586, and 0.545.
Brier scores were 0.264, 0.258, and 0.331. XGBoost 3.4.1 ran; the sklearn
fallback was not needed.

The validation-chosen F1 threshold was unstable for the tree models and is
not used to rank them. Isotonic calibration on the validation predictions
improved logistic and XGBoost Brier scores slightly and worsened the random
forest Brier score. No feature was flagged as a near-copy of the label.

### Results actually observed

The baselines do not separate this synthetic test set in a stable way. That
is a property of the small sample and the noisy label generator. It is not a
statement about heart-failure readmission in MIMIC-IV or in RWJBarnabas data.

### Validation

Held-out metrics, reload checks, and a unit test that a creatinine after the
anchor does not become `creat_last`.

### Limitations

Twenty-five test patients cannot rank these models. The synthetic label was
generated from a latent score that also influences age, prior encounters,
creatinine, and diuretic orders, so an importance ranking that mentions those
variables is partly by construction.

### Next steps

Repeat this exact configuration on an authorized cohort once the readmission
label has been checked against that data's discharge and follow-up fields.

## Week 3 / Milestone 3 — Longitudinal deep learning and transfer

### Research goal

Train a GRU and a compact Transformer on the same label and split, retain
treatment events as their own input type, and implement source training,
frozen evaluation, fine-tuning, and a from-scratch target model.

### Work implemented

Patient sequences use code, type, and treatment embeddings plus standardized
numeric values, missingness, and time gaps. Vocabularies are fit on training
patients. The transfer path harmonizes configured codes and units, refuses
mismatched outcome definitions, and supports full and head-only fine-tuning.
MC dropout and validation-fit isotonic regression are included as uncertainty
tools, not as out-of-distribution detectors.

I did not build a reinforcement-learning treatment policy. `milestone_3.md`
states what a later offline RL study would need: a decision-time state, an
action that is not inferred from an order alone, a pre-specified reward, and
an overlap assessment. Confounding and incomplete medication administration
data make a policy claim inappropriate with the current extracts.

### Technical methods

Demonstration width is 48, with one GRU layer or two Transformer layers and
four attention heads. Training selects the checkpoint with the best validation
AUPRC. The synthetic run used four epochs for the main fits and two epochs for
the Transformer transfer fits.

### Important files

`src/clinical_ml/models/neural.py`, `src/clinical_ml/training/loop.py`,
`scripts/train_longitudinal.py`, `scripts/run_transfer.py`.

### Experiments actually executed

On the same 25 synthetic MIMIC-like test patients, AUROC was 0.474 for the
GRU with treatment events, 0.383 for the GRU without them, and 0.539 for the
Transformer. Parameter counts were 16,757, 16,517, and 42,101.

GRU transfer to the synthetic CRDW-like domain, 21 test patients: frozen
AUROC 0.391, full fine-tune 0.536, head-only 0.400, from-scratch 0.855.
The Transformer transfer on the same protocol was more tightly grouped
(frozen 0.609, from-scratch 0.636). I am not interpreting the 0.855 value as
evidence of successful transport or of a strong clinical model. It is one
point estimate on 21 patients, and it disagrees with the other neural
estimates from the same protocol.

After harmonization, 0.65% of target test tokens were unknown to the source
vocabulary. CRDW-like creatinine values were converted from umol/L to mg/dL.

### Results actually observed

Sequence models did not beat the logistic regression point estimate on the
synthetic MIMIC-like test set. Transfer runs demonstrate that the training
stages execute and that frozen, fine-tuned, and scratch models can be scored
on a shared outcome. They do not estimate MIMIC-IV to CRDW generalization.

### Validation

Masking, unknown tokens, head-only freezing, checkpoint reload, and the
end-to-end smoke command. Pytest: 20 passed.

### Limitations

Sequence length is capped. Real MIMIC-IV and CRDW differ in population
(CRDW here is hypertension-selected), coding, units, medication
administration, and possibly outcome ascertainment. The code refuses to score
a transfer when the task definitions differ. It cannot detect a subtle
difference in how readmission is recorded if both sides use the same
configuration name.

### Next steps

The practical next experiment is a historical-phenotype count on the real
tables, then a feasibility check that 30-day readmission can be labeled with
a known follow-up end. Modeling on real data should wait for that check.
Cross-hospital fine-tuning should wait until the outcome definition is shown
to match.

## What I can say was accomplished

I can say that the cohort logic, the leakage controls, the three classical
models, the GRU, the Transformer, the treatment-event ablation, and the
transfer stages are implemented, tested, and executed on synthetic data whose
results are stored in the repository. I cannot say that a model predicts
heart-failure readmission in MIMIC-IV or at RWJBarnabas, because those data
were not available for this run.
