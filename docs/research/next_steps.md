# Next steps

These items are not done. They require authorized data or a design choice that
should be made with Professor Ljubic.

## Full clinical-data cohort validation

Run the historical SQL, or the Python extractor, on MIMIC-IV v3.1
`diagnoses_icd` and on `Hypertension With Comorbidities 20250404.csv`.
Compare patient counts with the earlier tallies (MIMIC-IV heart failure
31,369; CRDW heart failure 55,832) and record any gap. Do not replace those
tallies with the expanded phenotype. Investigate ICD-9/ICD-10 co-occurrence
on real CRDW rows with `coding_overlap_report` before interpreting it.

## Outcome feasibility

Confirm that a discharge time and a subsequent encounter can be defined in
both sources. CRDW relative day counters need a documented origin. Patients
without 30 days of follow-up should remain censored. Do not label them
negative. Do not define the outcome as a later `I50` or `428` code if the
scientific question is readmission in a cohort already defined by those codes.

## Real-data baselines and longitudinal comparison

Refit logistic regression, random forest, XGBoost, the GRU, and the
Transformer on the authorized heart-failure cohort with the same split rules.
Publish AUROC and AUPRC with intervals. Keep the test set untouched until the
final report.

## MIMIC-IV pretraining and CRDW fine-tuning

Use the existing transfer command only after the outcome definitions match.
Then compare frozen evaluation, partial fine-tuning, full fine-tuning, and a
CRDW model trained from scratch, including curves over the fraction of CRDW
training patients. The current code trains on the full target training split.
A fraction sweep is a small extension once real data exist. It should not be
reported on the synthetic generator as if it were a transport result.

Also check, with the local data dictionaries, the MIMIC itemids and CRDW
component codes in `configs/harmonization/default.yaml`. A shared name is not
evidence of a shared measurement.

## Treatment-aware ablations

Repeat the with-treatment and without-treatment comparison on real data.
State it as an observational association. Keep orders and administrations
separate, especially in CRDW, where an order is not a dose given.

## Uncertainty

The isotonic calibrator and MC dropout are already in the training path.
On real data, report Brier score before and after calibration and the spread
of MC dropout probabilities. Do not describe either one as proof that the
model knows when it is outside the training population.

## Later offline reinforcement learning

Only after the supervised pipeline has a credible real-data label:

- define the decision time and the state as the pre-decision timeline
- define actions from administrations and procedures, not from orders alone
- pre-specify the reward
- describe the behavior policy and where action support is missing
- plan an off-policy evaluation before any policy is fit

No recommender should be trained until that design is reviewed. The
representation in this repository is a prerequisite, not a policy.
