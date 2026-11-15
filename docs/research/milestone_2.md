# Milestone 2 — Baseline machine learning

## Research goal

Train honest classical baselines for the 30-day readmission task before
treating a neural network as an improvement.

## Work implemented

- Pre-anchor tabular features listed in `methodology.md`.
- Patient-disjoint stratified splits.
- Logistic regression, random forest, and XGBoost, with a small validation
  grid. If XGBoost is absent, sklearn gradient boosting is used and named as
  a fallback. In this environment XGBoost 3.4.1 imported, so the fallback was
  not used.
- Metrics: AUROC, AUPRC, precision, recall, F1, sensitivity, specificity,
  confusion matrix, Brier score, calibration bins, bootstrap intervals, and
  an isotonic calibrator fit on validation predictions.
- Permutation importance on the validation split and a correlation screen for
  near-copies of the label.
- Model save and reload, with a check that reloaded probabilities match.

## Important files

- `src/clinical_ml/features/tabular.py`
- `src/clinical_ml/models/baselines.py`
- `src/clinical_ml/evaluation/metrics.py`
- `src/clinical_ml/evaluation/interpret.py`
- `configs/experiments/baseline_hf_readmission.yaml`
- `scripts/train_baselines.py`

## Experiments actually executed

The baseline script was run on the synthetic MIMIC-like hospital. The held-out
test set has 25 patients, 11 of whom were readmitted inside 30 days. Training
used 74 patients and validation used 25.

Point estimates were close to chance, and the bootstrap intervals are wide.
That is the expected behavior of a small noisy synthetic task, and it is
reported as such in `experiment_results.md`. No threshold or feature was
chosen using the test set.

The validation-set F1 threshold for the tree models dropped low enough that
specificity on the test set became zero. Thresholded F1 is therefore not used
to rank models. AUROC and AUPRC are the comparison metrics. Threshold 0.5 is
reported only as a fixed operating point.

The correlation screen did not flag any training feature. Permutation
importance on the validation split pointed at heart rate, hemoglobin, age,
creatinine, and diuretic-order indicators, with standard deviations large
enough that the ranking should not be over-read. These are associations inside
a generator that tied the latent risk score to age, prior encounters,
creatinine, and diuretic orders.

## Validation

Tests check metric values on hand-built scores, the leakage audit that keeps
a post-anchor creatinine out of `creat_last`, disjoint splits, and reload
agreement inside the smoke run (maximum absolute difference 0).

## Limitations

A 25-patient test set cannot support a claim that one classical model is
better than another. Isotonic calibration changed Brier score in both
directions across models, which is consistent with a small validation set.
Class weights address imbalance. They do not create a calibrated probability
by themselves.

## Next step

Refit the same configuration on an authorized heart-failure extract after the
label is checked against an actual discharge and readmission definition in
that extract.
