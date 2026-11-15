# Experiment results

All numbers below were produced in this workspace by
`python3 scripts/run_smoke_tests.py --output artifacts/experiments/synthetic_smoke`.
The separate training commands were then run on the same generated files and
reproduced the same point estimates.

Reporting category for every row: **SYNTHETIC ENGINEERING VALIDATION**.

No MIMIC-IV or CRDW patient file was available. Nothing in this note is a
clinical performance estimate. Full JSON, including bootstrap resamples and
calibration bins, is in `artifacts/experiments/synthetic_smoke/results.json`.

Environment: Python 3.12.3, numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1,
xgboost 3.4.1, torch 2.15.0+cpu. Seed 7. XGBoost was used. The gradient-boosting
fallback was not used.

## Synthetic extracts

| Extract | Patients in file | Event rows after cleaning | Historical HF patients | Expanded HF patients | Labeled for 30-day readmission | Positive | Negative | Censored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MIMIC-like | 180 | 3,895 | 145 | 156 | 124 | 54 | 70 | 21 |
| CRDW-like | 150 | 3,276 | 121 | 130 | 102 | 52 | 50 | 19 |

Cleaning of the MIMIC-like file dropped 1 row with a missing patient id, 1
row with a non-finite time, and 1 exact duplicate. Those rows were injected
by the generator to test the cleaner.

Historical versus expanded patient counts on the MIMIC-like file:

| Phenotype | Heart failure | Stroke | Myocardial infarction |
| --- | ---: | ---: | ---: |
| Historical | 145 | 16 | 13 |
| Expanded | 156 | 26 | 21 |

The CRDW-like file had the same pattern at a smaller size (heart failure 121
versus 130, stroke 13 versus 21, MI 11 versus 18).

Same-encounter ICD-9 and ICD-10 overlap for historical heart failure was 72 of
145 MIMIC-like encounters (0.497) and 60 of 121 CRDW-like encounters (0.496).
The generator dual-codes every other patient, so this fraction tests the
overlap procedure. It is not a CRDW coding study.

CRDW-like medication-administration coverage was 0. MIMIC-like files contained
both orders and EMAR administrations. Harmonization converted 506 CRDW-like
laboratory or vital values into canonical units and rejected 1 impossible
creatinine and 1 unrecognized unit in each domain.

MIMIC-like sequence lengths: median 22 events, maximum 33.

## Task and split

Cohort `heart_failure_historical_v1`. Task
`hf_30day_all_cause_readmission`. Horizon 30 days. MIMIC-like split: 74 train,
25 validation, 25 test. The test set contained 11 positive labels. Target
transfer metrics use a separate CRDW-like split with 21 test patients, 11
positive.

## Baselines on the MIMIC-like test set

| Model | AUROC | AUROC 95% bootstrap interval | AUPRC | AUPRC 95% interval | Brier | F1 at 0.5 | Sensitivity | Specificity |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Logistic regression | 0.578 | 0.346–0.792 | 0.576 | 0.294–0.837 | 0.264 | 0.522 | 0.545 | 0.571 |
| Random forest | 0.519 | 0.263–0.769 | 0.586 | 0.307–0.814 | 0.258 | 0.476 | 0.455 | 0.643 |
| XGBoost | 0.513 | 0.245–0.734 | 0.545 | 0.300–0.765 | 0.331 | 0.480 | 0.545 | 0.429 |

Confusion matrices at threshold 0.5, written as `[[TN, FP], [FN, TP]]`:

- Logistic regression: `[[8, 6], [5, 6]]`
- Random forest: `[[9, 5], [6, 5]]`
- XGBoost: `[[6, 8], [5, 6]]`

Isotonic regression fit on validation predictions changed test Brier score from
0.264 to 0.242 for logistic regression, from 0.258 to 0.264 for random forest,
and from 0.331 to 0.251 for XGBoost. The tree-model F1 threshold selected on
validation predicted every test patient positive. That operating point is
recorded in the JSON and is not used to rank models.

No training feature had absolute correlation above 0.98 with the label.
Permutation importance was computed on the validation split. The leading
logistic features were last heart rate, last systolic pressure, COPD, and age.
Importances were small relative to their standard deviations.

Train times were 0.96 s, 7.93 s, and 1.04 s for logistic regression, random
forest, and XGBoost. Reloaded probabilities matched to a maximum absolute
difference of 0.

## Sequence models on the same MIMIC-like split

| Model | Treatment events | AUROC | AUPRC | Brier | Parameters |
| --- | --- | ---: | ---: | ---: | ---: |
| GRU | included | 0.474 | 0.500 | 0.249 | 16,757 |
| GRU | removed | 0.383 | 0.386 | 0.255 | 16,517 |
| Transformer | included | 0.539 | 0.577 | 0.249 | 42,101 |

Test-set unknown-token rate under the training vocabulary was 0 for these
within-domain runs. Reload maximum absolute difference was 0. MC dropout on
the treatment-aware GRU had mean predictive standard deviation 0.044.

## Transfer, synthetic MIMIC-like source to synthetic CRDW-like target

GRU, treatment events included:

| Stage | Patients scored | AUROC | AUPRC | Brier |
| --- | ---: | ---: | ---: | ---: |
| Source held-out | 25 | 0.474 | 0.500 | 0.249 |
| Frozen source on target | 21 | 0.391 | 0.467 | 0.254 |
| Full fine-tune | 21 | 0.536 | 0.551 | 0.252 |
| Head-only fine-tune | 21 | 0.400 | 0.471 | 0.254 |
| Target from scratch | 21 | 0.855 | 0.821 | 0.227 |

Transformer, same protocol, two epochs for the transfer fits:

| Stage | AUROC | AUPRC | Brier |
| --- | ---: | ---: | ---: |
| Source held-out | 0.539 | 0.577 | 0.249 |
| Frozen source on target | 0.609 | 0.705 | 0.254 |
| Full fine-tune | 0.600 | 0.721 | 0.245 |
| Head-only fine-tune | 0.600 | 0.699 | 0.250 |
| Target from scratch | 0.636 | 0.653 | 0.246 |

Target test tokens unknown to the source GRU vocabulary, after harmonization:
0.0065.

The from-scratch GRU point estimate is much higher than the other neural
estimates on a 21-patient test set. That spread is a reason to distrust a
single AUROC here, not a reason to prefer that run. No confidence interval was
stored for the transfer metrics. The baseline intervals above, on a similarly
small test set, commonly span several tenths.

## Reproduction

```bash
python3 scripts/run_smoke_tests.py --output artifacts/experiments/synthetic_smoke
python3 scripts/train_baselines.py \
  --source-dir artifacts/experiments/synthetic_smoke/data/mimic --source mimic \
  --config configs/experiments/baseline_hf_readmission.yaml \
  --output artifacts/experiments/baselines_mimic_synthetic
python3 scripts/train_longitudinal.py \
  --source-dir artifacts/experiments/synthetic_smoke/data/mimic --source mimic \
  --config configs/experiments/longitudinal_hf_readmission.yaml \
  --output artifacts/experiments/longitudinal_mimic_synthetic
python3 scripts/run_transfer.py \
  --source-dir artifacts/experiments/synthetic_smoke/data/mimic --source-name mimic \
  --target-dir artifacts/experiments/synthetic_smoke/data/crdw --target-name crdw \
  --config configs/experiments/transfer_mimic_to_crdw.yaml \
  --output artifacts/experiments/transfer_synthetic
```

The second, third, and fourth commands were executed after the smoke run and
printed the same AUROC, AUPRC, and Brier values as the smoke summary.

## Tests

`PYTHONPATH=src python3 -m pytest` passed 20 tests, including a smaller
end-to-end smoke with one epoch. That small test checks wiring. The table
above is from the full smoke configuration, not from the one-epoch test.
