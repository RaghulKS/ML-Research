# Milestone 3 — Longitudinal models and transfer

## Research goal

Add sequence models that respect variable-length histories, keep treatment
events distinct from observations, and implement the MIMIC-to-CRDW training
workflow without pretending the domains are the same population.

## Work implemented

- A PyTorch dataset of variable-length event sequences with padding masks.
- A GRU and a compact Transformer (width 48, 2 layers, 4 heads in the
  demonstration configuration).
- Treatment embeddings, plus an ablation that removes treatment events.
- Source training, frozen target evaluation, full fine-tuning, head-only
  fine-tuning, and a target model trained from scratch.
- Vocabulary extension for target-only codes, initialized from the unknown
  token. Head-only fine-tuning does not update those new rows.
- Isotonic calibration and MC dropout as uncertainty diagnostics.
- A hard stop when the source and target tasks disagree.

## Important files

- `src/clinical_ml/models/sequence.py`
- `src/clinical_ml/models/neural.py`
- `src/clinical_ml/training/loop.py`
- `src/clinical_ml/experiments.py`
- `configs/experiments/longitudinal_hf_readmission.yaml`
- `configs/experiments/transfer_mimic_to_crdw.yaml`
- `scripts/train_longitudinal.py`
- `scripts/run_transfer.py`

## Experiments actually executed

On the synthetic MIMIC-like test set (25 patients), the Transformer AUROC was
0.539 and the treatment-aware GRU AUROC was 0.474. Removing treatment events
lowered the GRU point estimate to 0.383. With 25 test patients those
differences sit inside ordinary sampling noise. They are not evidence that
recording a medication helps or hurts patients.

The GRU transfer onto the synthetic CRDW-like domain used a 21-patient target
test set (11 positive). Frozen source AUROC was 0.391. Full fine-tuning was
0.536. Head-only fine-tuning was 0.400. A model trained from scratch on the
target training split had AUROC 0.855. The same protocol with the Transformer
produced a from-scratch AUROC of 0.636 and a frozen AUROC of 0.609. The 0.855
figure is one small split, and the other neural estimates on that split do
not cluster near it. It is recorded because it was the measured value. It is
not a clinical result and it is not a stable ranking.

About 0.65% of target test tokens were unknown under the source vocabulary
after harmonization. Harmonization mapped furosemide and lasix to the same
loop-diuretic code and converted CRDW creatinine from umol/L to mg/dL (506
unit conversions on the synthetic CRDW file). Domain-specific drugs
(digoxin on the source side, hydralazine on the target side) are left
unmapped on purpose.

MC dropout mean predictive standard deviation for the treatment-aware GRU was
0.044. The Brier score of the averaged MC probabilities was 0.252, close to
the deterministic Brier score of 0.249. That is a stability check, not a
detection of out-of-distribution patients.

## Architecture and cost

The demonstration models are intentionally small.

| Model | Parameters | Recorded train time |
| --- | ---: | ---: |
| GRU, treatment events kept | 16,757 | 0.89 s |
| GRU, treatment events removed | 16,517 | 0.10 s |
| Transformer, treatment events kept | 42,101 | 0.15 s |

Times are wall clock on the CPU used for this run and include only the
training loop for that fit. They will change with hardware. The parameter
difference between the two GRUs is the vocabulary: treatment codes are not
added when those events are removed.

## Validation

Tests check unknown-token handling, padding masks, finite GRU and Transformer
logits, frozen encoder weights under head-only training, and checkpoint
reload. The smoke command also checks that the treatment ablation keeps the
same patients and that reloaded neural probabilities match.

## Limitations

Sequences are capped at 32 events and use the most recent pre-anchor events.
That is an information limit. Attention among those events is allowed because
the prediction is made once, at the anchor, with the history in view. The
model is not an online streaming model.

Static age uses a normalizer fit on the training domain. Applying the source
normalizer to the target is deliberate in the frozen and fine-tuned runs.
The scratch model fits its own.

## Offline reinforcement learning, later

The current tables can support a retrospective offline RL study only after
the causal design is specified. They do not support one now.

A future state could be the pre-decision event sequence already built here:
diagnoses, labs, vitals, and missingness up to a decision time. A future
action could be a medication order or a procedure, stored separately from
observations, and only an administration would count as a completed action.
Orders must not be recoded as actions taken. A future reward could be a
clinical outcome after the decision, such as readmission, death, or a lab
threshold, defined before looking at model fit. The transition is the next
observed state. It is an incomplete observation of the patient, not a
controlled step.

The obstacles are the usual ones for critical-care offline RL. Treatment is
confounded by indication. Many plausible actions never appear, so the
behavior policy has poor support. CRDW orders are not administrations.
MIMIC and CRDW clocks and coding systems differ. Off-policy evaluation with
a learned behavior policy would be sensitive to that missingness and to
overlap. None of those problems is solved by training the GRU.

No policy, no Q function, and no treatment recommender is implemented.

## Next step

Pretrain on authorized MIMIC-IV heart-failure timelines and fine-tune only
after a CRDW outcome has been defined with the same anchor, horizon, and
censoring rule. Until then, a cross-hospital AUROC is not interpretable.
