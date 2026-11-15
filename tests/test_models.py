"""Metrics, sequences, neural checkpoints, and a small wired pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from clinical_ml.evaluation.interpret import correlation_screen
from clinical_ml.evaluation.metrics import compute_metrics
from clinical_ml.experiments import run_smoke
from clinical_ml.models.neural import LongitudinalClassifier, set_trainable
from clinical_ml.models.sequence import Vocabulary, build_samples, collate_batch
from clinical_ml.models.sequence import ScalarNormalizer
from clinical_ml.training.loop import load_checkpoint, make_loader, predict_proba, save_checkpoint, train_model
import pandas as pd


def test_metrics_match_known_cases():
    perfect = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert perfect["auroc"] == 1.0
    assert perfect["auprc"] == 1.0
    assert perfect["brier"] < 0.05
    mixed = compute_metrics([0, 0, 1, 1], [0.2, 0.8, 0.7, 0.4], threshold=0.5)
    assert mixed["confusion_matrix"] == [[1, 1], [1, 1]]
    assert mixed["sensitivity"] == 0.5
    assert mixed["specificity"] == 0.5
    chance = compute_metrics([0, 1, 0, 1], [0.4, 0.4, 0.4, 0.4])
    assert chance["auroc"] == 0.5
    assert chance["brier"] == np.float64(0.26) or abs(chance["brier"] - 0.26) < 1e-9
    undefined = compute_metrics([0, 0, 0], [0.2, 0.3, 0.4])
    assert undefined["auroc"] is None


def test_correlation_screen_flags_a_copied_label():
    frame = pd.DataFrame({"age": [40, 50, 60, 70], "leak": [0, 0, 1, 1]})
    flags = correlation_screen(frame, [0, 0, 1, 1], threshold=0.98)
    assert [row["feature"] for row in flags] == ["leak"]


def test_vocab_fit_does_not_absorb_test_codes_and_masks_variable_lengths():
    vocab = Vocabulary()
    vocab.fit(["I509", "CREAT"])
    assert vocab.encode("HYDRALAZINE") == 1
    assert vocab.encode("I509") != 1
    normalizer = ScalarNormalizer()
    normalizer.fit_values(["CREAT", "CREAT"], [1.0, 2.0])
    normalizer.fit_age([60, 70])
    events = pd.DataFrame(
        [
            _row("p1", "e", 0, "diagnosis", "I509", 0),
            _row("p1", "e", 1, "lab", "CREAT", 1.2),
            _row("p2", "e", 0, "diagnosis", "I509", 0),
            _row("p2", "e", 1, "medication_order", "MED:LOOP_DIURETIC", 1),
            _row("p2", "e", 2, "lab", "CREAT", 1.4),
        ]
    )
    events["treatment_indicator"] = events["event_type"].isin(["medication_order", "procedure"])
    patients = pd.DataFrame(
        [
            {"patient_id": "p1", "age": 60, "sex": "F", "followup_end_time": 10, "time_origin": "relative_days", "source_dataset": "t"},
            {"patient_id": "p2", "age": 80, "sex": "M", "followup_end_time": 10, "time_origin": "relative_days", "source_dataset": "t"},
        ]
    )
    labeled = pd.DataFrame(
        [
            {"patient_id": "p1", "anchor_time": 1, "time_origin": "relative_days", "label": 0},
            {"patient_id": "p2", "anchor_time": 2, "time_origin": "relative_days", "label": 1},
        ]
    )
    with_tx = build_samples(events, patients, labeled, ["p1", "p2"], vocab, normalizer, 8, True)
    without = build_samples(events, patients, labeled, ["p1", "p2"], vocab, normalizer, 8, False)
    assert sum(sum(sample["treatment"]) for sample in without) == 0
    assert sum(len(sample["tokens"]) for sample in with_tx) > sum(len(sample["tokens"]) for sample in without)
    batch = collate_batch(with_tx)
    assert int(batch["mask"].sum()) == sum(len(sample["tokens"]) for sample in with_tx)
    assert torch.all(batch["mask"][0, int(batch["mask"][0].sum()) :] == 0) or batch["mask"].shape[1] >= 1
    model = LongitudinalClassifier("gru", n_codes=len(vocab), d_model=16, hidden_size=16, n_layers=1, dropout=0.1, max_len=8)
    logits = model(batch["tokens"], batch["types"], batch["treatment"], batch["nums"], batch["mask"], batch["static"])
    assert torch.isfinite(logits).all()
    transformer = LongitudinalClassifier(
        "transformer", n_codes=len(vocab), d_model=16, hidden_size=16, n_layers=1, n_heads=4, dropout=0.1, max_len=8
    )
    t_logits = transformer(batch["tokens"], batch["types"], batch["treatment"], batch["nums"], batch["mask"], batch["static"])
    assert torch.isfinite(t_logits).all()


def test_head_only_freeze_and_checkpoint_roundtrip(tmp_path: Path):
    vocab = Vocabulary()
    vocab.fit(["A", "B", "C"])
    normalizer = ScalarNormalizer()
    normalizer.fit_age([50, 60, 70, 80])
    samples = []
    for i in range(8):
        samples.append(
            {
                "patient_id": f"p{i}",
                "tokens": [vocab.encode("A"), vocab.encode("B")],
                "types": [2, 5],
                "treatment": [0, 0],
                "nums": [[0.0, 1.0, 0.0, 0.1], [0.2, 0.0, 0.1, 0.2]],
                "label": i % 2,
                "static": [0.0, 0.0, 1.0, 0.0],
            }
        )
    loader = make_loader(samples, batch_size=4, shuffle=True)
    val_loader = make_loader(samples, batch_size=4, shuffle=False)
    model = LongitudinalClassifier("gru", n_codes=len(vocab), d_model=16, hidden_size=16, dropout=0.1, max_len=4)
    set_trainable(model, "head_only")
    before = model.event_encoder.code_emb.weight.detach().clone()
    train_model(model, loader, val_loader, epochs=1, lr=1e-2, seed=1, patience=1)
    assert torch.equal(before, model.event_encoder.code_emb.weight)
    assert any(param.requires_grad for name, param in model.named_parameters() if name.startswith("head."))
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, vocab, normalizer, {"d_model": 16, "hidden_size": 16, "max_len": 4, "dropout": 0.1, "n_layers": 1})
    reloaded, re_vocab, _norm, _payload = load_checkpoint(path)
    assert len(re_vocab) == len(vocab)
    first = predict_proba(model, val_loader)[2]
    second = predict_proba(reloaded, val_loader)[2]
    assert np.allclose(first, second, atol=1e-5)


def test_small_smoke_pipeline(tmp_path: Path):
    payload = run_smoke(
        tmp_path / "smoke",
        seed=7,
        n_mimic=48,
        n_crdw=40,
        config={
            "task_name": "hf_30day_all_cause_readmission",
            "horizon_days": 30,
            "baselines": {
                "logistic_regression": {"C_grid": [1.0], "max_iter": 400},
                "random_forest": {"max_depth_grid": [4], "n_estimators": 20, "min_samples_leaf": 2},
                "xgboost": {"max_depth_grid": [2], "n_estimators": 15, "learning_rate": 0.1},
            },
            "n_boot": 20,
            "n_repeats": 2,
            "neural": {
                "d_model": 32,
                "hidden_size": 32,
                "dropout": 0.1,
                "max_len": 24,
                "lr": 1e-3,
                "epochs": 1,
                "batch_size": 16,
                "patience": 1,
                "n_layers": 1,
                "n_heads": 4,
                "mc_dropout_samples": 2,
            },
            "transformer_layers": 1,
            "transformer_heads": 4,
            "transformer_transfer_epochs": 1,
        },
    )
    assert payload["dataset_type"] == "SYNTHETIC ENGINEERING VALIDATION"
    assert (tmp_path / "smoke" / "results.json").exists()
    assert payload["mimic_cohorts"]["heart_failure_expanded_v1"]["n_patients"] > payload["mimic_cohorts"]["heart_failure_historical_v1"]["n_patients"]


def _row(pid, eid, time, event_type, code, treatment):
    return {
        "patient_id": pid,
        "encounter_id": eid,
        "event_time": time,
        "time_origin": "relative_days",
        "event_type": event_type,
        "event_code": code,
        "raw_code": code,
        "event_value": 1.0 if event_type == "lab" else float("nan"),
        "event_unit": "mg/dL" if event_type == "lab" else "",
        "source_dataset": "t",
        "event_description": code,
        "treatment_indicator": bool(treatment) if event_type == "medication_order" else False,
        "administration_confirmed": False,
        "provenance": "t",
        "missingness_indicator": False,
        "coding_system": "ICD10" if event_type == "diagnosis" else "",
    }
