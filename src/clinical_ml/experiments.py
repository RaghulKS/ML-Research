"""End-to-end cohort, baseline, longitudinal, and transfer experiments.

Test labels are not used to fit imputers, vocabularies, normalizers, or
hyperparameters. Synthetic provenance cannot be relabeled as a clinical result.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from clinical_ml.cohorts.definitions import get_definition
from clinical_ml.cohorts.extraction import cohort_patient_ids, cohort_summary
from clinical_ml.data.adapters import load_source
from clinical_ml.data.quality import coding_overlap_report, quality_report
from clinical_ml.data.synthetic import generate_domain, write_domain
from clinical_ml.errors import IncompatibleTaskError
from clinical_ml.evaluation.interpret import correlation_screen, impurity_importance_table, permutation_importance_table
from clinical_ml.evaluation.metrics import best_f1_threshold, bootstrap_ci, compute_metrics
from clinical_ml.features.tabular import FEATURE_COLUMNS, build_feature_frame, drop_all_nan_columns
from clinical_ml.models.baselines import fit_selected_baseline, predict_proba
from clinical_ml.models.neural import LongitudinalClassifier, model_kwargs_from_config, resize_code_embeddings, set_trainable
from clinical_ml.models.sequence import Vocabulary, build_samples, pre_anchor_events, unk_rate
from clinical_ml.models.sequence import ScalarNormalizer
from clinical_ml.preprocessing.harmonize import harmonize_events, load_harmonization
from clinical_ml.preprocessing.labels import IMPLEMENTATION_ID, build_readmission_labels
from clinical_ml.preprocessing.splits import assert_no_temporal_leakage, assert_patient_disjoint, patient_disjoint_split
from clinical_ml.reporting import SYNTHETIC, resolve_dataset_type
from clinical_ml.tracking import environment_snapshot, write_experiment
from clinical_ml.training.loop import load_checkpoint, make_loader, predict_mc_dropout, predict_proba as neural_predict, save_checkpoint, train_model
from clinical_ml.uncertainty import calibration_comparison
from clinical_ml.utils.seeding import set_seed

CONDITIONS = ("heart_failure", "stroke", "myocardial_infarction")


def prepare_domain(source_dir: Path | str, source: str, harmonization_path: Path | str | None = None, dataset_type: str | None = None) -> dict:
    loaded = load_source(source_dir, source)
    reporting = resolve_dataset_type(dataset_type, loaded.provenance)
    spec = load_harmonization(harmonization_path)
    harmonized, harmonization_report = harmonize_events(loaded.events, spec)
    summaries = {}
    for condition in CONDITIONS:
        for phenotype in ("historical", "expanded"):
            definition = get_definition(condition, phenotype)
            summaries[definition.version] = cohort_summary(harmonized, definition)
    return {
        "source": source,
        "dataset_type": reporting,
        "events": loaded.events,
        "events_harmonized": harmonized,
        "patients": loaded.patients,
        "encounters": loaded.encounters,
        "cleaning": loaded.cleaning,
        "warnings": loaded.warnings,
        "files_loaded": loaded.files_loaded,
        "provenance": loaded.provenance,
        "quality": quality_report(loaded.events, loaded.patients),
        "coding_overlap": coding_overlap_report(loaded.events),
        "harmonization_report": harmonization_report,
        "cohort_summaries": summaries,
        "harmonization_path": spec["path"],
    }


def prepare_task(domain: dict, cohort: str, phenotype: str, horizon_days: float, task_name: str) -> dict:
    definition = get_definition(cohort, phenotype)
    ids = cohort_patient_ids(domain["events_harmonized"], definition)
    labels = build_readmission_labels(
        domain["events_harmonized"],
        domain["patients"],
        domain["encounters"],
        ids,
        definition,
        horizon_days,
        task_name,
    )
    if labels["summary"].get("implementation") != IMPLEMENTATION_ID:
        raise IncompatibleTaskError("Label builder implementation drifted from readmission_v1.")
    return labels


def assert_same_task(left: dict, right: dict) -> None:
    keys = ["task", "implementation", "cohort_version", "horizon_days"]
    diffs = {}
    for key in keys:
        if left["summary"].get(key) != right["summary"].get(key):
            diffs[key] = (left["summary"].get(key), right["summary"].get(key))
    if diffs:
        raise IncompatibleTaskError(f"Outcome definitions differ: {diffs}")


def _feature_splits(domain: dict, labeled: pd.DataFrame, splits: dict[str, list[str]]) -> dict:
    features = build_feature_frame(
        domain["events_harmonized"], domain["patients"], domain["encounters"], labeled
    )
    assert_no_temporal_leakage(features, labeled)
    features = features.merge(labeled[["patient_id", "label"]], on="patient_id", how="left")
    matrices = {}
    for name, ids in splits.items():
        subset = features[features["patient_id"].isin(ids)].copy()
        matrices[name] = {
            "ids": subset["patient_id"].tolist(),
            "x": subset[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce"),
            "y": subset["label"].astype(int).to_numpy(),
            "frame": subset,
        }
    train_x, others, dropped = drop_all_nan_columns(
        matrices["train"]["x"],
        [matrices["validation"]["x"], matrices["test"]["x"]],
    )
    matrices["train"]["x"] = train_x
    matrices["validation"]["x"] = others[0]
    matrices["test"]["x"] = others[1]
    matrices["dropped_all_nan_columns"] = dropped
    matrices["features"] = features
    return matrices


def run_baselines(domain: dict, labels: dict, seed: int, model_config: dict, output_dir: Path, n_boot: int = 200, n_repeats: int = 5) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    labeled = labels["labeled"]
    if labeled.empty or labeled["label"].nunique() < 2:
        raise ValueError("Baseline training needs both outcome classes.")
    set_seed(seed)
    splits, split_info = patient_disjoint_split(labeled, seed=seed)
    matrices = _feature_splits(domain, labeled, splits)
    screen = correlation_screen(matrices["train"]["x"], matrices["train"]["y"])
    results = []
    models = {}
    started = time.perf_counter()
    for family in ("logistic_regression", "random_forest", "xgboost"):
        t0 = time.perf_counter()
        fitted = fit_selected_baseline(
            family,
            model_config.get(family, {}),
            matrices["train"]["x"],
            matrices["train"]["y"],
            matrices["validation"]["x"],
            matrices["validation"]["y"],
            seed,
        )
        test_p = predict_proba(fitted["model"], matrices["test"]["x"])
        val_p = predict_proba(fitted["model"], matrices["validation"]["x"])
        threshold = best_f1_threshold(matrices["validation"]["y"], val_p)
        test_metrics = compute_metrics(matrices["test"]["y"], test_p, threshold=0.5)
        tuned = compute_metrics(matrices["test"]["y"], test_p, threshold=threshold)
        calibration = calibration_comparison(matrices["validation"]["y"], val_p, matrices["test"]["y"], test_p)
        cis = {
            "auroc": bootstrap_ci(matrices["test"]["y"], test_p, "auroc", n_boot=n_boot, seed=seed),
            "auprc": bootstrap_ci(matrices["test"]["y"], test_p, "auprc", n_boot=n_boot, seed=seed),
        }
        importance = None
        impurity = None
        if len(np.unique(matrices["validation"]["y"])) > 1:
            importance = permutation_importance_table(
                fitted["model"], matrices["validation"]["x"], matrices["validation"]["y"], seed=seed, n_repeats=n_repeats
            )
            importance.to_csv(output_dir / f"permutation_importance_{fitted['spec'].name}.csv", index=False)
            impurity = impurity_importance_table(fitted["model"], list(matrices["train"]["x"].columns))
            if impurity is not None:
                impurity.to_csv(output_dir / f"impurity_importance_{fitted['spec'].name}.csv", index=False)
        model_path = output_dir / "models" / f"{fitted['spec'].name}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": fitted["model"],
                "feature_columns": list(matrices["train"]["x"].columns),
                "dataset_type": domain["dataset_type"],
                "spec": fitted["spec"].__dict__,
            },
            model_path,
        )
        reloaded = joblib.load(model_path)["model"]
        reloaded_p = predict_proba(reloaded, matrices["test"]["x"])
        if not np.allclose(test_p, reloaded_p, atol=1e-6):
            raise AssertionError(f"Reloaded {fitted['spec'].name} predictions changed.")
        record = {
            "model": fitted["spec"].name,
            "requested_family": family,
            "fallback": fitted["spec"].fallback,
            "hyperparameters": fitted["spec"].params,
            "validation_metrics_for_selection": fitted["validation_metrics"],
            "test_metrics": test_metrics,
            "test_metrics_at_validation_f1_threshold": tuned,
            "bootstrap_ci": cis,
            "calibration": calibration,
            "train_seconds": time.perf_counter() - t0,
            "model_path": str(model_path),
            "reload_max_abs_diff": float(np.max(np.abs(test_p - reloaded_p))),
        }
        results.append(record)
        models[fitted["spec"].name] = fitted["model"]
    payload = {
        "dataset_type": domain["dataset_type"],
        "task": labels["summary"],
        "split_info": split_info,
        "split_sizes": {name: len(ids) for name, ids in splits.items()},
        "n_features": int(matrices["train"]["x"].shape[1]),
        "dropped_all_nan_columns": matrices["dropped_all_nan_columns"],
        "leakage_screen": screen,
        "models": results,
        "seconds": time.perf_counter() - started,
        "note": _result_note(domain["dataset_type"]),
    }
    write_experiment(output_dir / "baseline_results.json", payload)
    pd.DataFrame(
        [
            {
                "model": row["model"],
                "dataset_type": domain["dataset_type"],
                "auroc": row["test_metrics"]["auroc"],
                "auprc": row["test_metrics"]["auprc"],
                "brier": row["test_metrics"]["brier"],
                "f1": row["test_metrics"]["f1"],
                "precision": row["test_metrics"]["precision"],
                "recall": row["test_metrics"]["recall"],
                "sensitivity": row["test_metrics"]["sensitivity"],
                "specificity": row["test_metrics"]["specificity"],
                "n": row["test_metrics"]["n"],
            }
            for row in results
        ]
    ).to_csv(output_dir / "baseline_metrics.csv", index=False)
    payload["splits"] = splits
    payload["matrices_ids"] = {name: matrices[name]["ids"] for name in ("train", "validation", "test")}
    return payload


def _fit_normalizer(domain, labeled, train_ids, use_treatment: bool) -> tuple[Vocabulary, ScalarNormalizer]:
    history = pre_anchor_events(domain["events_harmonized"], labeled, train_ids)
    if not use_treatment and len(history):
        history = history[~history["treatment_indicator"].astype(bool)]
    vocab = Vocabulary()
    vocab.fit(history["event_code"].tolist() if len(history) else [])
    normalizer = ScalarNormalizer()
    if len(history):
        numeric = history[history["event_type"].isin(["lab", "vital"])]
        normalizer.fit_values(numeric["event_code"].tolist(), numeric["event_value"].tolist())
    ages = domain["patients"].set_index("patient_id")["age"].reindex(train_ids)
    normalizer.fit_age(ages.tolist())
    return vocab, normalizer


def _samples(domain, labeled, ids, vocab, normalizer, max_len, use_treatment):
    return build_samples(
        domain["events_harmonized"],
        domain["patients"],
        labeled,
        ids,
        vocab,
        normalizer,
        max_len,
        use_treatment,
    )


def run_neural(
    domain: dict,
    labels: dict,
    splits: dict[str, list[str]],
    model_type: str,
    config: dict,
    seed: int,
    use_treatment: bool,
    output_dir: Path,
    freeze_mode: str = "full",
    init_checkpoint: Path | None = None,
    vocab: Vocabulary | None = None,
    normalizer: ScalarNormalizer | None = None,
    extend_vocab_from_train: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    labeled = labels["labeled"]
    if vocab is None or normalizer is None:
        vocab, normalizer = _fit_normalizer(domain, labeled, splits["train"], use_treatment)
    if extend_vocab_from_train:
        history = pre_anchor_events(domain["events_harmonized"], labeled, splits["train"])
        vocab.add(history["event_code"].tolist() if len(history) else [])
    samples = {
        name: _samples(domain, labeled, ids, vocab, normalizer, int(config.get("max_len", 32)), use_treatment)
        for name, ids in splits.items()
    }
    for name, rows in samples.items():
        if len(rows) != len(splits[name]):
            missing = set(splits[name]) - {row["patient_id"] for row in rows}
            raise ValueError(f"{model_type} dropped patients while building {name} sequences: {sorted(missing)[:5]}")
    batch_size = int(config.get("batch_size", 32))
    loaders = {name: make_loader(rows, batch_size, shuffle=(name == "train")) for name, rows in samples.items()}
    if init_checkpoint is not None:
        model, _saved_vocab, _saved_norm, _payload = load_checkpoint(init_checkpoint)
        resize_code_embeddings(model, len(vocab))
    else:
        kwargs = model_kwargs_from_config(model_type, len(vocab), config)
        model = LongitudinalClassifier(**kwargs)
    set_trainable(model, freeze_mode)
    t0 = time.perf_counter()
    history = train_model(
        model,
        loaders["train"],
        loaders["validation"],
        epochs=int(config.get("epochs", 4)),
        lr=float(config.get("lr", 1e-3)),
        seed=seed,
        patience=int(config.get("patience", 3)),
    )
    train_seconds = time.perf_counter() - t0
    test_ids, test_y, test_p = neural_predict(model, loaders["test"])
    val_ids, val_y, val_p = neural_predict(model, loaders["validation"])
    order = np.argsort(np.asarray(test_ids))
    test_ids = [test_ids[i] for i in order]
    test_y = test_y[order]
    test_p = test_p[order]
    test_metrics = compute_metrics(test_y, test_p)
    calibration = calibration_comparison(val_y, val_p, test_y, test_p)
    mc = predict_mc_dropout(model, loaders["test"], n_samples=int(config.get("mc_dropout_samples", 8)))
    mc_metrics = compute_metrics(mc["y"], mc["probability_mean"])
    checkpoint = output_dir / "models" / f"{model_type}_{'treatment' if use_treatment else 'no_treatment'}_{freeze_mode}.pt"
    save_checkpoint(
        checkpoint,
        model,
        vocab,
        normalizer,
        config,
        extra={"dataset_type": domain["dataset_type"], "use_treatment": use_treatment, "freeze_mode": freeze_mode},
    )
    reloaded, re_vocab, re_norm, _payload = load_checkpoint(checkpoint)
    re_samples = _samples(domain, labeled, splits["test"], re_vocab, re_norm, int(config.get("max_len", 32)), use_treatment)
    re_loader = make_loader(re_samples, batch_size, shuffle=False)
    re_ids, _re_y, re_p = neural_predict(reloaded, re_loader)
    re_order = np.argsort(np.asarray(re_ids))
    re_p = re_p[re_order]
    if not np.allclose(test_p, re_p, atol=1e-5):
        raise AssertionError(f"Reloaded {model_type} predictions changed.")
    return {
        "model": model_type,
        "dataset_type": domain["dataset_type"],
        "use_treatment": use_treatment,
        "freeze_mode": freeze_mode,
        "n_parameters": int(sum(param.numel() for param in model.parameters())),
        "n_trainable": int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
        "vocab_size": len(vocab),
        "unk_rate_test": unk_rate(samples["test"]),
        "training": history,
        "train_seconds": train_seconds,
        "test_metrics": test_metrics,
        "calibration": calibration,
        "mc_dropout": {
            "mean_predictive_std": mc["mean_predictive_std"],
            "brier_of_mean": mc_metrics["brier"],
            "auroc_of_mean": mc_metrics["auroc"],
            "note": "MC dropout is not out-of-distribution detection.",
        },
        "checkpoint": str(checkpoint),
        "reload_max_abs_diff": float(np.max(np.abs(test_p - re_p))),
        "note": _result_note(domain["dataset_type"]),
    }


def run_transfer(source: dict, target: dict, source_labels: dict, target_labels: dict, config: dict, seed: int, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    assert_same_task(source_labels, target_labels)
    if source["dataset_type"] == SYNTHETIC or target["dataset_type"] == SYNTHETIC:
        reporting = SYNTHETIC
    else:
        reporting = source["dataset_type"]
    set_seed(seed)
    source_splits, _ = patient_disjoint_split(source_labels["labeled"], seed=seed)
    target_splits, _ = patient_disjoint_split(target_labels["labeled"], seed=seed)
    model_type = config.get("model", "gru")
    neural_config = dict(config)
    # Source model.
    source_run = run_neural(
        source,
        source_labels,
        source_splits,
        model_type,
        neural_config,
        seed,
        use_treatment=True,
        output_dir=output_dir / "source",
    )
    source_vocab, source_norm = _fit_normalizer(source, source_labels["labeled"], source_splits["train"], True)
    frozen_samples = _samples(
        target,
        target_labels["labeled"],
        target_splits["test"],
        source_vocab,
        source_norm,
        int(neural_config.get("max_len", 32)),
        True,
    )
    frozen_loader = make_loader(frozen_samples, int(neural_config.get("batch_size", 32)), shuffle=False)
    frozen_model, _, _, _ = load_checkpoint(source_run["checkpoint"])
    _ids, frozen_y, frozen_p = neural_predict(frozen_model, frozen_loader)
    frozen_metrics = compute_metrics(frozen_y, frozen_p)
    # Full fine-tune extends the source vocabulary with target training codes only.
    full_config = dict(neural_config)
    full_config["epochs"] = int(config.get("finetune_epochs", neural_config.get("epochs", 4)))
    full_run = run_neural(
        target,
        target_labels,
        target_splits,
        model_type,
        full_config,
        seed,
        use_treatment=True,
        output_dir=output_dir / "finetune_full",
        freeze_mode="full",
        init_checkpoint=Path(source_run["checkpoint"]),
        vocab=source_vocab,
        normalizer=source_norm,
        extend_vocab_from_train=True,
    )
    head_vocab, _ = _fit_normalizer(source, source_labels["labeled"], source_splits["train"], True)
    head_run = run_neural(
        target,
        target_labels,
        target_splits,
        model_type,
        full_config,
        seed,
        use_treatment=True,
        output_dir=output_dir / "finetune_head",
        freeze_mode="head_only",
        init_checkpoint=Path(source_run["checkpoint"]),
        vocab=head_vocab,
        normalizer=source_norm,
        extend_vocab_from_train=True,
    )
    scratch_config = dict(neural_config)
    scratch_config["epochs"] = int(config.get("finetune_epochs", neural_config.get("epochs", 4)))
    scratch_run = run_neural(
        target,
        target_labels,
        target_splits,
        model_type,
        scratch_config,
        seed,
        use_treatment=True,
        output_dir=output_dir / "scratch",
    )
    payload = {
        "dataset_type": reporting,
        "task": source_labels["summary"],
        "source_heldout": source_run["test_metrics"],
        "target_frozen": frozen_metrics,
        "target_finetuned_full": full_run["test_metrics"],
        "target_finetuned_head_only": head_run["test_metrics"],
        "target_from_scratch": scratch_run["test_metrics"],
        "unk_rate_target_test_under_source_vocab": unk_rate(frozen_samples),
        "source_train_seconds": source_run["train_seconds"],
        "finetune_full_seconds": full_run["train_seconds"],
        "finetune_head_seconds": head_run["train_seconds"],
        "scratch_seconds": scratch_run["train_seconds"],
        "n_parameters": source_run["n_parameters"],
        "compatible_task": True,
        "harmonization": {
            "source": source["harmonization_report"],
            "target": target["harmonization_report"],
        },
        "caveat": (
            "Cross-domain numbers require the same outcome definition and comparable "
            "ascertainment. These synthetic domains satisfy that by construction. "
            "Real MIMIC-IV versus CRDW comparisons do not until that is demonstrated. "
            "Differences are observational associations, not treatment effects."
        ),
        "note": _result_note(reporting),
    }
    write_experiment(output_dir / "transfer_results.json", payload)
    return payload


def run_smoke(output_dir: Path | str, seed: int = 7, n_mimic: int = 180, n_crdw: int = 150, config: dict | None = None) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    config = config or {}
    mimic_dir = write_domain(generate_domain("mimic", n_mimic, seed), output_dir / "data" / "mimic")
    crdw_dir = write_domain(generate_domain("crdw", n_crdw, seed), output_dir / "data" / "crdw")
    mimic = prepare_domain(mimic_dir, "mimic")
    crdw = prepare_domain(crdw_dir, "crdw")
    if mimic["dataset_type"] != SYNTHETIC or crdw["dataset_type"] != SYNTHETIC:
        raise AssertionError("Synthetic provenance was not preserved.")
    hf_hist = mimic["cohort_summaries"]["heart_failure_historical_v1"]["n_patients"]
    hf_exp = mimic["cohort_summaries"]["heart_failure_expanded_v1"]["n_patients"]
    if hf_exp <= hf_hist:
        raise AssertionError("Expanded heart-failure definition did not add patients beyond the historical list.")
    task_name = config.get("task_name", "hf_30day_all_cause_readmission")
    horizon = float(config.get("horizon_days", 30))
    mimic_labels = prepare_task(mimic, "heart_failure", "historical", horizon, task_name)
    crdw_labels = prepare_task(crdw, "heart_failure", "historical", horizon, task_name)
    for labels in (mimic_labels, crdw_labels):
        summary = labels["summary"]
        if summary["n_positive"] < 1 or summary["n_negative"] < 1:
            raise AssertionError(f"Both classes were not present: {summary}")
        if summary["n_censored_insufficient_followup"] < 1:
            raise AssertionError("Censoring path was not exercised.")
    baseline_config = config.get(
        "baselines",
        {
            "logistic_regression": {"C_grid": [0.1, 1.0, 3.0], "max_iter": 1000},
            "random_forest": {"max_depth_grid": [4, 6], "n_estimators": 200, "min_samples_leaf": 3},
            "xgboost": {"max_depth_grid": [2, 3], "n_estimators": 80, "learning_rate": 0.08, "subsample": 0.9, "colsample_bytree": 0.9},
        },
    )
    neural_config = config.get(
        "neural",
        {
            "d_model": 48,
            "hidden_size": 48,
            "dropout": 0.2,
            "max_len": 32,
            "lr": 1e-3,
            "epochs": 4,
            "batch_size": 32,
            "patience": 3,
            "n_layers": 1,
            "n_heads": 4,
            "mc_dropout_samples": 8,
        },
    )
    baselines = run_baselines(
        mimic,
        mimic_labels,
        seed,
        baseline_config,
        output_dir / "baselines_mimic",
        n_boot=int(config.get("n_boot", 200)),
        n_repeats=int(config.get("n_repeats", 5)),
    )
    splits = baselines["splits"]
    assert_patient_disjoint(splits)
    gru = run_neural(mimic, mimic_labels, splits, "gru", neural_config, seed, True, output_dir / "gru_treatment")
    gru_config_no_tx = dict(neural_config)
    gru_no_tx = run_neural(mimic, mimic_labels, splits, "gru", gru_config_no_tx, seed, False, output_dir / "gru_no_treatment")
    transformer_config = dict(neural_config)
    transformer_config["n_layers"] = int(config.get("transformer_layers", 2))
    transformer_config["n_heads"] = int(config.get("transformer_heads", 4))
    transformer = run_neural(
        mimic, mimic_labels, splits, "transformer", transformer_config, seed, True, output_dir / "transformer_treatment"
    )
    if gru["test_metrics"]["n"] != gru_no_tx["test_metrics"]["n"]:
        raise AssertionError("Treatment ablation did not keep the same patients.")
    transfer_config = dict(neural_config)
    transfer_config.update(config.get("transfer", {}))
    transfer_config["model"] = "gru"
    transfer = run_transfer(mimic, crdw, mimic_labels, crdw_labels, transfer_config, seed, output_dir / "transfer")
    # A one-step transformer transfer confirms the same workflow is not GRU-specific.
    transformer_transfer_config = dict(transformer_config)
    transformer_transfer_config["model"] = "transformer"
    transformer_transfer_config["epochs"] = int(config.get("transformer_transfer_epochs", 2))
    transformer_transfer_config["finetune_epochs"] = int(config.get("transformer_transfer_epochs", 2))
    transformer_transfer = run_transfer(
        mimic,
        crdw,
        mimic_labels,
        crdw_labels,
        transformer_transfer_config,
        seed,
        output_dir / "transfer_transformer",
    )
    comparison = _comparison_table(baselines, [gru, gru_no_tx, transformer], transfer)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    payload = {
        "dataset_type": SYNTHETIC,
        "seed": seed,
        "n_mimic": n_mimic,
        "n_crdw": n_crdw,
        "environment": environment_snapshot(),
        "mimic_quality": mimic["quality"],
        "crdw_quality": crdw["quality"],
        "mimic_cleaning": mimic["cleaning"],
        "crdw_cleaning": crdw["cleaning"],
        "mimic_cohorts": mimic["cohort_summaries"],
        "crdw_cohorts": crdw["cohort_summaries"],
        "mimic_coding_overlap": mimic["coding_overlap"],
        "crdw_coding_overlap": crdw["coding_overlap"],
        "mimic_harmonization": mimic["harmonization_report"],
        "crdw_harmonization": crdw["harmonization_report"],
        "mimic_task": mimic_labels["summary"],
        "crdw_task": crdw_labels["summary"],
        "mimic_warnings": mimic["warnings"],
        "crdw_warnings": crdw["warnings"],
        "baselines": [{k: v for k, v in row.items()} for row in baselines["models"]],
        "gru_with_treatment": gru,
        "gru_without_treatment": gru_no_tx,
        "transformer_with_treatment": transformer,
        "transfer_gru": transfer,
        "transfer_transformer": transformer_transfer,
        "leakage_screen": baselines["leakage_screen"],
        "split_sizes": baselines["split_sizes"],
        "note": _result_note(SYNTHETIC),
    }
    write_experiment(output_dir / "results.json", payload)
    _assert_smoke(payload)
    return payload


def _comparison_table(baselines: dict, neural_runs: list[dict], transfer: dict) -> pd.DataFrame:
    rows = []
    for row in baselines["models"]:
        rows.append(
            {
                "model": row["model"],
                "dataset_type": baselines["dataset_type"],
                "evaluation": "mimic_heldout_test",
                "auroc": row["test_metrics"]["auroc"],
                "auprc": row["test_metrics"]["auprc"],
                "brier": row["test_metrics"]["brier"],
                "train_seconds": row["train_seconds"],
                "use_treatment": "tabular_features_include_orders_and_admins",
            }
        )
    for row in neural_runs:
        rows.append(
            {
                "model": row["model"],
                "dataset_type": row["dataset_type"],
                "evaluation": "mimic_heldout_test",
                "auroc": row["test_metrics"]["auroc"],
                "auprc": row["test_metrics"]["auprc"],
                "brier": row["test_metrics"]["brier"],
                "train_seconds": row["train_seconds"],
                "use_treatment": row["use_treatment"],
            }
        )
    for name, metrics in (
        ("transfer_source_heldout", transfer["source_heldout"]),
        ("transfer_target_frozen", transfer["target_frozen"]),
        ("transfer_target_full_finetune", transfer["target_finetuned_full"]),
        ("transfer_target_head_only", transfer["target_finetuned_head_only"]),
        ("transfer_target_scratch", transfer["target_from_scratch"]),
    ):
        rows.append(
            {
                "model": name,
                "dataset_type": transfer["dataset_type"],
                "evaluation": name,
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "brier": metrics["brier"],
                "train_seconds": None,
                "use_treatment": True,
            }
        )
    return pd.DataFrame(rows)


def _assert_smoke(payload: dict) -> None:
    if payload["dataset_type"] != SYNTHETIC:
        raise AssertionError("Smoke results must stay labeled as synthetic.")
    if payload["mimic_cleaning"]["dropped_missing_patient_id"] < 1:
        raise AssertionError("Missing patient ids were not dropped.")
    if payload["mimic_cleaning"]["dropped_duplicate_rows"] < 1:
        raise AssertionError("Duplicate rows were not dropped.")
    for row in payload["baselines"]:
        if row["test_metrics"]["auroc"] is None or row["test_metrics"]["brier"] is None:
            raise AssertionError(f"Baseline metrics missing for {row['model']}")
        if row["reload_max_abs_diff"] > 1e-6:
            raise AssertionError("Baseline reload drift.")
    for key in ("gru_with_treatment", "gru_without_treatment", "transformer_with_treatment"):
        metrics = payload[key]["test_metrics"]
        if metrics["auroc"] is None or not np.isfinite(metrics["brier"]):
            raise AssertionError(f"Neural metrics missing for {key}")
    if payload["gru_with_treatment"]["use_treatment"] is not True:
        raise AssertionError("Treatment-aware run was not marked.")
    if payload["gru_without_treatment"]["use_treatment"] is not False:
        raise AssertionError("Treatment-excluded run was not marked.")
    if "target_frozen" not in payload["transfer_gru"]:
        raise AssertionError("Transfer workflow did not evaluate the frozen source model.")


def _result_note(dataset_type: str) -> str:
    if dataset_type == SYNTHETIC:
        return (
            "SYNTHETIC ENGINEERING VALIDATION. These numbers check that the pipeline "
            "runs. They are not clinical performance and they are not estimates for "
            "MIMIC-IV or CRDW patients."
        )
    return (
        "Interpret this run only under its dataset_type. Feature importance is not "
        "a causal effect, and medication orders are not confirmed administrations."
    )


def task_config_from_yaml(path: Path | str) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
