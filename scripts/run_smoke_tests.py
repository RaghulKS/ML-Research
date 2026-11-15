#!/usr/bin/env python3
"""Deterministic synthetic demonstration of the full research pipeline.

This command trains real models and writes real metrics. Those metrics are
synthetic engineering checks, not clinical results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_ml.experiments import run_smoke, task_config_from_yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic end-to-end research pipeline.")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/experiments/synthetic_smoke")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    baseline = task_config_from_yaml(ROOT / "configs/experiments/baseline_hf_readmission.yaml")
    longitudinal = task_config_from_yaml(ROOT / "configs/experiments/longitudinal_hf_readmission.yaml")
    transfer = task_config_from_yaml(ROOT / "configs/experiments/transfer_mimic_to_crdw.yaml")
    seed = int(args.seed if args.seed is not None else baseline["synthetic"]["seed"])
    neural = dict(longitudinal.get("model", {}))
    neural.update(longitudinal.get("gru", {}))
    transfer_cfg = {
        key: value
        for key, value in transfer.items()
        if key not in {"task", "split", "notes", "freeze_modes", "compare_to_scratch"}
    }
    payload = run_smoke(
        args.output,
        seed=seed,
        n_mimic=int(baseline["synthetic"]["n_mimic"]),
        n_crdw=int(baseline["synthetic"]["n_crdw"]),
        config={
            "task_name": baseline["task"]["name"],
            "horizon_days": baseline["task"]["horizon_days"],
            "baselines": baseline["models"],
            "n_boot": baseline["bootstrap"]["n_boot"],
            "n_repeats": baseline["permutation"]["n_repeats"],
            "neural": neural,
            "transformer_layers": longitudinal["transformer"]["n_layers"],
            "transformer_heads": longitudinal["transformer"]["n_heads"],
            "transformer_transfer_epochs": 2,
            "transfer": transfer_cfg,
        },
    )
    print(payload["note"])
    print(f"Results: {args.output / 'results.json'}")
    for row in payload["baselines"]:
        metrics = row["test_metrics"]
        print(f"baseline {row['model']}: AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} Brier={metrics['brier']:.4f}")
    for key in ("gru_with_treatment", "gru_without_treatment", "transformer_with_treatment"):
        metrics = payload[key]["test_metrics"]
        print(f"{key}: AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} Brier={metrics['brier']:.4f}")
    transfer_metrics = payload["transfer_gru"]
    print(
        "transfer frozen AUROC="
        f"{transfer_metrics['target_frozen']['auroc']:.4f} "
        "full finetune AUROC="
        f"{transfer_metrics['target_finetuned_full']['auroc']:.4f} "
        "scratch AUROC="
        f"{transfer_metrics['target_from_scratch']['auroc']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
