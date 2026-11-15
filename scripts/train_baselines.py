#!/usr/bin/env python3
"""Train logistic regression, random forest, and XGBoost (or the documented fallback)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_ml.experiments import prepare_domain, prepare_task, run_baselines, task_config_from_yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train classical baselines for the readmission task.")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, choices=["mimic", "crdw"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/baseline_hf_readmission.yaml")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-type", default=None)
    args = parser.parse_args(argv)
    config = task_config_from_yaml(args.config)
    task = config["task"]
    domain = prepare_domain(args.source_dir, args.source, dataset_type=args.dataset_type)
    labels = prepare_task(domain, task["cohort"], task["phenotype"], float(task["horizon_days"]), task["name"])
    if labels["summary"]["n_positive"] < 1 or labels["summary"]["n_negative"] < 1:
        raise SystemExit("The labeled cohort does not contain both classes. No metrics were invented.")
    result = run_baselines(
        domain,
        labels,
        seed=int(config.get("split", {}).get("seed", 7)),
        model_config=config.get("models", {}),
        output_dir=args.output,
        n_boot=int(config.get("bootstrap", {}).get("n_boot", 200)),
        n_repeats=int(config.get("permutation", {}).get("n_repeats", 5)),
    )
    print(f"dataset_type={result['dataset_type']}")
    for row in result["models"]:
        metrics = row["test_metrics"]
        print(f"{row['model']}: AUROC={metrics['auroc']} AUPRC={metrics['auprc']} Brier={metrics['brier']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
