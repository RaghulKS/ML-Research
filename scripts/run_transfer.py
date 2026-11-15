#!/usr/bin/env python3
"""MIMIC-style source training, frozen target evaluation, fine-tuning, and a scratch target model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_ml.experiments import prepare_domain, prepare_task, run_transfer, task_config_from_yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a source-to-target fine-tuning workflow.")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source-name", required=True, choices=["mimic", "crdw"])
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--target-name", required=True, choices=["mimic", "crdw"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/transfer_mimic_to_crdw.yaml")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-type", default=None)
    args = parser.parse_args(argv)
    config = task_config_from_yaml(args.config)
    task = config["task"]
    neural = {key: value for key, value in config.items() if key not in {"task", "split", "notes", "freeze_modes", "compare_to_scratch"}}
    source = prepare_domain(args.source_dir, args.source_name, dataset_type=args.dataset_type)
    target = prepare_domain(args.target_dir, args.target_name, dataset_type=args.dataset_type)
    source_labels = prepare_task(source, task["cohort"], task["phenotype"], float(task["horizon_days"]), task["name"])
    target_labels = prepare_task(target, task["cohort"], task["phenotype"], float(task["horizon_days"]), task["name"])
    result = run_transfer(
        source,
        target,
        source_labels,
        target_labels,
        neural,
        seed=int(config.get("split", {}).get("seed", 7)),
        output_dir=args.output,
    )
    print(f"dataset_type={result['dataset_type']}")
    for name in (
        "source_heldout",
        "target_frozen",
        "target_finetuned_full",
        "target_finetuned_head_only",
        "target_from_scratch",
    ):
        metrics = result[name]
        print(f"{name}: AUROC={metrics['auroc']} AUPRC={metrics['auprc']} Brier={metrics['brier']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
