#!/usr/bin/env python3
"""Train a GRU and a compact Transformer, with and without treatment events."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_ml.experiments import prepare_domain, prepare_task, run_neural, task_config_from_yaml
from clinical_ml.preprocessing.splits import patient_disjoint_split
from clinical_ml.tracking import write_experiment
from clinical_ml.utils.seeding import set_seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train longitudinal GRU and Transformer models.")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, choices=["mimic", "crdw"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/longitudinal_hf_readmission.yaml")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-type", default=None)
    args = parser.parse_args(argv)
    config = task_config_from_yaml(args.config)
    task = config["task"]
    domain = prepare_domain(args.source_dir, args.source, dataset_type=args.dataset_type)
    labels = prepare_task(domain, task["cohort"], task["phenotype"], float(task["horizon_days"]), task["name"])
    if labels["summary"]["n_positive"] < 1 or labels["summary"]["n_negative"] < 1:
        raise SystemExit("The labeled cohort does not contain both classes. No metrics were invented.")
    seed = int(config.get("split", {}).get("seed", 7))
    set_seed(seed)
    splits, split_info = patient_disjoint_split(labels["labeled"], seed=seed)
    base = dict(config.get("model", {}))
    runs = []
    for model_type, use_treatment, folder in (
        ("gru", True, "gru_treatment"),
        ("gru", False, "gru_no_treatment"),
        ("transformer", True, "transformer_treatment"),
    ):
        model_config = dict(base)
        extra = config.get(model_type, {})
        model_config["n_layers"] = int(extra.get("n_layers", 1 if model_type == "gru" else 2))
        model_config["n_heads"] = int(extra.get("n_heads", 4))
        runs.append(
            run_neural(
                domain,
                labels,
                splits,
                model_type,
                model_config,
                seed,
                use_treatment,
                args.output / folder,
            )
        )
    payload = {
        "dataset_type": domain["dataset_type"],
        "task": labels["summary"],
        "split_info": split_info,
        "split_sizes": {name: len(ids) for name, ids in splits.items()},
        "runs": runs,
        "note": runs[0]["note"],
    }
    write_experiment(args.output / "longitudinal_results.json", payload)
    print(f"dataset_type={domain['dataset_type']}")
    for row in runs:
        metrics = row["test_metrics"]
        print(
            f"{row['model']} treatment={row['use_treatment']}: "
            f"AUROC={metrics['auroc']} AUPRC={metrics['auprc']} Brier={metrics['brier']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
