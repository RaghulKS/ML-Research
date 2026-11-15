#!/usr/bin/env python3
"""Extract a cohort from a MIMIC-like or CRDW-like directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_ml.cohorts.definitions import get_definition
from clinical_ml.cohorts.extraction import cohort_summary
from clinical_ml.experiments import prepare_domain
from clinical_ml.utils.jsonio import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a patient cohort from a local extract.")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, choices=["mimic", "crdw"])
    parser.add_argument("--cohort", default="heart_failure")
    parser.add_argument("--phenotype", default="historical", choices=["historical", "expanded"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-type", default=None)
    args = parser.parse_args(argv)
    domain = prepare_domain(args.source_dir, args.source, dataset_type=args.dataset_type)
    definition = get_definition(args.cohort, args.phenotype)
    summary = cohort_summary(domain["events_harmonized"], definition)
    summary["dataset_type"] = domain["dataset_type"]
    summary["source"] = args.source
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".json":
        write_json(args.output, summary)
    else:
        # One row per cohort patient, without clinical free text.
        from clinical_ml.cohorts.extraction import cohort_patient_ids
        import pandas as pd

        ids = cohort_patient_ids(domain["events_harmonized"], definition)
        pd.DataFrame({"patient_id": ids, "cohort_version": definition.version}).to_csv(args.output, index=False)
        write_json(args.output.with_suffix(".summary.json"), summary)
    print(f"{summary['cohort_version']}: {summary['n_patients']} patients ({domain['dataset_type']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
