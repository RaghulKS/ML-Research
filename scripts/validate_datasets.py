#!/usr/bin/env python3
"""Write a data-quality report for one local extract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinical_ml.experiments import prepare_domain
from clinical_ml.utils.jsonio import write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a MIMIC-like or CRDW-like directory.")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, choices=["mimic", "crdw"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-type", default=None)
    args = parser.parse_args(argv)
    domain = prepare_domain(args.source_dir, args.source, dataset_type=args.dataset_type)
    payload = {
        "dataset_type": domain["dataset_type"],
        "source": args.source,
        "files_loaded": domain["files_loaded"],
        "cleaning": domain["cleaning"],
        "warnings": domain["warnings"],
        "quality": domain["quality"],
        "coding_overlap": domain["coding_overlap"],
        "harmonization": domain["harmonization_report"],
        "cohort_summaries": domain["cohort_summaries"],
        "provenance": domain["provenance"],
    }
    write_json(args.output, payload)
    print(f"Wrote {args.output} ({domain['dataset_type']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
