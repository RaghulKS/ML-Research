"""Local experiment metadata."""

from __future__ import annotations

import importlib.metadata
import platform
from datetime import datetime, timezone
from pathlib import Path

from clinical_ml import __version__
from clinical_ml.utils.jsonio import write_json


def environment_snapshot() -> dict:
    packages = {}
    for name in ["numpy", "pandas", "scikit-learn", "xgboost", "torch", "pyarrow", "PyYAML", "joblib"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_version": __version__,
        "packages": packages,
    }


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_experiment(path: Path | str, payload: dict) -> None:
    payload = dict(payload)
    payload.setdefault("created_at_utc", stamp())
    payload.setdefault("environment", environment_snapshot())
    write_json(path, payload)
