"""Unit and code harmonization.

Mappings are explicit. A shared name is not treated as shared clinical meaning
unless it appears in the harmonization config. Diagnosis codes are normalized
but are not remapped to a different disease.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import yaml

from clinical_ml.data.schemas import normalize_icd, normalize_unit, normalize_unit_key


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_harmonization(path: Path | str | None = None) -> dict:
    if path is None:
        path = repo_root() / "configs" / "harmonization" / "default.yaml"
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    code_map = {str(k).strip().upper(): str(v).strip().upper() for k, v in raw.get("code_map", {}).items()}
    unit_if_missing = {str(k).strip().upper(): normalize_unit(v) for k, v in raw.get("unit_if_missing", {}).items()}
    canonical_unit = {str(k).strip().upper(): normalize_unit(v) for k, v in raw.get("canonical_unit", {}).items()}
    multipliers: dict[str, dict[str, float]] = {}
    for analyte, table in raw.get("to_canonical_multiplier", {}).items():
        multipliers[str(analyte).upper()] = {normalize_unit_key(unit): float(factor) for unit, factor in table.items()}
    ranges = {}
    for analyte, bounds in raw.get("plausible_range", {}).items():
        ranges[str(analyte).upper()] = (float(bounds[0]), float(bounds[1]))
    return {
        "code_map": code_map,
        "unit_if_missing": unit_if_missing,
        "canonical_unit": canonical_unit,
        "multipliers": multipliers,
        "ranges": ranges,
        "path": str(path),
    }


def harmonize_events(events: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict]:
    frame = events.copy()
    report = {
        "n_codes_mapped": 0,
        "n_units_imputed_from_dictionary": 0,
        "n_units_converted": 0,
        "n_unit_unconverted": 0,
        "n_invalid_values": 0,
    }
    if frame.empty:
        return frame, report

    is_dx = frame["event_type"].eq("diagnosis")
    if is_dx.any():
        frame.loc[is_dx, "event_code"] = frame.loc[is_dx, "event_code"].map(normalize_icd)

    mappable = frame["event_type"].isin(["lab", "vital", "medication_order", "medication_admin", "procedure"])
    if mappable.any():
        original = frame.loc[mappable, "event_code"].astype(str).str.strip().str.upper()
        mapped = original.map(spec["code_map"])
        changed = mapped.notna() & mapped.ne(original)
        report["n_codes_mapped"] = int(changed.sum())
        frame.loc[mappable, "event_code"] = mapped.fillna(original).to_numpy()

    numeric = frame["event_type"].isin(["lab", "vital"])
    if not numeric.any():
        return frame, report

    codes = frame["event_code"].astype(str)
    units = frame["event_unit"].map(normalize_unit)
    missing_unit = numeric & units.eq("")
    imputed_units = codes.map(spec["unit_if_missing"])
    can_impute = missing_unit & imputed_units.notna()
    units = units.mask(can_impute, imputed_units)
    report["n_units_imputed_from_dictionary"] = int(can_impute.sum())
    frame["event_unit"] = units

    values = pd.to_numeric(frame["event_value"], errors="coerce")
    new_values = values.copy()
    new_units = units.copy()
    converted = 0
    unconverted = 0
    invalid = 0
    for idx in frame.index[numeric]:
        analyte = str(frame.at[idx, "event_code"])
        value = new_values.at[idx]
        unit = new_units.at[idx]
        if analyte not in spec["canonical_unit"]:
            continue
        target_unit = spec["canonical_unit"][analyte]
        factor_table = spec["multipliers"].get(analyte, {})
        key = normalize_unit_key(unit)
        if key == "":
            new_values.at[idx] = math.nan
            frame.at[idx, "missingness_indicator"] = True
            frame.at[idx, "provenance"] = str(frame.at[idx, "provenance"]) + "|unit_missing"
            unconverted += 1
            continue
        if key not in factor_table:
            new_values.at[idx] = math.nan
            frame.at[idx, "missingness_indicator"] = True
            frame.at[idx, "provenance"] = str(frame.at[idx, "provenance"]) + "|unit_unconverted"
            unconverted += 1
            continue
        factor = factor_table[key]
        if pd.notna(value) and factor != 1.0:
            value = float(value) * factor
            converted += 1
        new_units.at[idx] = target_unit
        low, high = spec["ranges"].get(analyte, (None, None))
        if pd.notna(value) and low is not None and (value < low or value > high):
            new_values.at[idx] = math.nan
            frame.at[idx, "missingness_indicator"] = True
            frame.at[idx, "provenance"] = str(frame.at[idx, "provenance"]) + "|invalid_value"
            invalid += 1
        else:
            new_values.at[idx] = value
    frame["event_value"] = new_values
    frame["event_unit"] = new_units
    report["n_units_converted"] = converted
    report["n_unit_unconverted"] = unconverted
    report["n_invalid_values"] = invalid
    measurable = frame["event_type"].isin(["lab", "vital"]) & frame["event_value"].isna()
    frame.loc[measurable, "missingness_indicator"] = True
    return frame, report
