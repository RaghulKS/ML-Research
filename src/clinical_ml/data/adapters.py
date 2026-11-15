"""Source adapters for MIMIC-IV-style tables and CRDW-style extracts.

Each file is optional. A directory with only diagnoses can still build a
cohort. Medication rows from order tables are medication orders, not
administrations, unless the source is an administration table or an explicit
administration flag.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from clinical_ml.data.io import discover_file, pick_column, read_table
from clinical_ml.data.schemas import (
    ENCOUNTER_COLUMNS,
    EVENT_COLUMNS,
    PATIENT_COLUMNS,
    normalize_icd,
    normalize_identifier_series,
    normalize_unit,
    parse_icd_system,
    paired_codes,
)
from clinical_ml.preprocessing.clean import clean_events

UNIX = pd.Timestamp("1970-01-01", tz="UTC")
ADMIN_TOKENS = {
    "administered",
    "administration",
    "given",
    "completed_admin",
    "admin",
    "yes",
    "true",
    "y",
}


@dataclass
class DomainData:
    events: pd.DataFrame
    patients: pd.DataFrame
    encounters: pd.DataFrame
    source: str
    files_loaded: list[str] = field(default_factory=list)
    cleaning: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _parse_time(series: pd.Series) -> tuple[pd.Series, str]:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce"), "relative_days"
    as_text = series.astype("object").where(series.notna(), "")
    nonempty = as_text.map(lambda v: str(v).strip() not in {"", "nan", "None", "NaT", "null"})
    numeric = pd.to_numeric(series, errors="coerce")
    if nonempty.any() and int(numeric.notna().sum()) >= 0.8 * int(nonempty.sum()):
        return numeric, "relative_days"
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if nonempty.any() and int(parsed.notna().sum()) >= 0.5 * int(nonempty.sum()):
        days = (parsed - UNIX).dt.total_seconds() / 86400.0
        return days, "calendar_days_since_unix_epoch"
    return numeric, "relative_days"


def _col(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    name = pick_column(frame, names, required=True)
    return frame[name]


def _optional(frame: pd.DataFrame, names: list[str]) -> pd.Series | None:
    name = pick_column(frame, names, required=False)
    if name is None:
        return None
    return frame[name]


def _event_dict(**kwargs) -> dict:
    base = {column: "" for column in EVENT_COLUMNS}
    base["event_value"] = float("nan")
    base["treatment_indicator"] = False
    base["administration_confirmed"] = False
    base["missingness_indicator"] = False
    base.update(kwargs)
    return base


def _is_admin_flag(value) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in ADMIN_TOKENS


class MimicAdapter:
    """Read a MIMIC-IV-like directory or a dict of already loaded tables."""

    FILE_STEMS = {
        "patients": ["patients"],
        "admissions": ["admissions"],
        "diagnoses": ["diagnoses_icd", "diagnoses"],
        "prescriptions": ["prescriptions"],
        "emar": ["emar", "emar_detail"],
        "labs": ["labevents", "labs"],
        "vitals": ["chartevents", "vitals"],
        "procedures": ["procedures_icd", "procedures"],
    }

    def from_directory(self, path: Path | str, source_dataset: str = "mimic") -> DomainData:
        directory = Path(path)
        frames = {}
        loaded = []
        for key, stems in self.FILE_STEMS.items():
            found = discover_file(directory, stems)
            if found is not None:
                frames[key] = read_table(found)
                loaded.append(found.name)
        data = self.from_frames(frames, source_dataset=source_dataset)
        data.files_loaded = loaded
        data.provenance = _read_provenance(directory)
        return data

    def from_frames(self, frames: dict[str, pd.DataFrame], source_dataset: str = "mimic") -> DomainData:
        warnings: list[str] = []
        if not frames:
            raise ValueError("MIMIC adapter received no tables.")
        admissions = frames.get("admissions")
        admit_index = _admission_index(admissions)
        events: list[dict] = []
        if "diagnoses" in frames:
            events.extend(_mimic_diagnoses(frames["diagnoses"], admit_index, source_dataset, warnings))
        if "prescriptions" in frames:
            events.extend(_mimic_meds(frames["prescriptions"], source_dataset, administration=False))
        if "emar" in frames:
            events.extend(_mimic_meds(frames["emar"], source_dataset, administration=True))
        if "labs" in frames:
            events.extend(_mimic_measurements(frames["labs"], source_dataset, event_type="lab"))
        if "vitals" in frames:
            events.extend(_mimic_measurements(frames["vitals"], source_dataset, event_type="vital"))
        if "procedures" in frames:
            events.extend(_mimic_procedures(frames["procedures"], admit_index, source_dataset))
        event_frame = pd.DataFrame(events) if events else pd.DataFrame(columns=EVENT_COLUMNS)
        for column in EVENT_COLUMNS:
            if column not in event_frame.columns:
                event_frame[column] = pd.NA
        cleaned, cleaning = clean_events(event_frame[EVENT_COLUMNS])
        patients = _mimic_patients(frames.get("patients"), cleaned, source_dataset, warnings)
        encounters = _encounters_from_admissions(admit_index, cleaned, source_dataset, warnings)
        return DomainData(
            events=cleaned,
            patients=patients,
            encounters=encounters,
            source=source_dataset,
            cleaning=cleaning,
            warnings=warnings,
        )


class CrdwAdapter:
    """Read a CRDW-like directory. Diagnosis lists may contain many codes."""

    FILE_STEMS = {
        "diagnoses": ["diagnoses", "comorbidities", "hypertension_with_comorbidities"],
        "medications": ["medications", "meds"],
        "labs": ["labs"],
        "vitals": ["vitals"],
        "procedures": ["procedures"],
        "patients": ["patients"],
    }

    def from_directory(self, path: Path | str, source_dataset: str = "crdw") -> DomainData:
        directory = Path(path)
        frames = {}
        loaded = []
        for key, stems in self.FILE_STEMS.items():
            found = discover_file(directory, stems)
            if found is not None:
                frames[key] = read_table(found)
                loaded.append(found.name)
        data = self.from_frames(frames, source_dataset=source_dataset)
        data.files_loaded = loaded
        data.provenance = _read_provenance(directory)
        return data

    def from_frames(self, frames: dict[str, pd.DataFrame], source_dataset: str = "crdw") -> DomainData:
        warnings: list[str] = []
        if not frames:
            raise ValueError("CRDW adapter received no tables.")
        events: list[dict] = []
        ages: list[pd.DataFrame] = []
        if "diagnoses" in frames:
            dx_events, age_rows = _crdw_diagnoses(frames["diagnoses"], source_dataset)
            events.extend(dx_events)
            ages.append(age_rows)
        if "medications" in frames:
            events.extend(_crdw_meds(frames["medications"], source_dataset))
        if "labs" in frames:
            events.extend(_crdw_measurements(frames["labs"], source_dataset, "lab", ["LAB_CODE", "lab_code", "itemid"], ["LAB_VALUE", "lab_value", "valuenum"], ["LAB_UNIT", "lab_unit", "valueuom"]))
        if "vitals" in frames:
            events.extend(_crdw_measurements(frames["vitals"], source_dataset, "vital", ["VITAL_CODE", "vital_code", "itemid"], ["VITAL_VALUE", "vital_value", "valuenum"], ["VITAL_UNIT", "vital_unit", "valueuom"]))
        if "procedures" in frames:
            events.extend(_crdw_procedures(frames["procedures"], source_dataset))
        event_frame = pd.DataFrame(events) if events else pd.DataFrame(columns=EVENT_COLUMNS)
        for column in EVENT_COLUMNS:
            if column not in event_frame.columns:
                event_frame[column] = pd.NA
        cleaned, cleaning = clean_events(event_frame[EVENT_COLUMNS])
        origin = cleaned["time_origin"].mode().iloc[0] if len(cleaned) else "relative_days"
        patients = _crdw_patients(frames.get("patients"), ages, cleaned, source_dataset, origin, warnings)
        encounters = _encounters_from_events(cleaned, source_dataset, "last_event")
        return DomainData(
            events=cleaned,
            patients=patients,
            encounters=encounters,
            source=source_dataset,
            cleaning=cleaning,
            warnings=warnings,
        )


def load_source(path: Path | str, source: str) -> DomainData:
    source = source.lower()
    if source in {"mimic", "mimic-iv", "mimiciv"}:
        return MimicAdapter().from_directory(path, source_dataset="mimic")
    if source in {"crdw", "rutgers", "rwjbh"}:
        return CrdwAdapter().from_directory(path, source_dataset="crdw")
    raise ValueError(f"Unknown source '{source}'. Use 'mimic' or 'crdw'.")


def _read_provenance(directory: Path) -> dict:
    path = directory / "PROVENANCE.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _admission_index(admissions: pd.DataFrame | None) -> pd.DataFrame:
    if admissions is None or admissions.empty:
        return pd.DataFrame(columns=["patient_id", "encounter_id", "start_time", "end_time", "time_origin", "admission_type"])
    patient = normalize_identifier_series(_col(admissions, ["subject_id", "patient_id"]))
    encounter = normalize_identifier_series(_col(admissions, ["hadm_id", "encounter_id"]))
    start, start_origin = _parse_time(_col(admissions, ["admittime", "start_time"]))
    end, end_origin = _parse_time(_col(admissions, ["dischtime", "end_time", "disch_time"]))
    origin = start_origin if start.notna().sum() >= end.notna().sum() else end_origin
    if start_origin != end_origin:
        # Keep the start clock and drop incompatible ends rather than subtract them.
        end = pd.Series(float("nan"), index=admissions.index)
    admission_type = _optional(admissions, ["admission_type", "admissiontype"])
    frame = pd.DataFrame(
        {
            "patient_id": patient,
            "encounter_id": encounter,
            "start_time": start,
            "end_time": end,
            "time_origin": origin,
            "admission_type": admission_type.astype(str) if admission_type is not None else "",
        }
    )
    return frame


def _mimic_diagnoses(frame, admit_index, source, warnings) -> list[dict]:
    patient = normalize_identifier_series(_col(frame, ["subject_id", "patient_id"]))
    encounter = normalize_identifier_series(_col(frame, ["hadm_id", "encounter_id"]))
    raw_codes = _col(frame, ["icd_code", "code"])
    version = _optional(frame, ["icd_version", "icd_system", "coding_system"])
    lookup = admit_index.set_index(["patient_id", "encounter_id"]) if len(admit_index) else None
    rows = []
    missing_time = 0
    for idx in frame.index:
        declared = None if version is None else version.at[idx]
        key = (patient.at[idx], encounter.at[idx])
        if lookup is not None and key in lookup.index:
            info = lookup.loc[key]
            if isinstance(info, pd.DataFrame):
                info = info.iloc[0]
            event_time = info["start_time"]
            origin = info["time_origin"]
        else:
            event_time = float("nan")
            origin = "calendar_days_since_unix_epoch"
            missing_time += 1
        for raw, norm in paired_codes(raw_codes.at[idx]):
            system = parse_icd_system(declared, norm)
            rows.append(
                _event_dict(
                    patient_id=patient.at[idx],
                    encounter_id=encounter.at[idx],
                    event_time=event_time,
                    time_origin=origin,
                    event_type="diagnosis",
                    event_code=norm,
                    raw_code=raw,
                    source_dataset=source,
                    event_description=raw,
                    treatment_indicator=False,
                    administration_confirmed=False,
                    provenance="mimic:diagnoses_icd",
                    missingness_indicator=False,
                    coding_system=system,
                )
            )
    if missing_time:
        warnings.append(f"{missing_time} diagnosis rows had no matching admission time and will be dropped.")
    return rows


def _mimic_meds(frame, source, administration: bool) -> list[dict]:
    patient = normalize_identifier_series(_col(frame, ["subject_id", "patient_id"]))
    encounter = normalize_identifier_series(_col(frame, ["hadm_id", "encounter_id"]))
    times, origin = _parse_time(_col(frame, ["starttime", "charttime", "event_time"]))
    drug = _col(frame, ["drug", "medication", "med_name"])
    rows = []
    event_type = "medication_admin" if administration else "medication_order"
    provenance = "mimic:emar" if administration else "mimic:prescriptions"
    for idx in frame.index:
        name = "" if pd.isna(drug.at[idx]) else str(drug.at[idx]).strip()
        rows.append(
            _event_dict(
                patient_id=patient.at[idx],
                encounter_id=encounter.at[idx],
                event_time=times.at[idx],
                time_origin=origin,
                event_type=event_type,
                event_code=name.upper(),
                raw_code=name,
                source_dataset=source,
                event_description=name,
                treatment_indicator=True,
                administration_confirmed=administration,
                provenance=provenance,
                coding_system="",
            )
        )
    return rows


def _mimic_measurements(frame, source, event_type: str) -> list[dict]:
    patient = normalize_identifier_series(_col(frame, ["subject_id", "patient_id"]))
    enc_series = _optional(frame, ["hadm_id", "encounter_id"])
    if enc_series is None:
        encounter = pd.Series("", index=frame.index)
    else:
        encounter = normalize_identifier_series(enc_series)
    times, origin = _parse_time(_col(frame, ["charttime", "chartdate", "event_time"]))
    item = _col(frame, ["itemid", "item_id", "code"])
    value = pd.to_numeric(_col(frame, ["valuenum", "value", "event_value"]), errors="coerce")
    unit = _optional(frame, ["valueuom", "unit", "event_unit"])
    rows = []
    provenance = f"mimic:{'labevents' if event_type == 'lab' else 'chartevents'}"
    for idx in frame.index:
        code = "" if pd.isna(item.at[idx]) else str(item.at[idx]).strip()
        unit_text = "" if unit is None or pd.isna(unit.at[idx]) else normalize_unit(unit.at[idx])
        number = value.at[idx]
        rows.append(
            _event_dict(
                patient_id=patient.at[idx],
                encounter_id=encounter.at[idx],
                event_time=times.at[idx],
                time_origin=origin,
                event_type=event_type,
                event_code=code.upper(),
                raw_code=code,
                event_value=number,
                event_unit=unit_text,
                source_dataset=source,
                event_description=code,
                treatment_indicator=False,
                administration_confirmed=False,
                provenance=provenance,
                missingness_indicator=bool(pd.isna(number)),
                coding_system="",
            )
        )
    return rows


def _mimic_procedures(frame, admit_index, source) -> list[dict]:
    patient = normalize_identifier_series(_col(frame, ["subject_id", "patient_id"]))
    encounter = normalize_identifier_series(_col(frame, ["hadm_id", "encounter_id"]))
    if pick_column(frame, ["chartdate", "charttime", "event_time"], required=False):
        times, origin = _parse_time(_col(frame, ["chartdate", "charttime", "event_time"]))
    else:
        times = pd.Series(float("nan"), index=frame.index)
        origin = "calendar_days_since_unix_epoch"
    code = _col(frame, ["icd_code", "code", "proc_code"])
    version = _optional(frame, ["icd_version", "coding_system"])
    lookup = admit_index.set_index(["patient_id", "encounter_id"]) if len(admit_index) else None
    rows = []
    for idx in frame.index:
        raw = "" if pd.isna(code.at[idx]) else str(code.at[idx]).strip()
        event_time = times.at[idx]
        time_origin = origin
        if pd.isna(event_time) and lookup is not None:
            key = (patient.at[idx], encounter.at[idx])
            if key in lookup.index:
                info = lookup.loc[key]
                if isinstance(info, pd.DataFrame):
                    info = info.iloc[0]
                event_time = info["start_time"]
                time_origin = info["time_origin"]
        system = parse_icd_system(None if version is None else version.at[idx], normalize_icd(raw))
        rows.append(
            _event_dict(
                patient_id=patient.at[idx],
                encounter_id=encounter.at[idx],
                event_time=event_time,
                time_origin=time_origin,
                event_type="procedure",
                event_code=raw.upper(),
                raw_code=raw,
                source_dataset=source,
                event_description=raw,
                treatment_indicator=True,
                administration_confirmed=False,
                provenance="mimic:procedures_icd",
                coding_system=system,
            )
        )
    return rows


def _mimic_patients(patients, events, source, warnings) -> pd.DataFrame:
    if patients is None or patients.empty:
        return _patients_from_events(events, source)
    pid = normalize_identifier_series(_col(patients, ["subject_id", "patient_id"]))
    age = pd.to_numeric(_col(patients, ["anchor_age", "age", "patient_age"]), errors="coerce") if pick_column(patients, ["anchor_age", "age", "patient_age"], required=False) else pd.Series(float("nan"), index=patients.index)
    sex_col = _optional(patients, ["gender", "sex"])
    sex = sex_col.astype(str) if sex_col is not None else pd.Series("", index=patients.index)
    follow = _optional(patients, ["followup_end", "followup_end_time", "dod"])
    if follow is not None:
        follow_vals, follow_origin = _parse_time(follow)
    else:
        follow_vals, follow_origin = pd.Series(float("nan"), index=patients.index), ""
    event_origin = events["time_origin"].mode().iloc[0] if len(events) else follow_origin
    if follow_origin and event_origin and follow_origin != event_origin:
        warnings.append(
            "Patient follow-up clock does not match event time_origin. Follow-up end was discarded rather than compared across clocks."
        )
        follow_vals = pd.Series(float("nan"), index=patients.index)
        follow_origin = event_origin
    frame = pd.DataFrame(
        {
            "patient_id": pid,
            "age": age,
            "sex": sex,
            "followup_end_time": follow_vals,
            "time_origin": event_origin or follow_origin or "calendar_days_since_unix_epoch",
            "source_dataset": source,
        }
    )
    frame = frame[frame["patient_id"].astype(str).str.strip().ne("")].drop_duplicates("patient_id", keep="first")
    return frame.reset_index(drop=True)


def _crdw_diagnoses(frame, source) -> tuple[list[dict], pd.DataFrame]:
    patient = normalize_identifier_series(_col(frame, ["PAT_MRN_ID_ENCRYPT", "patient_id", "mrn"]))
    encounter = normalize_identifier_series(_col(frame, ["PAT_ENC_CSN_ID_ENCRYPT", "encounter_id"]))
    times, origin = _parse_time(_col(frame, ["DAYS_SINCE_DX_FRST_OBSRVD", "days_since_first_dx", "event_time"]))
    icd10 = _optional(frame, ["CURRENT_ICD10_LIST", "icd10", "icd10_code"])
    icd9 = _optional(frame, ["CURRENT_ICD9_LIST", "icd9", "icd9_code"])
    name = _optional(frame, ["DX_NAME", "diagnosis", "description"])
    age = _optional(frame, ["PATIENT_AGE", "age"])
    rows = []
    age_rows = []
    for idx in frame.index:
        description = "" if name is None or pd.isna(name.at[idx]) else str(name.at[idx]).strip()
        bundles = []
        if icd10 is not None:
            bundles.append(("ICD10", icd10.at[idx], "crdw:CURRENT_ICD10_LIST"))
        if icd9 is not None:
            bundles.append(("ICD9", icd9.at[idx], "crdw:CURRENT_ICD9_LIST"))
        for system, raw_field, provenance in bundles:
            for raw, norm in paired_codes(raw_field):
                # Trust the source column over the character heuristic when both exist.
                rows.append(
                    _event_dict(
                        patient_id=patient.at[idx],
                        encounter_id=encounter.at[idx],
                        event_time=times.at[idx],
                        time_origin=origin,
                        event_type="diagnosis",
                        event_code=norm,
                        raw_code=raw,
                        source_dataset=source,
                        event_description=description or raw,
                        treatment_indicator=False,
                        administration_confirmed=False,
                        provenance=provenance,
                        coding_system=system,
                    )
                )
        if age is not None and str(patient.at[idx]).strip():
            age_rows.append({"patient_id": patient.at[idx], "age": pd.to_numeric(age.at[idx], errors="coerce")})
    return rows, pd.DataFrame(age_rows)


def _crdw_meds(frame, source) -> list[dict]:
    patient = normalize_identifier_series(_col(frame, ["PAT_MRN_ID_ENCRYPT", "patient_id"]))
    encounter = normalize_identifier_series(_col(frame, ["PAT_ENC_CSN_ID_ENCRYPT", "encounter_id"]))
    times, origin = _parse_time(_col(frame, ["DAYS_SINCE_DX_FRST_OBSRVD", "order_day", "event_time"]))
    name = _col(frame, ["MED_NAME", "medication_name", "drug"])
    code = _optional(frame, ["MED_CODE", "medication_code"])
    status = _optional(frame, ["ADMIN_FLAG", "ADMINISTERED", "administration_flag"])
    # ORDER_STATUS alone does not confirm administration. Only an explicit admin flag does.
    rows = []
    for idx in frame.index:
        raw_name = "" if pd.isna(name.at[idx]) else str(name.at[idx]).strip()
        raw_code = raw_name if code is None or pd.isna(code.at[idx]) else str(code.at[idx]).strip()
        confirmed = _is_admin_flag(None if status is None else status.at[idx])
        rows.append(
            _event_dict(
                patient_id=patient.at[idx],
                encounter_id=encounter.at[idx],
                event_time=times.at[idx],
                time_origin=origin,
                event_type="medication_admin" if confirmed else "medication_order",
                event_code=(raw_code or raw_name).upper(),
                raw_code=raw_name or raw_code,
                source_dataset=source,
                event_description=raw_name,
                treatment_indicator=True,
                administration_confirmed=confirmed,
                provenance="crdw:medications_admin" if confirmed else "crdw:medications_order",
                coding_system="",
            )
        )
    return rows


def _crdw_measurements(frame, source, event_type, code_names, value_names, unit_names) -> list[dict]:
    patient = normalize_identifier_series(_col(frame, ["PAT_MRN_ID_ENCRYPT", "patient_id"]))
    encounter = normalize_identifier_series(_col(frame, ["PAT_ENC_CSN_ID_ENCRYPT", "encounter_id"]))
    times, origin = _parse_time(_col(frame, ["DAYS_SINCE_DX_FRST_OBSRVD", "result_day", "event_time"]))
    code = _col(frame, code_names)
    value = pd.to_numeric(_col(frame, value_names), errors="coerce")
    unit = _optional(frame, unit_names)
    rows = []
    for idx in frame.index:
        raw = "" if pd.isna(code.at[idx]) else str(code.at[idx]).strip()
        unit_text = "" if unit is None or pd.isna(unit.at[idx]) else normalize_unit(unit.at[idx])
        number = value.at[idx]
        rows.append(
            _event_dict(
                patient_id=patient.at[idx],
                encounter_id=encounter.at[idx],
                event_time=times.at[idx],
                time_origin=origin,
                event_type=event_type,
                event_code=raw.upper(),
                raw_code=raw,
                event_value=number,
                event_unit=unit_text,
                source_dataset=source,
                event_description=raw,
                treatment_indicator=False,
                provenance=f"crdw:{event_type}",
                missingness_indicator=bool(pd.isna(number)),
                coding_system="",
            )
        )
    return rows


def _crdw_procedures(frame, source) -> list[dict]:
    patient = normalize_identifier_series(_col(frame, ["PAT_MRN_ID_ENCRYPT", "patient_id"]))
    encounter = normalize_identifier_series(_col(frame, ["PAT_ENC_CSN_ID_ENCRYPT", "encounter_id"]))
    times, origin = _parse_time(_col(frame, ["DAYS_SINCE_DX_FRST_OBSRVD", "proc_day", "event_time"]))
    code = _col(frame, ["PROC_CODE", "procedure_code", "code"])
    rows = []
    for idx in frame.index:
        raw = "" if pd.isna(code.at[idx]) else str(code.at[idx]).strip()
        rows.append(
            _event_dict(
                patient_id=patient.at[idx],
                encounter_id=encounter.at[idx],
                event_time=times.at[idx],
                time_origin=origin,
                event_type="procedure",
                event_code=raw.upper(),
                raw_code=raw,
                source_dataset=source,
                event_description=raw,
                treatment_indicator=True,
                administration_confirmed=False,
                provenance="crdw:procedures",
                coding_system="",
            )
        )
    return rows


def _crdw_patients(patients, age_frames, events, source, origin, warnings) -> pd.DataFrame:
    ages = pd.concat(age_frames, ignore_index=True) if age_frames else pd.DataFrame(columns=["patient_id", "age"])
    if len(ages):
        ages = ages.groupby("patient_id", as_index=False)["age"].median()
    if patients is not None and len(patients):
        pid = normalize_identifier_series(_col(patients, ["PAT_MRN_ID_ENCRYPT", "patient_id"]))
        age = pd.to_numeric(_optional(patients, ["PATIENT_AGE", "age"]), errors="coerce") if pick_column(patients, ["PATIENT_AGE", "age"], False) else pd.Series(float("nan"), index=patients.index)
        sex_col = _optional(patients, ["SEX", "gender", "sex"])
        sex = sex_col.astype(str) if sex_col is not None else pd.Series("", index=patients.index)
        follow = _optional(patients, ["FOLLOWUP_END_DAYS", "followup_end", "followup_end_time"])
        if follow is not None:
            follow_vals, follow_origin = _parse_time(follow)
            if follow_origin != origin and follow_vals.notna().any():
                warnings.append("CRDW follow-up clock differs from diagnosis-day clock. Follow-up end was discarded.")
                follow_vals = pd.Series(float("nan"), index=patients.index)
        else:
            follow_vals = pd.Series(float("nan"), index=patients.index)
        frame = pd.DataFrame(
            {
                "patient_id": pid,
                "age": age,
                "sex": sex,
                "followup_end_time": follow_vals,
                "time_origin": origin,
                "source_dataset": source,
            }
        )
    else:
        frame = _patients_from_events(events, source)
    if len(ages):
        frame = frame.merge(ages.rename(columns={"age": "age_from_dx"}), on="patient_id", how="outer")
        frame["age"] = frame["age"].fillna(frame["age_from_dx"]) if "age" in frame else frame["age_from_dx"]
        if "age_from_dx" in frame:
            frame = frame.drop(columns=["age_from_dx"])
    for column in PATIENT_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame["source_dataset"] = frame["source_dataset"].replace("", source).fillna(source)
    frame["time_origin"] = frame["time_origin"].replace("", origin).fillna(origin)
    frame = frame[frame["patient_id"].astype(str).str.strip().ne("")].drop_duplicates("patient_id", keep="first")
    return frame[PATIENT_COLUMNS].reset_index(drop=True)


def _patients_from_events(events: pd.DataFrame, source: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=PATIENT_COLUMNS)
    rows = []
    for pid, group in events.groupby("patient_id", sort=False):
        rows.append(
            {
                "patient_id": pid,
                "age": float("nan"),
                "sex": "",
                "followup_end_time": float("nan"),
                "time_origin": group["time_origin"].iloc[0],
                "source_dataset": source,
            }
        )
    return pd.DataFrame(rows)


def _encounters_from_admissions(admit_index, events, source, warnings) -> pd.DataFrame:
    from_events = _encounters_from_events(events, source, "last_event")
    if admit_index.empty:
        return from_events
    table = admit_index.copy()
    table["source_dataset"] = source
    table["end_source"] = "admission_table"
    bad = table["end_time"].notna() & table["start_time"].notna() & (table["end_time"] < table["start_time"])
    if bad.any():
        warnings.append(f"{int(bad.sum())} admissions had discharge before admission. End time was set to the start.")
        table.loc[bad, "end_time"] = table.loc[bad, "start_time"]
    table["end_time"] = table["end_time"].fillna(table["start_time"])
    if len(from_events):
        key_known = set(zip(table["patient_id"].astype(str), table["encounter_id"].astype(str)))
        mask = [
            (str(pid), str(enc)) not in key_known
            for pid, enc in zip(from_events["patient_id"], from_events["encounter_id"])
        ]
        extra = from_events.loc[mask]
    else:
        extra = from_events
    combined = pd.concat([table, extra], ignore_index=True)
    for column in ENCOUNTER_COLUMNS:
        if column not in combined.columns:
            combined[column] = ""
    return combined[ENCOUNTER_COLUMNS].reset_index(drop=True)


def _encounters_from_events(events: pd.DataFrame, source: str, end_source: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=ENCOUNTER_COLUMNS)
    grouped = events.groupby(["patient_id", "encounter_id"], sort=False)
    rows = []
    for (pid, eid), group in grouped:
        rows.append(
            {
                "patient_id": pid,
                "encounter_id": eid,
                "start_time": float(group["event_time"].min()),
                "end_time": float(group["event_time"].max()),
                "time_origin": group["time_origin"].iloc[0],
                "source_dataset": source,
                "admission_type": "",
                "end_source": end_source,
            }
        )
    return pd.DataFrame(rows)
