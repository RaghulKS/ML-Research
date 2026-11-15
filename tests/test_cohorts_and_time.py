"""Cohort rules, adapters, clocks, and leakage safeguards."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from clinical_ml.cohorts.definitions import DEFINITIONS, code_matches, get_definition
from clinical_ml.cohorts.extraction import cohort_patient_ids, cohort_summary
from clinical_ml.cohorts.sql import render_crdw_sql, render_mimic_sql
from clinical_ml.data.adapters import CrdwAdapter, MimicAdapter
from clinical_ml.data.io import read_table
from clinical_ml.data.schemas import paired_codes, split_code_list
from clinical_ml.errors import DatasetLabelError, IncompatibleTaskError, LeakageError
from clinical_ml.features.tabular import build_feature_frame
from clinical_ml.preprocessing.clean import clean_events
from clinical_ml.preprocessing.harmonize import harmonize_events, load_harmonization
from clinical_ml.preprocessing.labels import build_readmission_labels
from clinical_ml.preprocessing.splits import assert_patient_disjoint, patient_disjoint_split
from clinical_ml.reporting import REAL, resolve_dataset_type

ROOT = Path(__file__).resolve().parents[1]


def test_historical_codes_match_the_supplied_lists():
    hf = get_definition("heart_failure", "historical")
    stroke = get_definition("stroke", "historical")
    mi = get_definition("myocardial_infarction", "historical")
    assert code_matches("I50.22", "ICD10", hf.rules)
    assert code_matches("428.21", "ICD9", hf.rules)
    assert code_matches("I63.9", "ICD10", stroke.rules)
    assert code_matches("434.91", "ICD9", stroke.rules)
    assert code_matches("I21.9", "ICD10", mi.rules)
    assert code_matches("410.91", "ICD9", mi.rules)
    assert not code_matches("I21.4", "ICD10", mi.rules)
    assert not code_matches("I21.90", "ICD10", mi.rules)
    assert not code_matches("410.71", "ICD9", mi.rules)
    assert not code_matches("I11.0", "ICD10", hf.rules)
    assert not code_matches("402.91", "ICD9", hf.rules)
    assert not code_matches("402.90", "ICD9", get_definition("heart_failure", "expanded").rules)
    assert not code_matches("I13.1", "ICD10", get_definition("heart_failure", "expanded").rules)
    assert code_matches("I11.0", "ICD10", get_definition("heart_failure", "expanded").rules)
    assert code_matches("I21.4", "ICD10", get_definition("myocardial_infarction", "expanded").rules)
    assert code_matches("I61.9", "ICD10", get_definition("stroke", "expanded").rules)
    assert not code_matches("I61.9", "ICD10", stroke.rules)


def test_declared_system_is_not_ignored():
    hf = get_definition("heart_failure", "historical").rules
    # A lettered code declared as ICD-9 must not satisfy an ICD-10 rule.
    assert not code_matches("I50.9", "ICD9", hf)


def test_comma_separated_codes_drop_empty_tokens():
    assert split_code_list("I50.9, , I10; E11.9") == ["I509", "I10", "E119"]
    pairs = paired_codes("I50.9, ***, I10")
    assert pairs == [("I50.9", "I509"), ("I10", "I10")]


def test_yaml_rules_match_the_registry():
    import yaml

    for condition in ("heart_failure", "stroke", "myocardial_infarction"):
        payload = yaml.safe_load((ROOT / "configs" / "cohorts" / f"{condition}.yaml").read_text())
        definition = get_definition(payload["condition"], payload["phenotype"])
        assert payload["version"] == definition.version
        parsed = [(row["system"], row["pattern"], row["match"]) for row in payload["rules"]]
        assert parsed == [(rule.system, rule.pattern, rule.match) for rule in definition.rules]


def test_sql_files_match_the_python_rules():
    for (condition, phenotype), definition in DEFINITIONS.items():
        mimic = (ROOT / "sql" / f"mimic_{condition}_{phenotype}.sql").read_text(encoding="utf-8")
        crdw = (ROOT / "sql" / f"crdw_{condition}_{phenotype}.sql").read_text(encoding="utf-8")
        assert mimic == render_mimic_sql(definition)
        assert crdw == render_crdw_sql(definition)
        assert "was not executed" in mimic
        assert "comma-separated" in crdw


def test_patient_dedup_and_encounter_grouping():
    events = _events(
        [
            ("p1", "e1", 0, "diagnosis", "I509", "ICD10"),
            ("p1", "e1", 1, "diagnosis", "I10", "ICD10"),
            ("p1", "e2", 10, "diagnosis", "I509", "ICD10"),
            ("p2", "e1", 0, "diagnosis", "I10", "ICD10"),
        ]
    )
    definition = get_definition("heart_failure", "historical")
    assert cohort_patient_ids(events, definition) == ["p1"]
    summary = cohort_summary(events, definition)
    assert summary["n_patients"] == 1
    assert summary["n_encounters"] == 2


def test_crdw_list_and_orders_are_not_administrations():
    diagnoses = pd.DataFrame(
        [
            {
                "PAT_MRN_ID_ENCRYPT": "c1",
                "PAT_ENC_CSN_ID_ENCRYPT": "v1",
                "CURRENT_ICD10_LIST": "I50.9, I10",
                "CURRENT_ICD9_LIST": "428.0",
                "DX_NAME": "Heart failure",
                "PATIENT_AGE": 70,
                "DAYS_SINCE_DX_FRST_OBSRVD": 3,
            }
        ]
    )
    medications = pd.DataFrame(
        [
            {
                "PAT_MRN_ID_ENCRYPT": "c1",
                "PAT_ENC_CSN_ID_ENCRYPT": "v1",
                "MED_NAME": "LASIX",
                "DAYS_SINCE_DX_FRST_OBSRVD": 3,
                "ORDER_STATUS": "ordered",
            }
        ]
    )
    data = CrdwAdapter().from_frames({"diagnoses": diagnoses, "medications": medications})
    codes = set(data.events.loc[data.events.event_type.eq("diagnosis"), "event_code"])
    assert codes == {"I509", "I10", "4280"}
    meds = data.events[data.events.event_type.eq("medication_order")]
    assert len(meds) == 1
    assert not bool(meds.iloc[0]["administration_confirmed"])
    assert data.events["time_origin"].iloc[0] == "relative_days"


def test_mimic_emar_is_administration_and_prescription_is_not():
    patients = pd.DataFrame([{"subject_id": "1", "gender": "F", "anchor_age": 70}])
    admissions = pd.DataFrame(
        [{"subject_id": "1", "hadm_id": "h1", "admittime": "2116-01-01", "dischtime": "2116-01-05", "admission_type": "EMERGENCY"}]
    )
    diagnoses = pd.DataFrame([{"subject_id": "1", "hadm_id": "h1", "icd_code": "I50.9", "icd_version": 10}])
    prescriptions = pd.DataFrame([{"subject_id": "1", "hadm_id": "h1", "starttime": "2116-01-01", "drug": "FUROSEMIDE"}])
    emar = pd.DataFrame([{"subject_id": "1", "hadm_id": "h1", "charttime": "2116-01-02", "medication": "FUROSEMIDE"}])
    data = MimicAdapter().from_frames(
        {"patients": patients, "admissions": admissions, "diagnoses": diagnoses, "prescriptions": prescriptions, "emar": emar}
    )
    orders = data.events[data.events.event_type.eq("medication_order")]
    admins = data.events[data.events.event_type.eq("medication_admin")]
    assert len(orders) == 1 and not bool(orders.iloc[0]["administration_confirmed"])
    assert len(admins) == 1 and bool(admins.iloc[0]["administration_confirmed"])
    assert data.events["time_origin"].iloc[0] == "calendar_days_since_unix_epoch"


def test_clean_sorts_and_drops_broken_rows():
    events = _events([("b", "e", 2, "diagnosis", "I10", "ICD10"), ("a", "e", 5, "diagnosis", "I10", "ICD10"), ("a", "e", 1, "diagnosis", "I10", "ICD10")])
    events = pd.concat([events, events.iloc[[0]]], ignore_index=True)
    blank = events.iloc[[0]].copy()
    blank["patient_id"] = ""
    events = pd.concat([events, blank], ignore_index=True)
    cleaned, report = clean_events(events)
    assert report["dropped_missing_patient_id"] == 1
    assert report["dropped_duplicate_rows"] == 1
    times = cleaned.loc[cleaned.patient_id.eq("a"), "event_time"].tolist()
    assert times == sorted(times)


def test_reader_handles_latin1_and_chunks(tmp_path: Path):
    path = tmp_path / "notes.csv"
    path.write_bytes("patient_id,note\n1,d\xe9compensation\n".encode("latin-1"))
    frame = read_table(path, chunksize=1)
    assert "décompensation" in frame.iloc[0]["note"] or "d" in frame.iloc[0]["note"]
    assert len(frame) == 1


def test_unit_conversion_is_idempotent_and_rejects_impossible_values():
    spec = load_harmonization(ROOT / "configs/harmonization/default.yaml")
    events = _events([("p", "e", 0, "lab", "CREAT", "")])
    events.loc[0, "event_type"] = "lab"
    events.loc[0, "event_code"] = "CREAT"
    events.loc[0, "event_value"] = 88.4
    events.loc[0, "event_unit"] = "umol/L"
    bad = events.iloc[[0]].copy()
    bad["event_value"] = 500
    bad["event_unit"] = "mg/dL"
    bad["event_time"] = 1
    weird = events.iloc[[0]].copy()
    weird["event_code"] = "SODIUM"
    weird["event_value"] = 140
    weird["event_unit"] = "furlongs"
    weird["event_time"] = 2
    frame = pd.concat([events, bad, weird], ignore_index=True)
    converted, report = harmonize_events(frame, spec)
    again, second = harmonize_events(converted, spec)
    creat = converted.loc[converted.event_code.eq("CREAT") & converted.event_time.eq(0), "event_value"].iloc[0]
    assert creat == pytest.approx(1.0, abs=1e-6)
    assert report["n_invalid_values"] == 1
    assert report["n_unit_unconverted"] == 1
    assert second["n_units_converted"] == 0
    assert pd.isna(again.loc[again.event_time.eq(1), "event_value"]).all()


def test_features_ignore_events_after_the_anchor():
    events = _events(
        [
            ("p", "e1", 0, "diagnosis", "I509", "ICD10"),
            ("p", "e1", 1, "lab", "CREAT", ""),
            ("p", "e2", 20, "lab", "CREAT", ""),
        ]
    )
    events.loc[events.event_time.eq(1), "event_value"] = 1.2
    events.loc[events.event_time.eq(20), "event_value"] = 3.3
    events["event_unit"] = "mg/dL"
    patients = pd.DataFrame([{"patient_id": "p", "age": 70, "sex": "F", "followup_end_time": 40, "time_origin": "relative_days", "source_dataset": "test"}])
    encounters = pd.DataFrame(
        [
            {"patient_id": "p", "encounter_id": "e1", "start_time": 0, "end_time": 2, "time_origin": "relative_days", "source_dataset": "test", "admission_type": "EMERGENCY", "end_source": "test"},
            {"patient_id": "p", "encounter_id": "e2", "start_time": 20, "end_time": 21, "time_origin": "relative_days", "source_dataset": "test", "admission_type": "", "end_source": "test"},
        ]
    )
    labeled = pd.DataFrame(
        [{"patient_id": "p", "index_encounter_id": "e1", "index_start": 0.0, "anchor_time": 2.0, "time_origin": "relative_days", "label": 1}]
    )
    features = build_feature_frame(events, patients, encounters, labeled)
    assert features.loc[0, "creat_last"] == pytest.approx(1.2)
    assert features.loc[0, "audit_max_history_time"] <= 2
    assert features.loc[0, "audit_n_events_after_anchor"] == 1


def test_readmission_label_censors_short_followup_and_is_not_the_hf_code():
    events = _events(
        [
            ("p1", "e1", 0, "diagnosis", "I509", "ICD10"),
            ("p1", "e2", 10, "diagnosis", "J189", "ICD10"),
            ("p2", "e1", 0, "diagnosis", "I509", "ICD10"),
            ("p3", "e1", 0, "diagnosis", "I509", "ICD10"),
            ("p3", "e2", 40, "diagnosis", "Z09", "ICD10"),
        ]
    )
    patients = pd.DataFrame(
        [
            {"patient_id": "p1", "age": 70, "sex": "M", "followup_end_time": 10, "time_origin": "relative_days", "source_dataset": "t"},
            {"patient_id": "p2", "age": 70, "sex": "M", "followup_end_time": 8, "time_origin": "relative_days", "source_dataset": "t"},
            {"patient_id": "p3", "age": 70, "sex": "F", "followup_end_time": 40, "time_origin": "relative_days", "source_dataset": "t"},
        ]
    )
    encounters = pd.DataFrame(
        [
            {"patient_id": "p1", "encounter_id": "e1", "start_time": 0, "end_time": 1, "time_origin": "relative_days", "source_dataset": "t", "admission_type": "", "end_source": "t"},
            {"patient_id": "p1", "encounter_id": "e2", "start_time": 10, "end_time": 12, "time_origin": "relative_days", "source_dataset": "t", "admission_type": "", "end_source": "t"},
            {"patient_id": "p2", "encounter_id": "e1", "start_time": 0, "end_time": 1, "time_origin": "relative_days", "source_dataset": "t", "admission_type": "", "end_source": "t"},
            {"patient_id": "p3", "encounter_id": "e1", "start_time": 0, "end_time": 1, "time_origin": "relative_days", "source_dataset": "t", "admission_type": "", "end_source": "t"},
            {"patient_id": "p3", "encounter_id": "e2", "start_time": 40, "end_time": 41, "time_origin": "relative_days", "source_dataset": "t", "admission_type": "", "end_source": "t"},
        ]
    )
    definition = get_definition("heart_failure", "historical")
    result = build_readmission_labels(events, patients, encounters, ["p1", "p2", "p3"], definition, 30, "hf_30day_all_cause_readmission")
    labels = result["labeled"].set_index("patient_id")["label"].to_dict()
    assert labels["p1"] == 1
    assert labels["p3"] == 0
    assert "p2" not in labels
    assert result["summary"]["n_censored_insufficient_followup"] == 1


def test_splits_are_patient_disjoint_and_synthetic_cannot_be_called_real():
    labeled = pd.DataFrame({"patient_id": [f"p{i}" for i in range(20)], "label": [i % 2 for i in range(20)]})
    splits, _info = patient_disjoint_split(labeled, seed=3)
    assert_patient_disjoint(splits)
    assert set().union(*[set(v) for v in splits.values()]) == set(labeled.patient_id)
    with pytest.raises(LeakageError):
        assert_patient_disjoint({"train": ["a"], "test": ["a"]})
    with pytest.raises(DatasetLabelError):
        resolve_dataset_type(REAL, {"dataset_type": "SYNTHETIC ENGINEERING VALIDATION", "not_clinical_data": True})


def test_incompatible_tasks_are_rejected():
    from clinical_ml.experiments import assert_same_task

    left = {"summary": {"task": "a", "implementation": "readmission_v1", "cohort_version": "v", "horizon_days": 30}}
    right = {"summary": {"task": "a", "implementation": "readmission_v1", "cohort_version": "v", "horizon_days": 90}}
    with pytest.raises(IncompatibleTaskError):
        assert_same_task(left, right)


def _events(rows):
    records = []
    for pid, eid, time, event_type, code, system in rows:
        records.append(
            {
                "patient_id": pid,
                "encounter_id": eid,
                "event_time": time,
                "time_origin": "relative_days",
                "event_type": event_type,
                "event_code": code,
                "raw_code": code,
                "event_value": float("nan"),
                "event_unit": "",
                "source_dataset": "test",
                "event_description": code,
                "treatment_indicator": event_type in {"medication_order", "medication_admin", "procedure"},
                "administration_confirmed": event_type == "medication_admin",
                "provenance": "test",
                "missingness_indicator": False,
                "coding_system": system,
            }
        )
    return pd.DataFrame(records)
