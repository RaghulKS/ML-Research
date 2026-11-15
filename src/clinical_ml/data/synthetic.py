"""Deterministic synthetic hospitals for engineering tests.

The two domains share a task definition and a latent risk process, but they do
not share clocks, code strings, or laboratory units. They are not clinical data
and must not be reported as clinical performance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_ml.reporting import SYNTHETIC

GENERATOR_VERSION = "synthetic_hospitals_v1"


def generate_domain(domain: str, n_patients: int, seed: int) -> dict[str, pd.DataFrame]:
    if domain not in {"mimic", "crdw"}:
        raise ValueError("domain must be 'mimic' or 'crdw'")
    if n_patients < 8:
        raise ValueError("n_patients must be at least 8 so both labels can be enforced")
    offset = 0 if domain == "mimic" else 10007
    rng = np.random.default_rng(seed + offset)
    stories = [_story(domain, i, rng) for i in range(n_patients)]
    _enforce_outcome_variety(stories)
    tables = _mimic_tables(stories) if domain == "mimic" else _crdw_tables(stories)
    tables["__meta__"] = {
        "dataset_type": SYNTHETIC,
        "generator": "clinical_ml.data.synthetic",
        "generator_version": GENERATOR_VERSION,
        "domain": domain,
        "n_patients": n_patients,
        "seed": seed,
        "not_clinical_data": True,
        "note": (
            "Latent risk influences creatinine, comorbidity burden, diuretic orders, "
            "and the readmission label. Discrimination on this extract is an engineering "
            "check, not a clinical result."
        ),
    }
    return tables


def write_domain(tables: dict, directory: Path | str) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    meta = tables["__meta__"]
    for name, frame in tables.items():
        if name.startswith("__"):
            continue
        frame.to_csv(directory / f"{name}.csv", index=False)
    (directory / "PROVENANCE.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return directory


def _story(domain: str, index: int, rng: np.random.Generator) -> dict:
    prefix = "M" if domain == "mimic" else "C"
    pid = f"{prefix}{index:05d}"
    age = int(rng.integers(48, 90))
    if domain == "crdw":
        age = min(96, age + 4)
    sex = "M" if rng.random() < 0.52 else "F"
    if index % 17 == 0:
        hf_mode = "expanded_only"
    elif index % 7 == 0:
        hf_mode = "none"
    else:
        hf_mode = "historical"
    if index % 19 == 0:
        stroke_mode = "expanded_only"
    elif index % 11 == 0:
        stroke_mode = "historical"
    else:
        stroke_mode = None
    if index % 23 == 0:
        mi_mode = "expanded_only"
    elif index % 13 == 0:
        mi_mode = "historical"
    else:
        mi_mode = None
    n_prior = int(rng.integers(0, 4))
    diuretic = hf_mode != "none" and bool(rng.random() < 0.62)
    beta_blocker = bool(rng.random() < 0.48)
    acei = bool(rng.random() < 0.36)
    high_burden = n_prior >= 2
    latent = 0.025 * (age - 70) + (0.50 if high_burden else 0.0) - (0.40 if diuretic else 0.0)
    latent += float(rng.normal(0, 0.55))
    if domain == "crdw":
        latent += 0.10
    creat = float(np.clip(0.95 + 0.22 * latent + float(rng.normal(0, 0.18)), 0.4, 6.0))
    logit = -0.25 + 0.55 * latent + float(rng.normal(0, 0.90))
    probability = float(1.0 / (1.0 + np.exp(-logit)))
    if hf_mode == "none":
        outcome = None
        gap = None
    elif rng.random() < 0.15:
        outcome = "censored"
        gap = int(rng.integers(2, 20))
    elif rng.random() < probability:
        outcome = "positive"
        gap = int(rng.integers(2, 28))
    else:
        outcome = "negative"
        gap = int(rng.integers(35, 80))
    los = int(rng.integers(3, 8))
    dual = index % 2 == 0
    return {
        "pid": pid,
        "age": age,
        "sex": sex,
        "domain": domain,
        "hf_mode": hf_mode,
        "stroke_mode": stroke_mode,
        "mi_mode": mi_mode,
        "n_prior": n_prior,
        "diuretic": diuretic,
        "beta_blocker": beta_blocker,
        "acei": acei,
        "extra_drug": (domain == "mimic" and index % 15 == 0) or (domain == "crdw" and index % 15 == 0),
        "creat": creat,
        "hgb": float(np.clip(13.0 - 0.15 * latent + float(rng.normal(0, 0.4)), 7.0, 17.0)),
        "sodium": float(np.clip(140 + float(rng.normal(0, 2.5)), 125, 155)),
        "potassium": float(np.clip(4.1 + float(rng.normal(0, 0.25)), 2.8, 6.2)),
        "sbp": float(np.clip(128 + 4 * latent + float(rng.normal(0, 8)), 90, 190)),
        "dbp": float(np.clip(76 + float(rng.normal(0, 6)), 50, 110)),
        "hr": float(np.clip(78 + float(rng.normal(0, 8)), 48, 140)),
        "include_bnp": bool(rng.random() < 0.55),
        "include_echo": index % 3 == 0,
        "include_admin": domain == "mimic" and diuretic and bool(rng.random() < 0.5),
        "dual": dual,
        "htn": domain == "crdw" or index % 3 != 0,
        "diabetes": index % 4 == 0,
        "ckd": index % 6 == 0,
        "copd": index % 9 == 0,
        "afib": index % 8 == 0,
        "outcome": outcome,
        "gap": gap,
        "los": los,
        "probability": probability,
        "inject_bad_creat": index == 1,
        "inject_bad_unit": index == 2,
    }


def _enforce_outcome_variety(stories: list[dict]) -> None:
    hf = [story for story in stories if story["hf_mode"] != "none"]
    if not hf:
        raise RuntimeError("Synthetic generator produced no heart-failure patients.")
    kinds = {story["outcome"] for story in hf}
    if "censored" not in kinds:
        hf[0]["outcome"] = "censored"
        hf[0]["gap"] = 8
    if "positive" not in {story["outcome"] for story in hf}:
        target = next(story for story in hf if story["outcome"] != "censored")
        target["outcome"] = "positive"
        target["gap"] = 12
    if "negative" not in {story["outcome"] for story in hf}:
        target = next(story for story in reversed(hf) if story["outcome"] != "censored")
        target["outcome"] = "negative"
        target["gap"] = 45


def _codes_for(story: dict) -> list[tuple[str, str, str]]:
    """Return (system, raw_code, description) for the index encounter."""
    codes: list[tuple[str, str, str]] = []
    dual = story["dual"]

    def add(system: str, raw: str, description: str, include: bool) -> None:
        if include:
            codes.append((system, raw, description))

    if story["hf_mode"] == "historical":
        add("ICD10", _choice_hf10(story), "Heart failure", True if dual or story["pid"].endswith(("0", "1", "2", "3", "4")) else False)
        add("ICD9", _choice_hf9(story), "Heart failure", dual or not any(c[0] == "ICD10" for c in codes))
        if not codes:
            codes.append(("ICD10", "I50.9", "Heart failure"))
    elif story["hf_mode"] == "expanded_only":
        add("ICD10", "I11.0", "Hypertensive heart disease with heart failure", True)
        add("ICD9", "402.91", "Hypertensive heart disease with heart failure", dual)
    if story["stroke_mode"] == "historical":
        add("ICD10", "I63.9", "Cerebral infarction", True)
        add("ICD9", "434.91", "Cerebral artery occlusion", dual)
    elif story["stroke_mode"] == "expanded_only":
        add("ICD10", "I61.9", "Intracerebral hemorrhage", True)
        add("ICD9", "431", "Intracerebral hemorrhage", dual)
    if story["mi_mode"] == "historical":
        add("ICD10", "I21.9", "Acute myocardial infarction, unspecified", True)
        add("ICD9", "410.90", "Acute myocardial infarction, unspecified", dual)
    elif story["mi_mode"] == "expanded_only":
        add("ICD10", "I21.4", "Non-ST elevation myocardial infarction", True)
        add("ICD9", "410.71", "Subendocardial infarction", dual)
    if story["htn"]:
        add("ICD10", "I10", "Essential hypertension", True)
        add("ICD9", "401.9", "Hypertension", dual)
    if story["diabetes"]:
        add("ICD10", "E11.9", "Type 2 diabetes", True)
        add("ICD9", "250.00", "Diabetes", dual)
    if story["ckd"]:
        codes.append(("ICD10", "N18.3", "Chronic kidney disease"))
    if story["copd"]:
        codes.append(("ICD10", "J44.9", "COPD"))
    if story["afib"]:
        codes.append(("ICD10", "I48.91", "Atrial fibrillation"))
    return codes


def _choice_hf10(story: dict) -> str:
    return ["I50.9", "I50.22", "I50.32", "I50.1"][story["age"] % 4]


def _choice_hf9(story: dict) -> str:
    return ["428.0", "428.21", "428.9", "428.1"][story["age"] % 4]


def _build_encounters(story: dict) -> list[dict]:
    encounters = []
    # Priors sit on the negative side of the index. CRDW later shifts to day 0.
    for k in range(story["n_prior"]):
        start = -40 - 70 * k
        encounters.append(
            {
                "key": f"P{k}",
                "start": start,
                "end": start + 1,
                "admission_type": "EMERGENCY" if k % 2 == 0 else "ELECTIVE",
                "codes": [("ICD10", "I10", "Essential hypertension")] if story["htn"] else [("ICD10", "Z00.0", "Encounter")],
                "meds": [],
                "admins": [],
                "labs": [],
                "vitals": [(start, "SBP", story["sbp"] - 2, "mmHg"), (start, "HR", story["hr"], "bpm")],
                "procedures": [],
            }
        )
    meds = []
    if story["diuretic"]:
        drug = "LASIX" if story["domain"] == "crdw" else "FUROSEMIDE"
        meds.append(drug)
    if story["beta_blocker"]:
        meds.append("METOPROLOL")
    if story["acei"]:
        meds.append("LISINOPRIL")
    if story["extra_drug"]:
        meds.append("DIGOXIN" if story["domain"] == "mimic" else "HYDRALAZINE")
    admins = ["FUROSEMIDE"] if story["include_admin"] else []
    labs = [
        (0, "CREAT", story["creat"], "mg/dL"),
        (1, "CREAT", float(np.clip(story["creat"] + 0.05, 0.4, 6.0)), "mg/dL"),
        (0, "SODIUM", story["sodium"], "mmol/L"),
        (0, "HGB", story["hgb"], "g/dL"),
    ]
    if story["potassium"] and story["age"] % 12 != 0:
        labs.append((1, "POTASSIUM", story["potassium"], "mmol/L"))
    if story["include_bnp"]:
        labs.append((0, "BNP", float(80 + 40 * max(story["creat"] - 1, 0)), "pg/mL"))
    if story["inject_bad_creat"]:
        labs.append((0, "CREAT", 500.0, "mg/dL"))
    if story["inject_bad_unit"]:
        labs.append((0, "SODIUM", story["sodium"], "furlongs"))
    los = story["los"]
    vitals = [
        (0, "SBP", story["sbp"], "mmHg"),
        (0, "DBP", story["dbp"], "mmHg"),
        (0, "HR", story["hr"], "bpm"),
        (los, "SBP", story["sbp"] - 3, "mmHg"),
    ]
    procedures = [(1, "ECHO")] if story["include_echo"] else []
    encounters.append(
        {
            "key": "INDEX",
            "start": 0,
            "end": los,
            "admission_type": "EMERGENCY",
            "codes": _codes_for(story),
            "meds": meds,
            "admins": admins,
            "labs": labs,
            "vitals": vitals,
            "procedures": procedures,
        }
    )
    if story["outcome"] == "positive":
        start = los + int(story["gap"])
        encounters.append(
            {
                "key": "NEXT",
                "start": start,
                "end": start + 3,
                "admission_type": "EMERGENCY",
                "codes": [("ICD10", "J18.9", "Pneumonia")],
                "meds": [],
                "admins": [],
                "labs": [(start, "CREAT", story["creat"], "mg/dL")],
                "vitals": [(start, "HR", story["hr"] + 5, "bpm")],
                "procedures": [],
            }
        )
        followup = start + 3
    elif story["outcome"] == "negative":
        start = los + int(story["gap"])
        encounters.append(
            {
                "key": "LATER",
                "start": start,
                "end": start + 1,
                "admission_type": "ELECTIVE",
                "codes": [("ICD10", "Z09", "Follow-up")],
                "meds": [],
                "admins": [],
                "labs": [(start, "SODIUM", story["sodium"], "mmol/L")],
                "vitals": [],
                "procedures": [],
            }
        )
        followup = start + 1
    elif story["outcome"] == "censored":
        followup = los + int(story["gap"])
    else:
        followup = los + 15
    story["followup_internal"] = followup
    return encounters


def _shift(encounters: list[dict], followup: float) -> tuple[list[dict], float]:
    times = [followup]
    for enc in encounters:
        times.extend([enc["start"], enc["end"]])
        for bucket in ("labs", "vitals", "procedures"):
            times.extend(item[0] for item in enc[bucket])
    shift = -min(times)
    for enc in encounters:
        enc["start"] += shift
        enc["end"] += shift
        enc["labs"] = [(t + shift, *rest) for t, *rest in enc["labs"]]
        enc["vitals"] = [(t + shift, *rest) for t, *rest in enc["vitals"]]
        enc["procedures"] = [(t + shift, *rest) for t, *rest in enc["procedures"]]
    return encounters, followup + shift


def _stamp(patient_index_days: float, day: float) -> str:
    base = pd.Timestamp("2116-01-01") + pd.Timedelta(days=float(patient_index_days + day))
    return base.strftime("%Y-%m-%d %H:%M:%S")


def _mimic_item(kind: str, code: str) -> str:
    return {
        ("lab", "CREAT"): "50912",
        ("lab", "SODIUM"): "50983",
        ("lab", "POTASSIUM"): "50971",
        ("lab", "HGB"): "51222",
        ("lab", "BNP"): "BNP",
        ("vital", "SBP"): "220179",
        ("vital", "DBP"): "220180",
        ("vital", "HR"): "220045",
    }[(kind, code)]


def _crdw_value(code: str, value: float, unit: str) -> tuple[float, str]:
    if code == "CREAT" and unit == "mg/dL":
        return value * 88.4, "umol/L"
    if code == "HGB" and unit == "g/dL":
        return value * 10.0, "g/L"
    return value, unit


def _mimic_tables(stories: list[dict]) -> dict[str, pd.DataFrame]:
    patients, admissions, diagnoses, prescriptions, emar = [], [], [], [], []
    labs, vitals, procedures = [], [], []
    for i, story in enumerate(stories):
        encounters = _build_encounters(story)
        offset = i * 17
        followup = story["followup_internal"]
        patients.append(
            {
                "subject_id": story["pid"],
                "gender": story["sex"],
                "anchor_age": story["age"],
                "followup_end": _stamp(offset, followup),
            }
        )
        for enc_i, enc in enumerate(encounters):
            hadm = f"{story['pid']}-E{enc_i}"
            admissions.append(
                {
                    "subject_id": story["pid"],
                    "hadm_id": hadm,
                    "admittime": _stamp(offset, enc["start"]),
                    "dischtime": _stamp(offset, enc["end"]),
                    "admission_type": enc["admission_type"],
                }
            )
            for seq, (system, raw, _name) in enumerate(enc["codes"], start=1):
                diagnoses.append(
                    {
                        "subject_id": story["pid"],
                        "hadm_id": hadm,
                        "seq_num": seq,
                        "icd_code": raw,
                        "icd_version": 9 if system == "ICD9" else 10,
                    }
                )
            for drug in enc["meds"]:
                prescriptions.append(
                    {
                        "subject_id": story["pid"],
                        "hadm_id": hadm,
                        "starttime": _stamp(offset, enc["start"]),
                        "drug": drug,
                    }
                )
            for drug in enc["admins"]:
                emar.append(
                    {
                        "subject_id": story["pid"],
                        "hadm_id": hadm,
                        "charttime": _stamp(offset, enc["start"]),
                        "medication": drug,
                    }
                )
            for t, code, value, unit in enc["labs"]:
                labs.append(
                    {
                        "subject_id": story["pid"],
                        "hadm_id": hadm,
                        "itemid": _mimic_item("lab", code),
                        "charttime": _stamp(offset, t),
                        "valuenum": value,
                        "valueuom": unit,
                    }
                )
            for t, code, value, unit in enc["vitals"]:
                vitals.append(
                    {
                        "subject_id": story["pid"],
                        "hadm_id": hadm,
                        "itemid": _mimic_item("vital", code),
                        "charttime": _stamp(offset, t),
                        "valuenum": value,
                        "valueuom": unit,
                    }
                )
            for t, code in enc["procedures"]:
                procedures.append(
                    {
                        "subject_id": story["pid"],
                        "hadm_id": hadm,
                        "chartdate": _stamp(offset, t),
                        "icd_code": "88.72" if code == "ECHO" else code,
                        "icd_version": 9,
                    }
                )
    if diagnoses:
        diagnoses.append(dict(diagnoses[0]))
        blank = dict(diagnoses[0])
        blank["subject_id"] = ""
        diagnoses.append(blank)
        bad_time = dict(labs[0])
        bad_time["charttime"] = ""
        labs.append(bad_time)
    return {
        "patients": pd.DataFrame(patients),
        "admissions": pd.DataFrame(admissions),
        "diagnoses_icd": pd.DataFrame(diagnoses),
        "prescriptions": pd.DataFrame(prescriptions),
        "emar": pd.DataFrame(emar),
        "labevents": pd.DataFrame(labs),
        "chartevents": pd.DataFrame(vitals),
        "procedures_icd": pd.DataFrame(procedures),
    }


def _crdw_tables(stories: list[dict]) -> dict[str, pd.DataFrame]:
    patients, diagnoses, medications, labs, vitals, procedures = [], [], [], [], [], []
    for story in stories:
        encounters, followup = _shift(_build_encounters(story), story["followup_internal"])
        patients.append(
            {
                "PAT_MRN_ID_ENCRYPT": story["pid"],
                "PATIENT_AGE": story["age"],
                "SEX": story["sex"],
                "FOLLOWUP_END_DAYS": followup,
            }
        )
        for enc_i, enc in enumerate(encounters):
            eid = f"{story['pid']}-E{enc_i}"
            icd10 = [raw for system, raw, _name in enc["codes"] if system == "ICD10"]
            icd9 = [raw for system, raw, _name in enc["codes"] if system == "ICD9"]
            descriptions = sorted({name for _s, _r, name in enc["codes"]})
            diagnoses.append(
                {
                    "PAT_MRN_ID_ENCRYPT": story["pid"],
                    "PAT_ENC_CSN_ID_ENCRYPT": eid,
                    "CURRENT_ICD10_LIST": ", ".join(icd10),
                    "CURRENT_ICD9_LIST": ", ".join(icd9),
                    "DX_NAME": "; ".join(descriptions),
                    "PATIENT_AGE": story["age"],
                    "DAYS_SINCE_DX_FRST_OBSRVD": enc["start"],
                }
            )
            for drug in enc["meds"]:
                medications.append(
                    {
                        "PAT_MRN_ID_ENCRYPT": story["pid"],
                        "PAT_ENC_CSN_ID_ENCRYPT": eid,
                        "MED_NAME": drug,
                        "DAYS_SINCE_DX_FRST_OBSRVD": enc["start"],
                        "ORDER_STATUS": "ordered",
                    }
                )
            for t, code, value, unit in enc["labs"]:
                stored, stored_unit = _crdw_value(code, value, unit)
                labs.append(
                    {
                        "PAT_MRN_ID_ENCRYPT": story["pid"],
                        "PAT_ENC_CSN_ID_ENCRYPT": eid,
                        "LAB_CODE": code,
                        "LAB_VALUE": stored,
                        "LAB_UNIT": stored_unit,
                        "DAYS_SINCE_DX_FRST_OBSRVD": t,
                    }
                )
            for t, code, value, unit in enc["vitals"]:
                vitals.append(
                    {
                        "PAT_MRN_ID_ENCRYPT": story["pid"],
                        "PAT_ENC_CSN_ID_ENCRYPT": eid,
                        "VITAL_CODE": code,
                        "VITAL_VALUE": value,
                        "VITAL_UNIT": unit,
                        "DAYS_SINCE_DX_FRST_OBSRVD": t,
                    }
                )
            for t, code in enc["procedures"]:
                procedures.append(
                    {
                        "PAT_MRN_ID_ENCRYPT": story["pid"],
                        "PAT_ENC_CSN_ID_ENCRYPT": eid,
                        "PROC_CODE": code,
                        "DAYS_SINCE_DX_FRST_OBSRVD": t,
                    }
                )
    if diagnoses:
        diagnoses.append(dict(diagnoses[0]))
        blank = dict(diagnoses[0])
        blank["PAT_MRN_ID_ENCRYPT"] = ""
        diagnoses.append(blank)
        bad_time = dict(labs[0])
        bad_time["DAYS_SINCE_DX_FRST_OBSRVD"] = ""
        labs.append(bad_time)
    return {
        "patients": pd.DataFrame(patients),
        "diagnoses": pd.DataFrame(diagnoses),
        "medications": pd.DataFrame(medications),
        "labs": pd.DataFrame(labs),
        "vitals": pd.DataFrame(vitals),
        "procedures": pd.DataFrame(procedures),
    }
