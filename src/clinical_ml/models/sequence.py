"""Variable-length clinical sequences.

Vocabularies and value normalizers are fitted on training patients only.
Codes that were not seen in training encode as ``<UNK>``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from clinical_ml.errors import LeakageError
from clinical_ml.features.tabular import _sex_flags

TYPE_TO_ID = {
    "diagnosis": 2,
    "medication_order": 3,
    "medication_admin": 4,
    "lab": 5,
    "vital": 6,
    "procedure": 7,
}
N_TYPES = 8
N_NUMERIC = 4
N_STATIC = 4


class Vocabulary:
    def __init__(self) -> None:
        self.itos = ["<PAD>", "<UNK>"]
        self.stoi = {"<PAD>": 0, "<UNK>": 1}

    def fit(self, codes) -> None:
        self.add(codes)

    def add(self, codes) -> list[str]:
        added = []
        for code in sorted({str(code) for code in codes}):
            if code in {"", "nan", "None", "<PAD>", "<UNK>"}:
                continue
            if code not in self.stoi:
                self.stoi[code] = len(self.itos)
                self.itos.append(code)
                added.append(code)
        return added

    def encode(self, code) -> int:
        return self.stoi.get(str(code), 1)

    def __len__(self) -> int:
        return len(self.itos)

    def to_dict(self) -> dict:
        return {"itos": list(self.itos)}

    @classmethod
    def from_dict(cls, payload: dict) -> "Vocabulary":
        vocab = cls()
        vocab.itos = list(payload["itos"])
        vocab.stoi = {code: index for index, code in enumerate(vocab.itos)}
        return vocab


class ScalarNormalizer:
    def __init__(self) -> None:
        self.mean = 0.0
        self.std = 1.0
        self.stats: dict[str, tuple[float, float]] = {}

    def fit_age(self, ages) -> None:
        values = pd.to_numeric(pd.Series(list(ages)), errors="coerce").dropna()
        if len(values) == 0:
            self.mean, self.std = 0.0, 1.0
            return
        self.mean = float(values.mean())
        self.std = float(values.std(ddof=0)) or 1.0
        if self.std < 1e-6:
            self.std = 1.0

    def transform_age(self, age) -> tuple[float, float]:
        if age is None or (isinstance(age, float) and math.isnan(age)) or pd.isna(age):
            return 0.0, 1.0
        return float(np.clip((float(age) - self.mean) / self.std, -8, 8)), 0.0

    def fit_values(self, codes, values, min_count: int = 2) -> None:
        frame = pd.DataFrame({"code": list(codes), "value": pd.to_numeric(pd.Series(list(values)), errors="coerce")})
        frame = frame[frame["value"].notna()]
        stats = {}
        for code, group in frame.groupby("code", sort=False):
            if len(group) < min_count:
                continue
            std = float(group["value"].std(ddof=0))
            if std < 1e-6:
                std = 1.0
            stats[str(code)] = (float(group["value"].mean()), std)
        self.stats = stats

    def transform_value(self, code, value) -> tuple[float, float]:
        if value is None or pd.isna(value):
            return 0.0, 1.0
        stats = self.stats.get(str(code))
        if stats is None:
            # Unseen code or too few training observations: keep the code token,
            # but do not invent a scale for the number.
            return 0.0, 1.0
        mean, std = stats
        return float(np.clip((float(value) - mean) / std, -8, 8)), 0.0

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "stats": self.stats}

    @classmethod
    def from_dict(cls, payload: dict) -> "ScalarNormalizer":
        normalizer = cls()
        normalizer.mean = float(payload.get("mean", 0.0))
        normalizer.std = float(payload.get("std", 1.0))
        normalizer.stats = {str(k): (float(v[0]), float(v[1])) for k, v in payload.get("stats", {}).items()}
        return normalizer


def pre_anchor_events(events: pd.DataFrame, labeled: pd.DataFrame, patient_ids: list[str]) -> pd.DataFrame:
    wanted = set(patient_ids)
    anchors = labeled.set_index("patient_id")
    subset = events[events["patient_id"].isin(wanted)].copy()
    if subset.empty:
        return subset
    keep = []
    for pid, group in subset.groupby("patient_id", sort=False):
        anchor = float(anchors.loc[pid, "anchor_time"])
        origin = anchors.loc[pid, "time_origin"]
        chosen = group[group["time_origin"].eq(origin) & (group["event_time"] <= anchor + 1e-6)]
        keep.append(chosen)
    if not keep:
        return subset.iloc[0:0]
    return pd.concat(keep, ignore_index=True)


def static_vector(patient, age_normalizer: ScalarNormalizer) -> list[float]:
    if patient is None:
        return [0.0, 1.0, 0.0, 1.0]
    if isinstance(patient, pd.DataFrame):
        patient = patient.iloc[0]
    age_z, age_missing = age_normalizer.transform_age(patient["age"] if "age" in patient.index else np.nan)
    sex_male, sex_missing = _sex_flags(patient["sex"] if "sex" in patient.index else "")
    return [age_z, age_missing, float(sex_male), float(sex_missing)]


def build_samples(
    events: pd.DataFrame,
    patients: pd.DataFrame,
    labeled: pd.DataFrame,
    patient_ids: list[str],
    vocab: Vocabulary,
    normalizer: ScalarNormalizer,
    max_len: int,
    use_treatment: bool,
) -> list[dict]:
    anchors = labeled.set_index("patient_id")
    patient_table = patients.set_index("patient_id") if len(patients) else pd.DataFrame()
    grouped = {pid: group for pid, group in events.groupby("patient_id", sort=False)} if len(events) else {}
    samples = []
    allowed = set(TYPE_TO_ID)
    for pid in patient_ids:
        record = anchors.loc[pid]
        if isinstance(record, pd.DataFrame):
            record = record.iloc[0]
        group = grouped.get(pid)
        if group is None:
            continue
        history = group[group["time_origin"].eq(record["time_origin"]) & (group["event_time"] <= float(record["anchor_time"]) + 1e-6)]
        history = history[history["event_type"].isin(allowed)]
        if not use_treatment:
            history = history[~history["treatment_indicator"].astype(bool)]
        history = history.sort_values(["event_time", "event_type", "event_code"], kind="mergesort").tail(max_len)
        if history.empty:
            continue
        times = history["event_time"].to_numpy(dtype=float)
        gaps = np.diff(times, prepend=times[0])
        gaps[0] = 0.0
        gaps = np.clip(gaps, 0, None)
        to_anchor = float(record["anchor_time"]) - times
        if np.any(to_anchor < -1e-6):
            raise LeakageError(f"Sequence for {pid} contains an event after the anchor.")
        tokens, types, treatment, nums = [], [], [], []
        for position, row in enumerate(history.itertuples(index=False)):
            z_value, missing = normalizer.transform_value(row.event_code, row.event_value)
            tokens.append(vocab.encode(row.event_code))
            types.append(TYPE_TO_ID.get(row.event_type, 1))
            treatment.append(1 if bool(row.treatment_indicator) else 0)
            nums.append(
                [
                    z_value,
                    missing,
                    float(math.log1p(gaps[position])),
                    float(math.log1p(max(float(to_anchor[position]), 0.0))),
                ]
            )
        patient = patient_table.loc[pid] if len(patient_table) and pid in patient_table.index else None
        samples.append(
            {
                "patient_id": str(pid),
                "tokens": tokens,
                "types": types,
                "treatment": treatment,
                "nums": nums,
                "event_times": times.tolist(),
                "anchor_time": float(record["anchor_time"]),
                "label": int(record["label"]),
                "static": static_vector(patient, normalizer),
            }
        )
    return samples


class ClinicalSequenceDataset(Dataset):
    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def collate_batch(samples: list[dict]) -> dict:
    if any(len(sample["tokens"]) == 0 for sample in samples):
        raise ValueError("Empty sequence reached the collate function.")
    batch = len(samples)
    length = max(len(sample["tokens"]) for sample in samples)
    tokens = torch.zeros(batch, length, dtype=torch.long)
    types = torch.zeros(batch, length, dtype=torch.long)
    treatment = torch.zeros(batch, length, dtype=torch.long)
    nums = torch.zeros(batch, length, N_NUMERIC, dtype=torch.float32)
    mask = torch.zeros(batch, length, dtype=torch.float32)
    static = torch.zeros(batch, N_STATIC, dtype=torch.float32)
    labels = torch.zeros(batch, dtype=torch.float32)
    ids = []
    for i, sample in enumerate(samples):
        n = len(sample["tokens"])
        tokens[i, :n] = torch.tensor(sample["tokens"], dtype=torch.long)
        types[i, :n] = torch.tensor(sample["types"], dtype=torch.long)
        treatment[i, :n] = torch.tensor(sample["treatment"], dtype=torch.long)
        nums[i, :n] = torch.tensor(sample["nums"], dtype=torch.float32)
        mask[i, :n] = 1.0
        static[i] = torch.tensor(sample["static"], dtype=torch.float32)
        labels[i] = float(sample["label"])
        ids.append(sample["patient_id"])
    return {
        "tokens": tokens,
        "types": types,
        "treatment": treatment,
        "nums": nums,
        "mask": mask,
        "static": static,
        "labels": labels,
        "patient_ids": ids,
    }


def unk_rate(samples: list[dict]) -> float | None:
    total = sum(len(sample["tokens"]) for sample in samples)
    if total == 0:
        return None
    unknown = sum(token == 1 for sample in samples for token in sample["tokens"])
    return float(unknown / total)
