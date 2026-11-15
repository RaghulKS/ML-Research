"""Neural training, inference, MC dropout, and checkpoints."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from clinical_ml.evaluation.metrics import compute_metrics
from clinical_ml.models.neural import LongitudinalClassifier, model_kwargs_from_config
from clinical_ml.models.sequence import ClinicalSequenceDataset, ScalarNormalizer, Vocabulary, collate_batch
from clinical_ml.utils.seeding import set_seed


def make_loader(samples: list[dict], batch_size: int, shuffle: bool) -> DataLoader:
    if not samples:
        raise ValueError("Cannot build a loader from zero sequences.")
    return DataLoader(
        ClinicalSequenceDataset(samples),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
    )


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def predict_proba(model: LongitudinalClassifier, loader: DataLoader, device: str = "cpu") -> tuple[list[str], np.ndarray, np.ndarray]:
    device_t = torch.device(device)
    model.eval()
    ids: list[str] = []
    labels = []
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = _batch_to_device(batch, device_t)
            logits = model(batch["tokens"], batch["types"], batch["treatment"], batch["nums"], batch["mask"], batch["static"])
            probability = torch.sigmoid(logits)
            ids.extend(batch["patient_ids"])
            labels.append(batch["labels"].detach().cpu().numpy())
            probs.append(probability.detach().cpu().numpy())
    return ids, np.concatenate(labels), np.concatenate(probs)


def predict_mc_dropout(model: LongitudinalClassifier, loader: DataLoader, n_samples: int = 8, device: str = "cpu") -> dict:
    """Mean and std of probabilities with dropout left on.

    This is a stability check around the fitted predictor. It is not
    out-of-distribution detection and it is not a calibrated interval by itself.
    """
    device_t = torch.device(device)
    model.train()
    collected = []
    ids = None
    labels = None
    with torch.no_grad():
        for _ in range(n_samples):
            sample_ids: list[str] = []
            sample_y = []
            sample_p = []
            for batch in loader:
                batch = _batch_to_device(batch, device_t)
                logits = model(
                    batch["tokens"], batch["types"], batch["treatment"], batch["nums"], batch["mask"], batch["static"]
                )
                sample_ids.extend(batch["patient_ids"])
                sample_y.append(batch["labels"].detach().cpu().numpy())
                sample_p.append(torch.sigmoid(logits).detach().cpu().numpy())
            if ids is None:
                ids = sample_ids
                labels = np.concatenate(sample_y)
            collected.append(np.concatenate(sample_p))
    model.eval()
    stack = np.vstack(collected)
    return {
        "patient_ids": ids,
        "y": labels,
        "probability_mean": stack.mean(axis=0),
        "probability_std": stack.std(axis=0),
        "mean_predictive_std": float(stack.std(axis=0).mean()),
        "n_samples": n_samples,
    }


def train_model(
    model: LongitudinalClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    seed: int,
    patience: int,
    device: str = "cpu",
) -> dict:
    set_seed(seed)
    device_t = torch.device(device)
    model.to(device_t)
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters. Check the freeze mode.")
    y_parts = []
    for batch in train_loader:
        y_parts.append(batch["labels"].numpy())
    y = np.concatenate(y_parts)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32, device=device_t))
    optimizer = torch.optim.Adam(trainable, lr=lr)
    best_score = -1e18
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    wait = 0
    history = []
    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            batch = _batch_to_device(batch, device_t)
            optimizer.zero_grad()
            logits = model(batch["tokens"], batch["types"], batch["treatment"], batch["nums"], batch["mask"], batch["static"])
            loss = criterion(logits, batch["labels"])
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss was not finite.")
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * int(batch["labels"].shape[0])
            seen += int(batch["labels"].shape[0])
        _ids, val_y, val_p = predict_proba(model, val_loader, device=device)
        val_metrics = compute_metrics(val_y, val_p)
        score = val_metrics["auprc"]
        if score is None:
            score = -(val_metrics["brier"] if val_metrics["brier"] is not None else 1.0)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(seen, 1),
                "val_auroc": val_metrics["auroc"],
                "val_auprc": val_metrics["auprc"],
                "val_brier": val_metrics["brier"],
            }
        )
        if score > best_score + 1e-8:
            best_score = float(score)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= int(patience):
                break
    model.load_state_dict(best_state)
    model.eval()
    return {"best_epoch": best_epoch, "best_val_auprc": best_score, "history": history, "epochs_ran": len(history)}


def save_checkpoint(path: Path | str, model: LongitudinalClassifier, vocab: Vocabulary, normalizer: ScalarNormalizer, config: dict, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "model_kwargs": model_kwargs_from_config(model.model_type, model.event_encoder.code_emb.num_embeddings, config),
        "vocab": vocab.to_dict(),
        "normalizer": normalizer.to_dict(),
        "config": config,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: Path | str, device: str = "cpu") -> tuple[LongitudinalClassifier, Vocabulary, ScalarNormalizer, dict]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = LongitudinalClassifier(**payload["model_kwargs"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    vocab = Vocabulary.from_dict(payload["vocab"])
    normalizer = ScalarNormalizer.from_dict(payload["normalizer"])
    return model, vocab, normalizer, payload
