"""GRU and compact Transformer classifiers for pre-anchor event sequences.

Treatment events use a learned embedding that is separate from diagnosis and
measurement tokens. Removing treatment events is an ablation of inputs, not an
estimate of treatment effect.
"""

from __future__ import annotations

import torch
from torch import nn

from clinical_ml.models.sequence import N_NUMERIC, N_STATIC, N_TYPES


class EventEncoder(nn.Module):
    def __init__(self, n_codes: int, n_types: int, d_model: int, n_numeric: int = N_NUMERIC) -> None:
        super().__init__()
        self.code_emb = nn.Embedding(n_codes, d_model, padding_idx=0)
        self.type_emb = nn.Embedding(n_types, d_model, padding_idx=0)
        self.treatment_emb = nn.Embedding(2, d_model)
        self.num_proj = nn.Linear(n_numeric, d_model)

    def forward(self, tokens, types, treatment, nums):
        return (
            self.code_emb(tokens)
            + self.type_emb(types)
            + self.treatment_emb(treatment)
            + self.num_proj(nums)
        )


class LongitudinalClassifier(nn.Module):
    def __init__(
        self,
        model_type: str,
        n_codes: int,
        d_model: int = 48,
        hidden_size: int = 48,
        n_layers: int = 1,
        n_heads: int = 4,
        dropout: float = 0.2,
        max_len: int = 32,
        n_types: int = N_TYPES,
        n_static: int = N_STATIC,
    ) -> None:
        super().__init__()
        if model_type not in {"gru", "transformer"}:
            raise ValueError(f"Unsupported model_type {model_type}")
        self.model_type = model_type
        self.max_len = max_len
        self.event_encoder = EventEncoder(n_codes, n_types, d_model)
        self.drop = nn.Dropout(dropout)
        if model_type == "gru":
            self.backbone = nn.GRU(
                input_size=d_model,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            head_in = hidden_size + n_static
        else:
            if d_model % n_heads != 0:
                raise ValueError("d_model must be divisible by n_heads")
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            try:
                self.backbone = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
            except TypeError:
                self.backbone = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.pos = nn.Embedding(max_len, d_model)
            head_in = d_model + n_static
        self.head = nn.Linear(head_in, 1)

    def forward(self, tokens, types, treatment, nums, mask, static):
        if (mask.sum(dim=1) == 0).any():
            raise RuntimeError("Batch contains an empty patient sequence.")
        hidden = self.event_encoder(tokens, types, treatment, nums)
        if self.model_type == "transformer":
            positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
            positions = positions.expand(tokens.size(0), -1)
            hidden = hidden + self.pos(positions)
        hidden = self.drop(hidden)
        if self.model_type == "gru":
            lengths = mask.sum(dim=1).to(dtype=torch.long).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(hidden, lengths, batch_first=True, enforce_sorted=False)
            _output, state = self.backbone(packed)
            pooled = state[-1]
        else:
            encoded = self.backbone(hidden, src_key_padding_mask=mask <= 0)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / denom
        pooled = self.drop(pooled)
        return self.head(torch.cat([pooled, static], dim=-1)).squeeze(-1)


def set_trainable(model: nn.Module, mode: str) -> None:
    if mode not in {"full", "head_only", "freeze_embeddings"}:
        raise ValueError(f"Unknown freeze mode: {mode}")
    for name, param in model.named_parameters():
        if mode == "full":
            param.requires_grad = True
        elif mode == "head_only":
            param.requires_grad = name.startswith("head.")
        else:
            frozen = name.startswith("event_encoder.code_emb") or name.startswith("event_encoder.type_emb")
            param.requires_grad = not frozen


def resize_code_embeddings(model: LongitudinalClassifier, new_size: int) -> None:
    current = model.event_encoder.code_emb
    if new_size == current.num_embeddings:
        return
    if new_size < current.num_embeddings:
        raise ValueError("Refusing to shrink code embeddings.")
    expanded = nn.Embedding(new_size, current.embedding_dim, padding_idx=0)
    with torch.no_grad():
        expanded.weight[: current.num_embeddings] = current.weight
        expanded.weight[current.num_embeddings :] = current.weight[1]
    model.event_encoder.code_emb = expanded


def model_kwargs_from_config(model_type: str, n_codes: int, config: dict) -> dict:
    return {
        "model_type": model_type,
        "n_codes": n_codes,
        "d_model": int(config.get("d_model", 48)),
        "hidden_size": int(config.get("hidden_size", 48)),
        "n_layers": int(config.get("n_layers", 1 if model_type == "gru" else 2)),
        "n_heads": int(config.get("n_heads", 4)),
        "dropout": float(config.get("dropout", 0.2)),
        "max_len": int(config.get("max_len", 32)),
    }
