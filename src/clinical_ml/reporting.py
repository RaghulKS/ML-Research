"""Labels that keep synthetic runs distinct from clinical results."""

from __future__ import annotations

from clinical_ml.errors import DatasetLabelError

REAL = "REAL CLINICAL RESULT"
SYNTHETIC = "SYNTHETIC ENGINEERING VALIDATION"
NOT_EVALUATED = "IMPLEMENTED BUT NOT CLINICALLY EVALUATED"
BLOCKED = "BLOCKED BY DATA OR INFRASTRUCTURE"

ALLOWED = {REAL, SYNTHETIC, NOT_EVALUATED, BLOCKED}


def resolve_dataset_type(requested: str | None, provenance: dict | None) -> str:
    """Choose a reporting category without upgrading synthetic data to real."""
    provenance = provenance or {}
    prov_type = provenance.get("dataset_type")
    if prov_type == SYNTHETIC or provenance.get("not_clinical_data") is True:
        if requested == REAL:
            raise DatasetLabelError(
                "This extract is marked synthetic. Refusing to report it as a real clinical result."
            )
        return SYNTHETIC
    if requested in ALLOWED:
        return requested
    return NOT_EVALUATED
