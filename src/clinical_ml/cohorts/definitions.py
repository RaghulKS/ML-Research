"""Historical and expanded phenotype definitions.

Historical rules reproduce the previously supplied code lists. They are not a
complete clinical phenotype. Expanded rules are broader research definitions
and are never substituted for historical counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from clinical_ml.data.schemas import infer_system_series, normalize_icd


@dataclass(frozen=True)
class CodeRule:
    system: str
    pattern: str
    match: str

    def __post_init__(self) -> None:
        if self.system not in {"ICD9", "ICD10"}:
            raise ValueError(f"Unsupported coding system: {self.system}")
        if self.match not in {"prefix", "exact"}:
            raise ValueError(f"Unsupported match type: {self.match}")
        if not self.normalized:
            raise ValueError(f"Empty code pattern: {self.pattern}")

    @property
    def normalized(self) -> str:
        return normalize_icd(self.pattern)


@dataclass(frozen=True)
class CohortDefinition:
    condition: str
    phenotype: str
    version: str
    description: str
    rules: tuple[CodeRule, ...]

    @property
    def key(self) -> str:
        return f"{self.condition}_{self.phenotype}_v1" if not self.version else self.version


def _hf_historical() -> tuple[CodeRule, ...]:
    # Previously supplied: ICD-10 I50*, ICD-9 428*.
    return (
        CodeRule("ICD10", "I50", "prefix"),
        CodeRule("ICD9", "428", "prefix"),
    )


def _hf_expanded() -> tuple[CodeRule, ...]:
    # Adds hypertensive and rheumatic heart-failure codes. Prefixes that also
    # cover disease WITHOUT heart failure (I11, I13, 402, 404) are not used.
    return _hf_historical() + (
        CodeRule("ICD10", "I11.0", "exact"),
        CodeRule("ICD10", "I13.0", "exact"),
        CodeRule("ICD10", "I13.2", "exact"),
        CodeRule("ICD10", "I09.81", "exact"),
        CodeRule("ICD9", "402.01", "exact"),
        CodeRule("ICD9", "402.11", "exact"),
        CodeRule("ICD9", "402.91", "exact"),
        CodeRule("ICD9", "404.01", "exact"),
        CodeRule("ICD9", "404.03", "exact"),
        CodeRule("ICD9", "404.11", "exact"),
        CodeRule("ICD9", "404.13", "exact"),
        CodeRule("ICD9", "404.91", "exact"),
        CodeRule("ICD9", "404.93", "exact"),
    )


def _stroke_historical() -> tuple[CodeRule, ...]:
    # Previously supplied: ICD-10 I63*, ICD-9 434*.
    return (
        CodeRule("ICD10", "I63", "prefix"),
        CodeRule("ICD9", "434", "prefix"),
    )


def _stroke_expanded() -> tuple[CodeRule, ...]:
    # Ischemic plus hemorrhagic and unspecified stroke. 433* includes
    # precerebral occlusion with or without infarction; that limitation is
    # intentional and documented. TIA (G45 / 435) is not included.
    return _stroke_historical() + (
        CodeRule("ICD10", "I60", "prefix"),
        CodeRule("ICD10", "I61", "prefix"),
        CodeRule("ICD10", "I64", "exact"),
        CodeRule("ICD9", "430", "exact"),
        CodeRule("ICD9", "431", "exact"),
        CodeRule("ICD9", "433", "prefix"),
        CodeRule("ICD9", "436", "exact"),
    )


def _mi_historical() -> tuple[CodeRule, ...]:
    # Previously supplied: ICD-10 I21.9 (no wildcard) and ICD-9 410.9*.
    # I21.90 is a more specific ICD-10-CM string and does NOT match I21.9.
    return (
        CodeRule("ICD10", "I21.9", "exact"),
        CodeRule("ICD9", "410.9", "prefix"),
    )


def _mi_expanded() -> tuple[CodeRule, ...]:
    return (
        CodeRule("ICD10", "I21", "prefix"),
        CodeRule("ICD10", "I22", "prefix"),
        CodeRule("ICD9", "410", "prefix"),
    )


DEFINITIONS: dict[tuple[str, str], CohortDefinition] = {
    ("heart_failure", "historical"): CohortDefinition(
        condition="heart_failure",
        phenotype="historical",
        version="heart_failure_historical_v1",
        description=(
            "Historical heart-failure list: ICD-10 prefix I50 and ICD-9 prefix 428. "
            "This is the definition behind the previously reported counts."
        ),
        rules=_hf_historical(),
    ),
    ("heart_failure", "expanded"): CohortDefinition(
        condition="heart_failure",
        phenotype="expanded",
        version="heart_failure_expanded_v1",
        description=(
            "Broader heart-failure research phenotype. Includes the historical "
            "I50/428 family plus selected hypertensive and rheumatic heart-failure "
            "codes. Not a replacement for historical counts."
        ),
        rules=_hf_expanded(),
    ),
    ("stroke", "historical"): CohortDefinition(
        condition="stroke",
        phenotype="historical",
        version="stroke_historical_v1",
        description="Historical stroke list: ICD-10 prefix I63 and ICD-9 prefix 434.",
        rules=_stroke_historical(),
    ),
    ("stroke", "expanded"): CohortDefinition(
        condition="stroke",
        phenotype="expanded",
        version="stroke_expanded_v1",
        description=(
            "Broader stroke research phenotype including hemorrhagic and "
            "unspecified stroke. Not a replacement for historical counts."
        ),
        rules=_stroke_expanded(),
    ),
    ("myocardial_infarction", "historical"): CohortDefinition(
        condition="myocardial_infarction",
        phenotype="historical",
        version="mi_historical_v1",
        description=(
            "Historical myocardial-infarction list: exact ICD-10 I21.9 and "
            "ICD-9 prefix 410.9. This is intentionally narrow."
        ),
        rules=_mi_historical(),
    ),
    ("myocardial_infarction", "expanded"): CohortDefinition(
        condition="myocardial_infarction",
        phenotype="expanded",
        version="mi_expanded_v1",
        description=(
            "Broader myocardial-infarction research phenotype: ICD-10 I21*/I22* "
            "and ICD-9 410*. Not a replacement for historical counts."
        ),
        rules=_mi_expanded(),
    ),
}


def get_definition(condition: str, phenotype: str) -> CohortDefinition:
    key = (condition, phenotype)
    if key not in DEFINITIONS:
        known = ", ".join(f"{c}/{p}" for c, p in sorted(DEFINITIONS))
        raise KeyError(f"Unknown cohort {condition}/{phenotype}. Known: {known}")
    return DEFINITIONS[key]


def code_matches(code: str, system: str | None, rules: tuple[CodeRule, ...] | list[CodeRule]) -> bool:
    norm = normalize_icd(code)
    if not norm:
        return False
    resolved = system or ""
    if resolved not in {"ICD9", "ICD10"}:
        resolved = "ICD10" if norm[0].isalpha() else "ICD9" if norm[0].isdigit() else ""
    for rule in rules:
        if rule.system != resolved:
            continue
        if rule.match == "exact" and norm == rule.normalized:
            return True
        if rule.match == "prefix" and norm.startswith(rule.normalized):
            return True
    return False


def match_mask(codes: pd.Series, systems: pd.Series, rules: tuple[CodeRule, ...] | list[CodeRule]) -> pd.Series:
    """Vectorized ICD match. Systems disagreeing with a rule never match it."""
    norm = codes.fillna("").astype(str).map(normalize_icd)
    declared = systems.fillna("").astype(str).str.upper().str.replace("-", "", regex=False)
    declared = declared.replace({"9": "ICD9", "10": "ICD10", "ICD9": "ICD9", "ICD10": "ICD10"})
    inferred = infer_system_series(norm)
    resolved = declared.where(declared.isin(["ICD9", "ICD10"]), inferred)
    hit = pd.Series(False, index=codes.index)
    for rule in rules:
        if rule.match == "exact":
            code_hit = norm.eq(rule.normalized)
        else:
            code_hit = norm.str.startswith(rule.normalized, na=False) & norm.ne("")
        hit = hit | (code_hit & resolved.eq(rule.system))
    return hit.fillna(False)
