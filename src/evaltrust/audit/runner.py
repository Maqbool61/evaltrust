"""Runs every applicable audit check and assembles the final report."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.schema import EvalData, Finding
from .benchmark_health import audit_benchmark_health
from .judge_reliability import audit_judge_reliability
from .repeatability import audit_repeatability
from .statistical import audit_statistical_validity
from .verdict import Verdict, compute_verdict


@dataclass(frozen=True)
class AuditReport:
    model_a: str
    model_b: str
    n_examples: int
    source_format: str
    findings: list[Finding]
    verdict: Verdict
    models_available: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """A JSON-serializable representation of the whole audit."""
        return {
            "models": [self.model_a, self.model_b],
            "model_a": self.model_a,
            "model_b": self.model_b,
            "models_available": self.models_available,
            "n_examples": self.n_examples,
            "source_format": self.source_format,
            "verdict": self.verdict.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
        }


def _mean_score(data: EvalData, model: str) -> float:
    vals = [ex.scores[model] for ex in data.examples if model in ex.scores]
    return float(np.mean(vals)) if vals else float("-inf")


def _pick_models(data: EvalData) -> tuple[str, str]:
    """Compare the two strongest models by mean score (stable, documented)."""
    if len(data.models) < 2:
        raise ValueError("EvalTrust needs at least two models to compare.")
    ranked = sorted(data.models, key=lambda m: _mean_score(data, m), reverse=True)
    return ranked[0], ranked[1]


def run_audit(
    data: EvalData,
    model_a: str | None = None,
    model_b: str | None = None,
    alpha: float = 0.05,
    equivalence_margin: float = 0.05,
    seed: int = 0,
) -> AuditReport:
    if model_a is None or model_b is None:
        model_a, model_b = _pick_models(data)

    findings: list[Finding] = []
    findings += audit_statistical_validity(
        data, model_a, model_b, alpha=alpha,
        equivalence_margin=equivalence_margin, seed=seed)
    findings += audit_benchmark_health(data, [model_a, model_b])
    findings += audit_repeatability(data, model_a, model_b)
    findings += audit_judge_reliability(data, model_a, model_b)

    return AuditReport(
        model_a=model_a,
        model_b=model_b,
        n_examples=data.n_examples,
        source_format=data.source_format,
        findings=findings,
        verdict=compute_verdict(findings),
        models_available=list(data.models),
    )
