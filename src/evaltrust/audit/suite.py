"""Audit a multi-metric evaluation suite.

Real evals score several metrics per example (correctness, safety, helpfulness).
A suite is just a set of named single-metric datasets, so we audit each one with
the existing engine, comparing the *same* pair of models throughout, and correct
the significance threshold for the number of metrics tested.

Testing many metrics at the same alpha inflates false positives (test 20 metrics
at 0.05 and one looks "significant" by luck). Bonferroni divides the threshold by
the number of metrics, which is the simplest defensible correction.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from ..core.schema import EvalData
from .runner import AuditReport, run_audit
from .verdict import VerdictLevel

# Worst-to-best ordering for rolling metric verdicts up into one.
_RANK = {VerdictLevel.LOW: 0, VerdictLevel.MODERATE: 1, VerdictLevel.HIGH: 2}


@dataclass(frozen=True)
class SuiteReport:
    reports: "OrderedDict[str, AuditReport]"
    alpha: float
    corrected_alpha: float
    correction: str

    @property
    def overall_level(self) -> VerdictLevel:
        """The worst verdict across metrics — the suite is only as trustworthy as
        its weakest metric."""
        return min((r.verdict.level for r in self.reports.values()),
                   key=lambda lvl: _RANK[lvl])

    def to_dict(self) -> dict:
        return {
            "overall_level": self.overall_level.name,
            "alpha": self.alpha,
            "corrected_alpha": self.corrected_alpha,
            "correction": self.correction,
            "metrics": {m: r.to_dict() for m, r in self.reports.items()},
        }


def _suite_models(suite: dict[str, EvalData], model_a, model_b) -> tuple[str, str]:
    """Pick one model pair to compare across every metric.

    Ranks models by their mean score averaged over all metrics, so the same two
    models are compared consistently rather than a different pair per metric.
    """
    if model_a is not None and model_b is not None:
        return model_a, model_b

    totals: "OrderedDict[str, list[float]]" = OrderedDict()
    for data in suite.values():
        for m in data.models:
            vals = [ex.scores[m] for ex in data.examples if m in ex.scores]
            if vals:
                totals.setdefault(m, []).append(float(np.mean(vals)))
    if len(totals) < 2:
        raise ValueError("A suite needs at least two models to compare.")
    ranked = sorted(totals, key=lambda m: np.mean(totals[m]), reverse=True)
    return ranked[0], ranked[1]


def audit_suite(
    suite: dict[str, EvalData],
    model_a: str | None = None,
    model_b: str | None = None,
    alpha: float = 0.05,
    equivalence_margin: float = 0.05,
    seed: int = 0,
    correct: bool = True,
) -> SuiteReport:
    if not suite:
        raise ValueError("The suite is empty.")

    model_a, model_b = _suite_models(suite, model_a, model_b)

    k = len(suite)
    corrected_alpha = alpha / k if (correct and k > 1) else alpha
    correction = (f"Bonferroni: alpha {alpha} / {k} metrics = {corrected_alpha:.4f}"
                  if corrected_alpha != alpha else "none (single metric)")

    reports: "OrderedDict[str, AuditReport]" = OrderedDict()
    for metric, data in suite.items():
        reports[metric] = run_audit(
            data, model_a=model_a, model_b=model_b,
            alpha=corrected_alpha, equivalence_margin=equivalence_margin, seed=seed)

    return SuiteReport(reports=reports, alpha=alpha,
                       corrected_alpha=corrected_alpha, correction=correction)
