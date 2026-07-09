"""Audit a multi-metric evaluation suite.

Real evals score several metrics per example (correctness, safety, helpfulness).
A suite is just a set of named single-metric datasets, so we audit each one with
the existing engine, comparing the *same* pair of models throughout, and correct
the significance threshold for the number of metrics tested.

Testing many metrics at the same alpha inflates false positives (test 20 metrics
at 0.05 and one looks "significant" by luck). Two corrections are available:

- **Bonferroni** divides the threshold by the number of metrics (``alpha / k``) —
  the simplest defensible correction, and the default.
- **Holm-Bonferroni** is a step-down refinement: it ranks the metrics by p-value
  and tests the i-th smallest against ``alpha / (k - i)``, so it rejects at least
  as many metrics as Bonferroni while controlling the same family-wise error
  rate. Because a metric's threshold depends on the *rank* of its p-value, Holm
  runs in two passes — once to read every p-value, then again re-running each
  metric at its Holm-effective alpha so its status, prose, and equivalence CI are
  all consistent with the correction applied.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace

import numpy as np

from ..config import AuditConfig
from ..core.schema import EvalData, Finding
from ..stats.multiplicity import holm_bonferroni
from .runner import AuditReport, run_audit
from .verdict import VerdictLevel, enforce_level

# Worst-to-best ordering for rolling metric verdicts up into one.
_RANK = {VerdictLevel.LOW: 0, VerdictLevel.MODERATE: 1, VerdictLevel.HIGH: 2}

_METHODS = ("bonferroni", "holm", "none")


@dataclass(frozen=True)
class SuiteReport:
    reports: "OrderedDict[str, AuditReport]"
    alpha: float
    corrected_alpha: float
    correction: str
    metric_alphas: "OrderedDict[str, float]" = field(default_factory=OrderedDict)
    adjusted_p: "OrderedDict[str, float]" = field(default_factory=OrderedDict)

    @property
    def overall_level(self) -> VerdictLevel:
        """The worst verdict across metrics — the suite is only as trustworthy as
        its weakest metric."""
        return min((r.verdict.level for r in self.reports.values()),
                   key=lambda lvl: _RANK[lvl])

    def raise_if_below(self, minimum: "str | VerdictLevel" = "moderate") -> "SuiteReport":
        """Raise UntrustworthyError if the suite's overall (weakest) confidence is
        below ``minimum``. Returns self so it can be chained."""
        enforce_level(self.overall_level, minimum, context="the metric suite")
        return self

    def to_dict(self) -> dict:
        return {
            "overall_level": self.overall_level.name,
            "alpha": self.alpha,
            "corrected_alpha": self.corrected_alpha,
            "correction": self.correction,
            "metric_alphas": dict(self.metric_alphas),
            "adjusted_p": dict(self.adjusted_p),
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
    config: "AuditConfig | None" = None,
    correction: str | None = None,
) -> SuiteReport:
    """Audit every metric in a suite for the same model pair.

    ``correction`` selects the multiple-comparison correction over
    ``{"bonferroni", "holm", "none"}``; ``None`` defers to the config
    (``AuditConfig.correction``, default ``"bonferroni"``). ``correct`` is a
    deprecated legacy switch — ``correct=False`` still forces no correction; new
    code should pass ``correction="none"`` instead.
    """
    if not suite:
        raise ValueError("The suite is empty.")

    cfg = config or AuditConfig(alpha=alpha, equivalence_margin=equivalence_margin,
                                seed=seed)

    # Resolve the correction: an explicit argument beats the config; the legacy
    # `correct=False` switch still forces no correction.
    method = correction if correction is not None else cfg.correction
    if not correct:
        method = "none"
    if method not in _METHODS:
        raise ValueError(
            f"correction must be one of {_METHODS}, got {method!r}")

    model_a, model_b = _suite_models(suite, model_a, model_b)
    k = len(suite)

    # A single metric can't inflate anything, so no correction ever applies.
    if k == 1 or method == "none":
        return _uncorrected_suite(suite, model_a, model_b, cfg, single=(k == 1))
    if method == "bonferroni":
        return _bonferroni_suite(suite, model_a, model_b, cfg, k)
    return _holm_suite(suite, model_a, model_b, cfg, k)


def _run_metrics(suite, model_a, model_b, cfg_for) -> "OrderedDict[str, AuditReport]":
    """Run one audit per metric; ``cfg_for(metric)`` supplies each metric's config."""
    reports: "OrderedDict[str, AuditReport]" = OrderedDict()
    for metric, data in suite.items():
        reports[metric] = run_audit(
            data, model_a=model_a, model_b=model_b, config=cfg_for(metric))
    return reports


def _decision(report: AuditReport) -> "Finding | None":
    for f in report.findings:
        if f.details.get("check") == "decision":
            return f
    return None


def _pvalue(report: AuditReport) -> float:
    finding = _decision(report)
    return float(finding.details["p_value"]) if finding is not None else 1.0


def _n_significant(reports) -> int:
    return sum(1 for r in reports.values()
               if (f := _decision(r)) is not None
               and f.details.get("outcome") == "significant")


def _uncorrected_suite(suite, model_a, model_b, cfg, single: bool) -> SuiteReport:
    reports = _run_metrics(suite, model_a, model_b, lambda _m: cfg)
    metric_alphas = OrderedDict((m, cfg.alpha) for m in suite)
    adjusted_p = OrderedDict((m, _pvalue(reports[m])) for m in suite)
    description = "none (single metric)" if single else "none (uncorrected)"
    return SuiteReport(reports=reports, alpha=cfg.alpha, corrected_alpha=cfg.alpha,
                       correction=description, metric_alphas=metric_alphas,
                       adjusted_p=adjusted_p)


def _bonferroni_suite(suite, model_a, model_b, cfg, k: int) -> SuiteReport:
    corrected_alpha = cfg.alpha / k
    metric_cfg = replace(cfg, alpha=corrected_alpha)
    reports = _run_metrics(suite, model_a, model_b, lambda _m: metric_cfg)
    metric_alphas = OrderedDict((m, corrected_alpha) for m in suite)
    adjusted_p = OrderedDict(
        (m, min(k * _pvalue(reports[m]), 1.0)) for m in suite)
    description = (f"Bonferroni: alpha {cfg.alpha} / {k} metrics "
                  f"= {corrected_alpha:.4f}")
    return SuiteReport(reports=reports, alpha=cfg.alpha,
                       corrected_alpha=corrected_alpha, correction=description,
                       metric_alphas=metric_alphas, adjusted_p=adjusted_p)


def _holm_suite(suite, model_a, model_b, cfg, k: int) -> SuiteReport:
    metrics = list(suite)

    # Pass 1: run each metric at the raw alpha only to read its p-value (the
    # p-value doesn't depend on alpha, so this pass exists purely to rank them).
    pass1 = _run_metrics(suite, model_a, model_b, lambda _m: cfg)
    pvalues = [_pvalue(pass1[m]) for m in metrics]

    rejected, adjusted = holm_bonferroni(pvalues, cfg.alpha)
    effective = _holm_effective_alphas(pvalues, rejected, cfg.alpha)
    alpha_for = dict(zip(metrics, effective))

    # Pass 2: re-run each metric at its Holm-effective alpha so the decision
    # status, prose, and equivalence CI (which uses 1 - 2*alpha) match it.
    reports = _run_metrics(
        suite, model_a, model_b,
        lambda m: replace(cfg, alpha=alpha_for[m]))

    metric_alphas = OrderedDict((m, alpha_for[m]) for m in metrics)
    adjusted_p = OrderedDict((m, adjusted[i]) for i, m in enumerate(metrics))
    # corrected_alpha stays a scalar for back-compat: Holm's most conservative
    # (first) step is alpha / k, the same value Bonferroni uses everywhere.
    corrected_alpha = cfg.alpha / k
    description = (f"Holm-Bonferroni (step-down): {_n_significant(reports)} "
                  f"of {k} metrics significant")
    return SuiteReport(reports=reports, alpha=cfg.alpha,
                       corrected_alpha=corrected_alpha, correction=description,
                       metric_alphas=metric_alphas, adjusted_p=adjusted_p)


def _holm_effective_alphas(pvalues, rejected, alpha: float) -> list[float]:
    """Per-metric alpha reproducing the Holm decision under the audit's ``p < alpha``.

    A rejected metric uses its own step threshold ``alpha / (k - rank)``; a
    retained metric uses the threshold of the first failure, ``alpha / (k - m)``
    where ``m`` is the number rejected (the step-down stops there). Ties are
    broken by input order (a stable sort), exactly as ``holm_bonferroni`` does,
    so two metrics with an identical p-value can receive different step
    thresholds — that is a genuine property of step-down Holm, not an artefact.

    One subtlety: ``holm_bonferroni`` rejects with ``adjusted_p <= alpha`` (the
    statsmodels convention) while the audit decides significance with a strict
    ``p < alpha``. On the measure-zero boundary where a rejected metric's p-value
    equals its step threshold exactly, we nudge the threshold up by one ULP so
    the strict comparison still fires — keeping ``p < effective_alpha``
    equivalent to Holm's rejection for every metric, with no observable effect
    away from that exact tie.
    """
    p = np.asarray(pvalues, dtype=float)
    k = p.size
    order = np.argsort(p, kind="stable")
    rank = np.empty(k, dtype=int)
    rank[order] = np.arange(k)
    n_rejected = int(sum(rejected))

    effective = []
    for i in range(k):
        if rejected[i]:
            threshold = alpha / (k - rank[i])
            if p[i] >= threshold:  # exact tie: p == threshold, Holm rejects via <=
                threshold = float(np.nextafter(threshold, np.inf))
            effective.append(float(threshold))
        else:
            effective.append(alpha / (k - n_rejected))
    return effective
