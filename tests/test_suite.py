"""Tests for auditing a multi-metric suite.

A suite is audited as one single-metric audit per metric, sharing the same model
pair, with the significance threshold corrected for the number of metrics tested
(Bonferroni) so testing many metrics doesn't manufacture false positives.
"""

import json

import pytest

from evaltrust.audit.suite import audit_suite
from evaltrust.audit.verdict import VerdictLevel
from evaltrust.config import AuditConfig
from evaltrust.core.schema import EvalData, Example


def metric_data(a_scores, b_scores):
    examples = [
        Example(id=str(i), scores={"A": float(a), "B": float(b)})
        for i, (a, b) in enumerate(zip(a_scores, b_scores))
    ]
    return EvalData(models=["A", "B"], examples=examples,
                    source_format="test", metadata={})


def outcome(report, metric):
    (decision,) = [f for f in report.reports[metric].findings
                   if f.details.get("check") == "decision"]
    return decision.details["outcome"]


def test_audits_every_metric():
    suite = {
        "correctness": metric_data([0] * 200, [1] * 180 + [0] * 20),
        "safety": metric_data([1] * 100, [1] * 100),
    }
    report = audit_suite(suite, seed=0)
    assert set(report.reports.keys()) == {"correctness", "safety"}


def test_shares_one_model_pair_across_metrics():
    suite = {"m1": metric_data([0] * 30, [1] * 30),
             "m2": metric_data([1] * 30, [0] * 30)}
    report = audit_suite(suite)
    pairs = {(r.model_a, r.model_b) for r in report.reports.values()}
    assert len(pairs) == 1  # same two models compared for every metric


def test_bonferroni_corrects_alpha_by_metric_count():
    suite = {f"m{i}": metric_data([0] * 40, [1] * 40) for i in range(5)}
    report = audit_suite(suite, alpha=0.05)
    assert report.corrected_alpha == 0.05 / 5
    assert "bonferroni" in report.correction.lower()


def test_no_correction_for_single_metric():
    report = audit_suite({"score": metric_data([0] * 40, [1] * 40)}, alpha=0.05)
    assert report.corrected_alpha == 0.05


def test_overall_level_is_the_worst_metric():
    suite = {
        "good": metric_data([0] * 200, [1] * 180 + [0] * 20),   # clear win -> HIGH
        "noise": metric_data([0, 1] * 60, [1, 0] * 60),         # noise -> LOW
    }
    report = audit_suite(suite, seed=0)
    assert report.overall_level is VerdictLevel.LOW


def test_to_dict_is_json_serializable():
    suite = {"correctness": metric_data([0] * 60, [1] * 55 + [0] * 5),
             "safety": metric_data([1] * 60, [1] * 58 + [0] * 2)}
    d = audit_suite(suite, seed=0).to_dict()
    text = json.dumps(d)
    parsed = json.loads(text)
    assert set(parsed["metrics"].keys()) == {"correctness", "safety"}
    assert parsed["overall_level"] in {"HIGH", "MODERATE", "LOW"}
    assert "corrected_alpha" in parsed
    assert "metric_alphas" in parsed
    assert "adjusted_p" in parsed


# ---------------------------------------------------------------------------
# Holm-Bonferroni step-down correction
# ---------------------------------------------------------------------------

# Two metrics engineered so Holm and Bonferroni disagree at alpha = 0.05, k = 2
# (per-metric threshold alpha/2 = 0.025):
#   - "strong":     20 discordant pairs all favouring B -> McNemar p ~ 2e-6.
#   - "borderline": 10 vs 2 discordant pairs -> McNemar p = 0.03857, which sits
#     between the Bonferroni threshold (0.025) and Holm's second step (0.05).
def _strong():
    return metric_data([0] * 20, [1] * 20)


def _borderline():
    return metric_data([0] * 10 + [1] * 2, [1] * 10 + [0] * 2)


def test_holm_rejects_a_metric_bonferroni_does_not():
    suite = {"strong": _strong(), "borderline": _borderline()}
    bonf = audit_suite(suite, alpha=0.05, correction="bonferroni", seed=0)
    holm = audit_suite(suite, alpha=0.05, correction="holm", seed=0)

    # Both agree the strong metric is a real win.
    assert outcome(bonf, "strong") == "significant"
    assert outcome(holm, "strong") == "significant"
    # The borderline metric (p = 0.0386) clears Holm's second step (0.05) but not
    # the Bonferroni threshold (0.025) — this is the whole point of the feature.
    assert outcome(bonf, "borderline") != "significant"
    assert outcome(holm, "borderline") == "significant"


def test_holm_reruns_each_metric_at_its_effective_alpha():
    # The decision finding records the alpha it used; it must equal the metric's
    # Holm-effective alpha, so its status AND its equivalence CI (which uses
    # 1 - 2*alpha) are consistent with the correction actually applied.
    suite = {"strong": _strong(), "borderline": _borderline()}
    holm = audit_suite(suite, alpha=0.05, correction="holm", seed=0)
    for metric, report in holm.reports.items():
        (decision,) = [f for f in report.findings
                       if f.details.get("check") == "decision"]
        assert decision.details["alpha"] == holm.metric_alphas[metric]
    # strong is rejected first (step alpha/2), borderline second (step alpha/1).
    assert holm.metric_alphas["strong"] == pytest.approx(0.05 / 2)
    assert holm.metric_alphas["borderline"] == pytest.approx(0.05 / 1)


def test_holm_keeps_corrected_alpha_scalar_and_names_itself():
    suite = {f"m{i}": _borderline() for i in range(5)}
    holm = audit_suite(suite, alpha=0.05, correction="holm", seed=0)
    # Back-compat: corrected_alpha stays a scalar = Holm's most conservative step.
    assert holm.corrected_alpha == pytest.approx(0.05 / 5)
    assert "holm" in holm.correction.lower()


def test_holm_adjusted_p_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    suite = {"strong": _strong(), "borderline": _borderline()}
    holm = audit_suite(suite, alpha=0.05, correction="holm", seed=0)
    # Recover the raw per-metric p-values from an uncorrected run and adjust them
    # with the reference implementation.
    raw = audit_suite(suite, alpha=0.05, correction="none", seed=0)
    metrics = list(suite.keys())
    pvals = []
    for m in metrics:
        (dec,) = [f for f in raw.reports[m].findings
                  if f.details.get("check") == "decision"]
        pvals.append(dec.details["p_value"])
    _, ref_adjusted, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    got = [holm.adjusted_p[m] for m in metrics]
    assert got == pytest.approx(list(ref_adjusted))


def test_correction_none_applies_no_correction():
    suite = {"a": _borderline(), "b": _borderline()}
    none = audit_suite(suite, alpha=0.05, correction="none", seed=0)
    assert none.corrected_alpha == 0.05
    assert all(a == 0.05 for a in none.metric_alphas.values())


def test_invalid_correction_raises():
    suite = {"a": _borderline(), "b": _borderline()}
    with pytest.raises(ValueError):
        audit_suite(suite, correction="bogus")


def test_legacy_correct_false_means_no_correction():
    suite = {"a": _borderline(), "b": _borderline()}
    report = audit_suite(suite, alpha=0.05, correct=False, seed=0)
    assert report.corrected_alpha == 0.05


def test_correction_read_from_config():
    suite = {"strong": _strong(), "borderline": _borderline()}
    cfg = AuditConfig(alpha=0.05, correction="holm")
    report = audit_suite(suite, config=cfg, seed=0)
    # The config-driven Holm run rejects the borderline metric.
    assert outcome(report, "borderline") == "significant"
    assert "holm" in report.correction.lower()


def test_explicit_correction_overrides_config():
    suite = {"strong": _strong(), "borderline": _borderline()}
    cfg = AuditConfig(alpha=0.05, correction="holm")
    # An explicit argument beats the config file.
    report = audit_suite(suite, config=cfg, correction="bonferroni", seed=0)
    assert "bonferroni" in report.correction.lower()
    assert outcome(report, "borderline") != "significant"


def _holm_pvalues(suite, alpha=0.05):
    raw = audit_suite(suite, alpha=alpha, correction="none", seed=0)
    pvals = []
    for m in suite:
        (dec,) = [f for f in raw.reports[m].findings
                  if f.details.get("check") == "decision"]
        pvals.append(dec.details["p_value"])
    return pvals


def test_holm_outcomes_match_the_rejection_mask():
    # The per-metric decision outcome ("significant" or not) must agree with
    # holm_bonferroni's reject array — the audit's re-run and the reference must
    # not disagree.
    from evaltrust.stats.multiplicity import holm_bonferroni
    suite = {"strong": _strong(), "borderline": _borderline()}
    holm = audit_suite(suite, alpha=0.05, correction="holm", seed=0)
    rejected, _ = holm_bonferroni(_holm_pvalues(suite), 0.05)
    significant = [outcome(holm, m) == "significant" for m in suite]
    assert significant == rejected


def test_holm_effective_alpha_reproduces_rejection_on_the_boundary():
    # A p-value sitting EXACTLY on its Holm step threshold is the one case where
    # the audit's strict `p < alpha` and Holm's `p <= threshold` could disagree.
    # p = 0.025 at k=2, alpha=0.05 lands on the first step (0.05 / 2 = 0.025).
    from evaltrust.audit.suite import _holm_effective_alphas
    from evaltrust.stats.multiplicity import holm_bonferroni
    pvals, alpha = [0.025, 0.5], 0.05
    rejected, _ = holm_bonferroni(pvals, alpha)
    assert rejected == [True, False]           # Holm rejects the boundary metric
    eff = _holm_effective_alphas(pvals, rejected, alpha)
    decided = [pvals[i] < eff[i] for i in range(len(pvals))]
    assert decided == rejected                 # the strict `<` re-run agrees


def test_bonferroni_adjusted_p_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    suite = {"strong": _strong(), "borderline": _borderline()}
    bonf = audit_suite(suite, alpha=0.05, correction="bonferroni", seed=0)
    _, ref_adjusted, _, _ = multipletests(
        _holm_pvalues(suite), alpha=0.05, method="bonferroni")
    got = [bonf.adjusted_p[m] for m in suite]
    assert got == pytest.approx(list(ref_adjusted))
