"""Tests for auditing a multi-metric suite.

A suite is audited as one single-metric audit per metric, sharing the same model
pair, with the significance threshold corrected for the number of metrics tested
(Bonferroni) so testing many metrics doesn't manufacture false positives.
"""

import json

from evaltrust.audit.suite import audit_suite
from evaltrust.audit.verdict import VerdictLevel
from evaltrust.core.schema import EvalData, Example


def metric_data(a_scores, b_scores):
    examples = [
        Example(id=str(i), scores={"A": float(a), "B": float(b)})
        for i, (a, b) in enumerate(zip(a_scores, b_scores))
    ]
    return EvalData(models=["A", "B"], examples=examples,
                    source_format="test", metadata={})


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
