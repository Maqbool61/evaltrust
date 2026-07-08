"""Auditing a single model's eval (no second model to compare against).

The question changes from "is B better than A?" to "can I trust this score?":
how precise is it, does it clear a target, is the benchmark healthy.
"""

import numpy as np

from evaltrust.audit.single import audit_single
from evaltrust.core.schema import EvalData, Example, Status


def data_for(scores, model="m"):
    examples = [Example(id=str(i), scores={model: float(s)})
                for i, s in enumerate(scores)]
    return EvalData(models=[model], examples=examples, source_format="test",
                    metadata={})


def by_check(findings, check):
    (f,) = [f for f in findings if f.details.get("check") == check]
    return f


def test_produces_a_precision_finding():
    findings = audit_single(data_for([1] * 60 + [0] * 40), "m")
    prec = by_check(findings, "single_precision")
    assert "ci_low" in prec.details and "ci_high" in prec.details
    assert prec.details["ci_low"] < prec.details["mean"] < prec.details["ci_high"]


def test_large_sample_is_precise():
    scores = [1] * 700 + [0] * 300           # 70% over 1000 examples -> tight CI
    prec = by_check(audit_single(data_for(scores), "m"), "single_precision")
    assert prec.status is Status.PASS


def test_tiny_sample_is_imprecise():
    prec = by_check(audit_single(data_for([1, 1, 0, 1, 0, 0, 1, 0]), "m"),
                    "single_precision")
    assert prec.status is Status.WARN
    assert "example" in prec.how_to_fix.lower()


def test_threshold_clearly_above_passes():
    scores = [1] * 900 + [0] * 100           # 90%, CI well above 0.8
    dec = by_check(audit_single(data_for(scores), "m", threshold=0.8), "threshold")
    assert dec.status is Status.PASS
    assert dec.details["outcome"] == "above"


def test_threshold_clearly_below_fails():
    scores = [1] * 500 + [0] * 500           # 50%, below 0.8
    dec = by_check(audit_single(data_for(scores), "m", threshold=0.8), "threshold")
    assert dec.status is Status.FAIL
    assert dec.details["outcome"] == "below"


def test_threshold_borderline_is_inconclusive():
    # ~80% on a small sample: CI straddles the 0.8 bar.
    dec = by_check(audit_single(data_for([1] * 16 + [0] * 4), "m", threshold=0.8),
                   "threshold")
    assert dec.status is Status.WARN
    assert dec.details["outcome"] == "inconclusive"


def test_no_threshold_means_no_threshold_finding():
    findings = audit_single(data_for([1] * 50 + [0] * 50), "m")
    assert not [f for f in findings if f.details.get("check") == "threshold"]


def test_includes_benchmark_health():
    findings = audit_single(data_for([1] * 60 + [0] * 40), "m")
    assert any(f.pillar == "Benchmark Health" for f in findings)


def test_every_finding_obeys_the_golden_rule():
    for f in audit_single(data_for([1] * 60 + [0] * 40), "m", threshold=0.5):
        assert f.why.strip() and f.how_detected.strip() and f.how_to_fix.strip()


def test_run_audit_dispatches_single_model():
    from evaltrust.audit.runner import run_audit
    report = run_audit(data_for([1] * 100 + [0] * 100))
    assert report.is_single is True
    assert report.model_b is None
    assert report.to_dict()["models"] == ["m"]


def test_threshold_forces_single_mode_even_with_two_models():
    from evaltrust.audit.runner import run_audit
    d = EvalData(models=["A", "B"],
                 examples=[Example(str(i), {"A": i % 2, "B": 1}) for i in range(50)],
                 source_format="t", metadata={})
    report = run_audit(d, threshold=0.9)
    assert report.is_single is True
