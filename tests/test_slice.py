"""Tests for the Per-slice Comparison audit (issue #84)."""

from evaltrust.audit.slice import audit_slices
from evaltrust.audit.runner import run_audit
from evaltrust.core.schema import EvalData, Example, Status


def _data(examples):
    return EvalData(models=["A", "B"], examples=examples,
                    source_format="test", metadata={})


def _by_check(findings, check):
    (f,) = [f for f in findings if f.details.get("check") == check]
    return f


def test_missing_attribute_returns_skip():
    data = _data([Example(str(i), {"A": 0.0, "B": 1.0}) for i in range(10)])
    (f,) = audit_slices(data, "A", "B", slice_by="category")
    assert f.status is Status.SKIP
    assert f.details["reason"] == "attribute_absent"


def test_flags_slice_that_regresses_against_overall():
    # Overall B > A (B wins because 'easy' dominates), but the 'hard' slice
    # regresses (A > B on it).
    examples = []
    for i in range(60):
        examples.append(Example(f"e{i}",
                                {"A": 0.0, "B": 1.0},
                                attributes={"difficulty": "easy"}))
    for i in range(20):
        examples.append(Example(f"h{i}",
                                {"A": 1.0, "B": 0.0},
                                attributes={"difficulty": "hard"}))
    data = _data(examples)
    (f,) = audit_slices(data, "A", "B", slice_by="difficulty", seed=0)
    assert f.status is Status.WARN
    assert "hard" in f.details["regressions"]
    assert "easy" not in f.details["regressions"]
    # Bonferroni across k=2 slices halves alpha.
    assert f.details["corrected_alpha"] == 0.05 / 2


def test_no_regression_when_all_slices_agree_with_overall():
    examples = []
    for i in range(30):
        examples.append(Example(f"e{i}",
                                {"A": 0.0, "B": 1.0},
                                attributes={"language": "en"}))
    for i in range(30):
        examples.append(Example(f"f{i}",
                                {"A": 0.0, "B": 1.0},
                                attributes={"language": "fr"}))
    data = _data(examples)
    (f,) = audit_slices(data, "A", "B", slice_by="language", seed=0)
    assert f.status is Status.PASS
    assert f.details["regressions"] == []


def test_small_slices_are_reported_but_not_tested():
    examples = [
        Example("s0", {"A": 0.0, "B": 1.0}, attributes={"cat": "tiny"}),
        Example("s1", {"A": 0.0, "B": 1.0}, attributes={"cat": "tiny"}),
    ]
    examples += [
        Example(f"b{i}", {"A": 0.0, "B": 1.0}, attributes={"cat": "big"})
        for i in range(20)
    ]
    data = _data(examples)
    (f,) = audit_slices(data, "A", "B", slice_by="cat", seed=0)
    tiny = next(s for s in f.details["slices"] if s["value"] == "tiny")
    big = next(s for s in f.details["slices"] if s["value"] == "big")
    assert tiny["assessed"] is False
    assert tiny["reason"] == "too_few_examples"
    assert big["assessed"] is True


def test_run_audit_appends_slice_finding_when_slice_by_is_set():
    examples = []
    for i in range(40):
        examples.append(Example(f"e{i}", {"A": 0.0, "B": 1.0},
                                attributes={"topic": "math"}))
    for i in range(20):
        examples.append(Example(f"g{i}", {"A": 1.0, "B": 0.0},
                                attributes={"topic": "grammar"}))
    data = _data(examples)
    report = run_audit(data, model_a="A", model_b="B", slice_by="topic", seed=0)
    slice_f = _by_check(report.findings, "slice_comparison")
    assert slice_f.details["slice_by"] == "topic"
    # 'grammar' regresses against the (B-favoured) overall direction.
    assert "grammar" in slice_f.details["regressions"]


def test_run_audit_without_slice_by_does_not_add_slice_finding():
    data = _data([Example(f"e{i}", {"A": 0.0, "B": 1.0}) for i in range(10)])
    report = run_audit(data, model_a="A", model_b="B", seed=0)
    assert not any(f.details.get("check") == "slice_comparison"
                   for f in report.findings)


def test_native_adapter_reads_attributes_field():
    from evaltrust.adapters.generic import NativeNestedAdapter
    raw = {
        "examples": [
            {"id": "q1", "scores": {"A": 1.0, "B": 0.0},
             "attributes": {"category": "math", "difficulty": "easy"}},
            {"id": "q2", "scores": {"A": 0.0, "B": 1.0},
             "attributes": {"category": "code"}},
            {"id": "q3", "scores": {"A": 1.0, "B": 1.0}},
        ]
    }
    data = NativeNestedAdapter().parse(raw)
    assert data.examples[0].attributes == {"category": "math", "difficulty": "easy"}
    assert data.examples[1].attributes == {"category": "code"}
    assert data.examples[2].attributes is None
