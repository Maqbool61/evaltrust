"""Tests for the public Python API and machine-readable serialization.

Industry teams need to call EvalTrust from code and pipe its output into CI and
experiment trackers, not just read a terminal box. This is that surface.
"""

import json

from evaltrust import audit, audit_suite
from evaltrust.audit.runner import AuditReport
from evaltrust.audit.suite import SuiteReport
from evaltrust.core.schema import EvalData, Example, Finding, Status


def make_data(scores_by_model, n):
    examples = [
        Example(id=str(i), scores={m: float(s[i]) for m, s in scores_by_model.items()})
        for i in range(n)
    ]
    return EvalData(models=list(scores_by_model), examples=examples,
                    source_format="test", metadata={})


def test_audit_accepts_an_evaldata_object():
    # 90% vs 0% — a clear win without saturating the benchmark.
    report = audit(make_data({"A": [0] * 200, "B": [1] * 180 + [0] * 20}, 200))
    assert isinstance(report, AuditReport)
    assert report.verdict.level.name == "HIGH"


def test_audit_accepts_a_file_path(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"models": ["A", "B"],
                             "examples": [{"id": "q1", "scores": {"A": 1, "B": 0}}]}))
    report = audit(str(p))
    assert {report.model_a, report.model_b} == {"A", "B"}


def test_audit_accepts_two_paths_for_single_model_files(tmp_path):
    for name, model, rate in [("a.json", "gpt", 0.6), ("b.json", "claude", 0.9)]:
        (tmp_path / name).write_text(json.dumps({"models": [model], "examples": [
            {"id": str(i), "scores": {model: 1 if i < int(100 * rate) else 0}}
            for i in range(100)]}))
    report = audit([str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert {report.model_a, report.model_b} == {"gpt", "claude"}


def test_finding_to_dict_has_golden_rule_fields():
    f = Finding(pillar="P", title="t", status=Status.WARN,
                why="w", how_detected="h", how_to_fix="x", details={"p_value": 0.2})
    d = f.to_dict()
    assert d["status"] == "WARN"
    assert d["pillar"] == "P" and d["why"] == "w"
    assert d["details"]["p_value"] == 0.2


def test_report_to_dict_is_json_serializable():
    report = audit(make_data({"A": [0] * 60, "B": [1] * 55 + [0] * 5}, 60))
    d = report.to_dict()
    text = json.dumps(d)  # must not raise
    round_tripped = json.loads(text)
    assert round_tripped["verdict"]["level"] in {"HIGH", "MODERATE", "LOW"}
    assert round_tripped["models"] == [report.model_a, report.model_b]
    assert len(round_tripped["findings"]) == len(report.findings)
    assert {f["pillar"] for f in round_tripped["findings"]} >= {"Statistical Validity"}


def test_verdict_to_dict_lists_driver_titles():
    report = audit(make_data({"A": [0, 1] * 60, "B": [1, 0] * 60}, 120))
    d = report.to_dict()
    assert d["verdict"]["level"] == "LOW"
    assert isinstance(d["verdict"]["drivers"], list)
    assert all(isinstance(x, str) for x in d["verdict"]["drivers"])


def test_audit_suite_from_a_mapping():
    suite = {
        "correctness": make_data({"A": [0] * 200, "B": [1] * 180 + [0] * 20}, 200),
        "tone": make_data({"A": [0, 1] * 60, "B": [1, 0] * 60}, 120),
    }
    report = audit_suite(suite)
    assert isinstance(report, SuiteReport)
    assert set(report.reports.keys()) == {"correctness", "tone"}


def test_audit_suite_from_a_file(tmp_path):
    p = tmp_path / "s.json"
    rows = []
    for i in range(60):
        rows += [{"id": str(i), "model": "A", "metric": "m1", "score": 0},
                 {"id": str(i), "model": "B", "metric": "m1", "score": 1},
                 {"id": str(i), "model": "A", "metric": "m2", "score": 1},
                 {"id": str(i), "model": "B", "metric": "m2", "score": 1}]
    p.write_text(json.dumps(rows))
    report = audit_suite(str(p))
    assert set(report.reports.keys()) == {"m1", "m2"}
