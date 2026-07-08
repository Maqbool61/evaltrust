"""Tests for multi-metric ingestion: a `metric` dimension splits the data into
one dataset per metric, so the existing single-metric engine can audit each."""

import pytest

from evaltrust.adapters.common import Record, records_to_suite
from evaltrust.adapters.generic import dicts_to_records


def test_records_split_into_one_dataset_per_metric():
    records = [
        Record("q1", "A", 1.0, metric="correctness"),
        Record("q1", "B", 0.0, metric="correctness"),
        Record("q1", "A", 1.0, metric="safety"),
        Record("q1", "B", 1.0, metric="safety"),
    ]
    suite = records_to_suite(records, "test")
    assert list(suite.keys()) == ["correctness", "safety"]
    assert suite["correctness"].examples[0].scores == {"A": 1.0, "B": 0.0}
    assert suite["safety"].examples[0].scores == {"A": 1.0, "B": 1.0}


def test_single_metric_records_make_a_one_entry_suite():
    records = [Record("q1", "A", 1.0), Record("q1", "B", 0.0)]
    suite = records_to_suite(records, "test")
    assert list(suite.keys()) == ["score"]
    assert suite["score"].n_examples == 1


def test_metric_order_follows_first_appearance():
    records = [
        Record("q1", "A", 1.0, metric="safety"),
        Record("q1", "A", 1.0, metric="correctness"),
    ]
    assert list(records_to_suite(records, "t").keys()) == ["safety", "correctness"]


def test_long_records_with_metric_column_become_multi_metric():
    rows = [
        {"id": "q1", "model": "A", "metric": "correctness", "score": 1},
        {"id": "q1", "model": "B", "metric": "correctness", "score": 0},
        {"id": "q1", "model": "A", "metric": "safety", "score": 1},
        {"id": "q1", "model": "B", "metric": "safety", "score": 1},
    ]
    records = dicts_to_records(rows)
    assert {r.metric for r in records} == {"correctness", "safety"}


def test_records_without_metric_default_to_score():
    assert Record("q1", "A", 1.0).metric == "score"


# --- load_suite from disk -----------------------------------------------------

import json

from evaltrust.core.ingest import load_suite


def test_load_suite_multi_metric_csv(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("id,model,metric,score\n"
                 "q1,A,correctness,1\nq1,B,correctness,0\n"
                 "q1,A,safety,1\nq1,B,safety,1\n")
    suite = load_suite(str(p))
    assert set(suite.keys()) == {"correctness", "safety"}


def test_load_suite_single_metric_csv_is_one_entry(tmp_path):
    p = tmp_path / "w.csv"
    p.write_text("id,A,B\nq1,1,0\nq2,0,1\n")
    suite = load_suite(str(p))
    assert list(suite.keys()) == ["score"]


def test_load_suite_generic_json_with_metric(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps([
        {"id": "q1", "model": "A", "metric": "correctness", "score": 1},
        {"id": "q1", "model": "B", "metric": "correctness", "score": 0},
        {"id": "q1", "model": "A", "metric": "safety", "score": 1},
        {"id": "q1", "model": "B", "metric": "safety", "score": 0},
    ]))
    assert set(load_suite(str(p)).keys()) == {"correctness", "safety"}


def test_load_suite_native_file_is_single_metric(tmp_path):
    p = tmp_path / "n.json"
    p.write_text(json.dumps({"models": ["A", "B"],
                             "examples": [{"id": "q1", "scores": {"A": 1, "B": 0}}]}))
    assert list(load_suite(str(p)).keys()) == ["score"]
