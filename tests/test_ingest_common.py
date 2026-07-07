"""Tests for the shared ingestion primitives every adapter builds on.

The universal shape underneath every eval tool's output is a stream of
(example, model, score) records. Get that grouping exactly right once and every
adapter becomes a thin mapping onto it.
"""

import numpy as np
import pytest

from evallab.adapters.common import Record, coerce_score, records_to_evaldata


# ---------------------------------------------------------------------------
# coerce_score: normalise the many ways tools spell a pass/fail or number
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (1, 1.0), (0, 0.0), (0.8, 0.8), (True, 1.0), (False, 0.0),
    ("0.8", 0.8), ("1", 1.0),
    ("pass", 1.0), ("PASS", 1.0), ("fail", 0.0), ("Fail", 0.0),
    ("true", 1.0), ("false", 0.0), ("yes", 1.0), ("no", 0.0),
    ("correct", 1.0), ("incorrect", 0.0),
])
def test_coerce_score_normalises_common_spellings(raw, expected):
    assert coerce_score(raw) == pytest.approx(expected)


def test_coerce_score_rejects_nonsense():
    with pytest.raises(ValueError):
        coerce_score("banana")


def test_coerce_score_rejects_none():
    with pytest.raises(ValueError):
        coerce_score(None)


# ---------------------------------------------------------------------------
# records_to_evaldata: group flat records into canonical examples
# ---------------------------------------------------------------------------

def test_groups_records_into_examples_and_models():
    records = [
        Record("q1", "A", 1.0), Record("q1", "B", 0.0),
        Record("q2", "A", 0.0), Record("q2", "B", 1.0),
    ]
    data = records_to_evaldata(records, "test")
    assert data.n_examples == 2
    assert set(data.models) == {"A", "B"}
    assert data.examples[0].scores == {"A": 1.0, "B": 0.0}


def test_model_order_follows_first_appearance():
    records = [Record("q1", "B", 1.0), Record("q1", "A", 0.0)]
    assert records_to_evaldata(records, "test").models == ["B", "A"]


def test_repeated_records_without_judges_become_runs():
    records = [
        Record("q1", "A", 1.0), Record("q1", "A", 0.0), Record("q1", "A", 1.0),
        Record("q1", "B", 0.0), Record("q1", "B", 0.0),
    ]
    data = records_to_evaldata(records, "test")
    ex = data.examples[0]
    assert ex.runs == {"A": [1.0, 0.0, 1.0], "B": [0.0, 0.0]}
    # Final score is the mean across runs.
    assert ex.scores["A"] == pytest.approx(2 / 3)
    assert data.has_runs is True


def test_judge_tagged_records_become_judges_and_average_score():
    records = [
        Record("q1", "A", 1.0, judge="gpt"),
        Record("q1", "A", 0.0, judge="claude"),
        Record("q1", "B", 1.0, judge="gpt"),
        Record("q1", "B", 1.0, judge="claude"),
    ]
    data = records_to_evaldata(records, "test")
    ex = data.examples[0]
    assert ex.judges == {"gpt": {"A": 1.0, "B": 1.0}, "claude": {"A": 0.0, "B": 1.0}}
    assert ex.scores["A"] == pytest.approx(0.5)  # mean over judges
    assert data.has_judges is True


def test_empty_records_raise():
    with pytest.raises(ValueError):
        records_to_evaldata([], "test")
