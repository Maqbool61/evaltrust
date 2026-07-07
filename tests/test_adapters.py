"""Tests for format adapters and auto-detection.

Each adapter answers two questions: does this raw object look like my format
(detect), and if so, map it to canonical EvalData (parse). Detection is by
structural fingerprint, never by file name.
"""

import pytest

from evallab.adapters.generic import GenericRecordsAdapter, NativeNestedAdapter
from evallab.adapters.promptfoo import PromptfooAdapter
from evallab.adapters.registry import detect_adapter, UnknownFormatError


# ---------------------------------------------------------------------------
# Native nested JSON (structured programmatic output)
# ---------------------------------------------------------------------------

NATIVE = {
    "models": ["A", "B"],
    "examples": [
        {"id": "q1", "scores": {"A": 1, "B": 0}},
        {"id": "q2", "scores": {"A": 0, "B": 1},
         "runs": {"A": [0, 1], "B": [1, 1]}},
    ],
}


def test_native_nested_detects_and_parses():
    a = NativeNestedAdapter()
    assert a.detect(NATIVE)
    data = a.parse(NATIVE)
    assert data.models == ["A", "B"]
    assert data.n_examples == 2
    assert data.examples[1].runs == {"A": [0.0, 1.0], "B": [1.0, 1.0]}


def test_native_nested_rejects_a_plain_list():
    assert not NativeNestedAdapter().detect([{"id": 1, "model": "A", "score": 1}])


# ---------------------------------------------------------------------------
# Generic records: long format (one row per model) and wide format
# ---------------------------------------------------------------------------

LONG = [
    {"id": "q1", "model": "gpt", "score": 1},
    {"id": "q1", "model": "claude", "score": 0},
    {"id": "q2", "model": "gpt", "score": 0},
    {"id": "q2", "model": "claude", "score": 1},
]

WIDE = [
    {"question": "q1", "gpt": 1, "claude": 0},
    {"question": "q2", "gpt": 0, "claude": 1},
]


def test_generic_long_records_parse():
    a = GenericRecordsAdapter()
    assert a.detect(LONG)
    data = a.parse(LONG)
    assert set(data.models) == {"gpt", "claude"}
    assert data.n_examples == 2


def test_generic_wide_records_parse():
    data = GenericRecordsAdapter().parse(WIDE)
    assert set(data.models) == {"gpt", "claude"}
    assert data.examples[0].scores["gpt"] == 1.0


def test_generic_records_wrapped_in_results_key():
    a = GenericRecordsAdapter()
    wrapped = {"results": LONG}
    assert a.detect(wrapped)
    assert a.parse(wrapped).n_examples == 2


# ---------------------------------------------------------------------------
# Promptfoo (the natural multi-provider comparison format)
# ---------------------------------------------------------------------------

PROMPTFOO = {
    "results": {
        "results": [
            {"provider": {"id": "openai:gpt-4"}, "testIdx": 0, "score": 1, "success": True},
            {"provider": {"id": "anthropic:claude"}, "testIdx": 0, "score": 0, "success": False},
            {"provider": {"id": "openai:gpt-4"}, "testIdx": 1, "score": 0, "success": False},
            {"provider": {"id": "anthropic:claude"}, "testIdx": 1, "score": 1, "success": True},
        ],
        "table": {"head": {"prompts": []}},
    },
    "version": 3,
}


def test_promptfoo_detects_and_parses_providers_as_models():
    a = PromptfooAdapter()
    assert a.detect(PROMPTFOO)
    data = a.parse(PROMPTFOO)
    assert set(data.models) == {"openai:gpt-4", "anthropic:claude"}
    assert data.n_examples == 2


def test_promptfoo_falls_back_to_success_when_no_score():
    raw = {"results": {"results": [
        {"provider": "m1", "testIdx": 0, "success": True},
        {"provider": "m2", "testIdx": 0, "success": False},
    ]}}
    data = PromptfooAdapter().parse(raw)
    assert data.examples[0].scores == {"m1": 1.0, "m2": 0.0}


# ---------------------------------------------------------------------------
# Auto-detection routing
# ---------------------------------------------------------------------------

def test_detect_routes_promptfoo_before_generic():
    assert detect_adapter(PROMPTFOO).source_format == "promptfoo"


def test_detect_routes_native_nested():
    assert detect_adapter(NATIVE).source_format == "native"


def test_detect_routes_generic_records():
    assert detect_adapter(LONG).source_format == "generic"


def test_detect_raises_helpful_error_on_unknown_shape():
    with pytest.raises(UnknownFormatError):
        detect_adapter({"totally": "unrecognised"})
