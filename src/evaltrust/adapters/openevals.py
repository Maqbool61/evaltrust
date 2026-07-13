"""OpenEvals adapter (langchain-ai/openevals).

OpenEvals evaluates one model per run and returns a list of
``EvaluatorResult`` dicts — each with a ``key`` (metric name) and a
``score``.  A single export therefore contains one model; users compare
two runs with ``evaltrust audit runA.json runB.json``.

Detection is structural: a non-empty list whose first element contains
both ``key`` and ``score`` fields.  The ``comment`` field is optional
but common; other keys (``input``, ``output``, ``metadata``, ...) are
tolerated and ignored.

Robustness mirrors the CSV/Inspect paths: a row whose score is missing or
can't be read as a number is skipped and counted, rather than sinking the
whole file, so the Data Quality finding reflects the drop.
"""

from __future__ import annotations

from collections import OrderedDict

from ..core.schema import EvalData
from .common import Record, coerce_score, records_to_suite

# Explicit example-id fields only. Deliberately excludes ``input``/``question``/
# ``prompt``: two distinct evaluations that share the same input text are
# separate examples, not repeated runs of one, so we never key on the free-text
# input (which would silently merge them). With no explicit id, the row's
# position is a stable, collision-free fallback.
_ID_KEYS = ("id", "example_id", "test_id", "case_id", "index", "idx")


def _looks_like_openevals(raw) -> bool:
    if not isinstance(raw, list) or not raw:
        return False
    first = raw[0]
    return isinstance(first, dict) and "key" in first and "score" in first


def _example_id(row: dict, idx: int) -> str:
    for key in _ID_KEYS:
        if row.get(key) is not None:
            return str(row[key])
    return str(idx)


class OpenEvalsAdapter:
    source_format = "openevals"

    def detect(self, raw) -> bool:
        return _looks_like_openevals(raw)

    def _to_suite(self, raw) -> "OrderedDict[str, EvalData]":
        if not isinstance(raw, list) or not raw:
            raise ValueError("No OpenEvals results list found")

        records: list[Record] = []
        skipped = 0
        for idx, row in enumerate(raw):
            if not isinstance(row, dict):
                continue                 # not a result row; nothing to drop
            raw_score = row.get("score")
            if raw_score is None:
                skipped += 1             # missing score, counted like a bad cell
                continue
            try:
                score = coerce_score(raw_score)
            except (ValueError, TypeError):
                skipped += 1             # present but unreadable, counted
                continue
            metric = str(row.get("key") or "score")
            records.append(Record(_example_id(row, idx), "model", score, metric=metric))

        if not records:
            raise ValueError("No usable scores found in the OpenEvals results")
        return records_to_suite(records, self.source_format, {"skipped_rows": skipped})

    def parse(self, raw) -> EvalData:
        # Single-audit path: first metric is the audited one.
        return next(iter(self._to_suite(raw).values()))

    def parse_suite(self, raw) -> "OrderedDict[str, EvalData]":
        # Suite path: every distinct key fans out into its own metric.
        return self._to_suite(raw)
