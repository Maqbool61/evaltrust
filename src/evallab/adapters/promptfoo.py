"""Promptfoo adapter.

Promptfoo is the natural fit for EvalLab: it evaluates several *providers* across
the same test cases, which is exactly the A-vs-B comparison the auditor is built
for. Each provider becomes a model; each test case becomes an example.
"""

from __future__ import annotations

from ..core.schema import EvalData
from .common import Record, coerce_score, records_to_evaldata


def _rows(raw) -> list | None:
    results = raw.get("results") if isinstance(raw, dict) else None
    if isinstance(results, dict) and isinstance(results.get("results"), list):
        return results["results"]
    return None


def _provider_name(provider) -> str:
    if isinstance(provider, dict):
        return str(provider.get("id") or provider.get("label") or provider)
    return str(provider)


class PromptfooAdapter:
    source_format = "promptfoo"

    def detect(self, raw) -> bool:
        rows = _rows(raw)
        return bool(rows) and isinstance(rows[0], dict) and "provider" in rows[0]

    def parse(self, raw) -> EvalData:
        rows = _rows(raw)
        if not rows:
            raise ValueError("No promptfoo results array found")

        records: list[Record] = []
        for idx, row in enumerate(rows):
            model = _provider_name(row.get("provider"))
            ex_id = str(row.get("testIdx", row.get("test_idx", idx)))
            if row.get("score") is not None:
                score = coerce_score(row["score"])
            elif "success" in row:
                score = coerce_score(row["success"])
            else:
                continue
            records.append(Record(ex_id, model, score))

        return records_to_evaldata(records, self.source_format)
