"""Line adapter for lm-evaluation-harness sample logs."""

from __future__ import annotations

import json
from pathlib import Path
import re

from .common import Record, coerce_score


# lm_eval/evaluator.py builds these fields before evaluation_tracker.py writes
# each sample. Its ``metrics`` list names the task-defined metric fields.
_RESERVED_FIELDS = {
    "doc_id",
    "doc",
    "target",
    "arguments",
    "resps",
    "filtered_resps",
    "filter",
    "metrics",
    "doc_hash",
    "prompt_hash",
    "target_hash",
}
# samples_<task>_<timestamp>.jsonl; timestamp is shared with results_<timestamp>.json
_TIMESTAMP_SUFFIX = re.compile(
    r"^(?P<task>.+)_(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:\.\d+)?)$"
)


def _model_from_path(path: Path | None) -> str:
    stem = path.stem if path is not None else "model"
    if stem.startswith("samples_"):
        stem = stem[len("samples_"):]
    match = _TIMESTAMP_SUFFIX.fullmatch(stem)
    return match.group("task") if match else stem


def _samples_timestamp(path: Path) -> str | None:
    stem = path.stem
    if stem.startswith("samples_"):
        stem = stem[len("samples_"):]
    match = _TIMESTAMP_SUFFIX.fullmatch(stem)
    return match.group("timestamp") if match else None


def _model_name_from_results(path: Path) -> str | None:
    """Read top-level ``model_name`` from a results_*.json file.

    Any failure degrades to None so a broken sibling never crashes a loadable
    samples file.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        return None
    if not isinstance(raw, dict):
        return None
    name = raw.get("model_name")
    if isinstance(name, str) and name:
        return name
    return None


def _choose_results_file(path: Path) -> Path | None:
    """Pick a sibling results_*.json for ``path``, or None.

    Prefer a timestamp match with the samples filename. If none match and
    exactly one results_*.json sits beside the samples file, use that one.
    The sole-file fallback can mislabel when a directory mixes unrelated
    runs; callers that care should keep each run in its own directory.
    """
    try:
        candidates = sorted(path.parent.glob("results_*.json"))
    except OSError:
        return None
    if not candidates:
        return None

    timestamp = _samples_timestamp(path)
    if timestamp is not None:
        prefix = "results_"
        for candidate in candidates:
            stem = candidate.stem
            if stem.startswith(prefix) and stem[len(prefix):] == timestamp:
                return candidate

    if len(candidates) == 1:
        return candidates[0]
    return None


def _resolve_model(path: Path | None) -> tuple[str, dict]:
    """Return (model_name, name metadata).

    When a sibling results file supplies a non-empty model_name string, mark
    the name as not inferred and record the results filename only.
    """
    inferred = _model_from_path(path)
    if path is None:
        return inferred, {"model_name_inferred": True}

    results_path = _choose_results_file(path)
    if results_path is None:
        return inferred, {"model_name_inferred": True}

    name = _model_name_from_results(results_path)
    if name is None:
        return inferred, {"model_name_inferred": True}

    return name, {
        "model_name_inferred": False,
        "model_name_source": results_path.name,
    }


class LMEvalAdapter:
    source_format = "lm-eval"

    def detect_lines(self, rows: list[dict]) -> bool:
        if not rows or not isinstance(rows[0], dict):
            return False
        first = rows[0]
        # lm-eval samples have no model key, so generic long rows cannot match.
        return (
            "doc_id" in first
            and ("resps" in first or "filtered_resps" in first)
            and "model" not in first
        )

    def parse_lines(
        self, rows: list[dict], *, path: Path | None = None
    ) -> tuple[list[Record], dict]:
        if not self.detect_lines(rows):
            raise ValueError("Not an lm-eval sample log")

        model, name_meta = _resolve_model(path)
        records: list[Record] = []
        skipped = 0
        for row in rows:
            doc_id = row.get("doc_id")
            if doc_id is None:
                skipped += 1
                continue
            declared_metrics = row.get("metrics")
            metric_fields = (
                declared_metrics
                if isinstance(declared_metrics, list)
                else row.keys()
            )
            for metric in metric_fields:
                if metric in _RESERVED_FIELDS:
                    continue
                raw = row.get(metric)
                try:
                    score = coerce_score(raw)
                except ValueError:
                    skipped += 1
                    continue
                records.append(Record(str(doc_id), model, score, metric=str(metric)))

        if not records:
            raise ValueError("No scored samples found in the lm-eval sample log")
        return records, {"skipped_rows": skipped, **name_meta}
