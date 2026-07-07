"""Load an evaluation file from disk and normalise it to canonical EvalData.

The user runs ``evallab audit results.json`` (or ``.csv``) and never thinks about
formats. This module reads the file, routes JSON through structural auto-detection
and CSV through the shared record extractor, and returns EvalData.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .schema import EvalData
from ..adapters.common import records_to_evaldata
from ..adapters.generic import dicts_to_records
from ..adapters.registry import UnknownFormatError, detect_adapter


def _load_csv(text: str) -> EvalData:
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise UnknownFormatError("The CSV file has no data rows.")
    return records_to_evaldata(dicts_to_records(rows), "csv")


def _load_json(text: str) -> EvalData:
    raw = json.loads(text)
    return detect_adapter(raw).parse(raw)


def load(path: str) -> EvalData:
    """Read ``path`` and return canonical EvalData.

    Routing is by extension, with a content fallback: a ``.json`` file goes
    through JSON auto-detection, a ``.csv`` file through the CSV reader, and
    anything else is tried as JSON then CSV.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such evaluation file: {path}")

    text = p.read_text()
    suffix = p.suffix.lower()

    if suffix == ".csv":
        return _load_csv(text)
    if suffix == ".json":
        return _load_json(text)

    try:
        return _load_json(text)
    except (json.JSONDecodeError, UnknownFormatError):
        return _load_csv(text)
