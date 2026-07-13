import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CITATION = ROOT / "CITATION.cff"


def test_citation_is_valid_cff():
    """CITATION.cff must pass CFF schema validation, else GitHub silently
    drops the citation button. A version-only check can't catch that."""
    if not shutil.which("cffconvert"):
        pytest.skip("cffconvert not installed (pip install -e .[dev])")
    result = subprocess.run(
        ["cffconvert", "--validate", "-i", str(CITATION)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_citation_version_matches_pyproject():
    """CITATION.cff version must track pyproject.toml so citations aren't stale."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    expected = pyproject["project"]["version"]
    # Regex not YAML parse — avoids a PyYAML dep; CFF `version` is
    # always a top-level flat scalar, so it can't misfire. Upgrade to a real
    # YAML parse only if the field ever becomes nested/multi-line.
    cff = CITATION.read_text()
    match = re.search(r"^version:\s*(.+)$", cff, re.MULTILINE)
    assert match, "no version: line in CITATION.cff"
    actual = match.group(1).strip().strip('"')
    assert actual == expected, (
        f"CITATION.cff version ({actual}) != pyproject.toml version ({expected}); "
        f"bump CITATION.cff to {expected}"
    )
