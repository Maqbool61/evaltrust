"""The CLI exit-code contract that CI gates depend on (issue #67):

    0  audit ran and the verdict met the gate (or no gate)
    1  audit ran but the gate failed (--strict / --fail-under / diff regression)
    2  audit could not run (bad usage, missing file, bad format, bad config)
"""

import json

from typer.testing import CliRunner

from evaltrust.cli import app

runner = CliRunner()


def _write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return str(p)


def _clean_win(tmp_path):
    raw = {"models": ["A", "B"], "examples": [
        {"id": str(i), "scores": {"A": 0, "B": 1 if i < 180 else 0}}
        for i in range(200)]}
    return _write(tmp_path, "win.json", raw)


def _noise(tmp_path):
    raw = {"models": ["A", "B"], "examples": [
        {"id": str(i), "scores": {"A": i % 2, "B": (i + 1) % 2}}
        for i in range(120)]}
    return _write(tmp_path, "noise.json", raw)


# --- 0: ran and passed the gate ------------------------------------------------

def test_exit_0_when_no_gate_set(tmp_path):
    assert runner.invoke(app, ["audit", _noise(tmp_path)]).exit_code == 0


def test_exit_0_when_gate_is_met(tmp_path):
    r = runner.invoke(app, ["audit", _clean_win(tmp_path), "--fail-under", "moderate"])
    assert r.exit_code == 0


# --- 1: ran but the gate failed ------------------------------------------------

def test_exit_1_on_strict_low_confidence(tmp_path):
    assert runner.invoke(app, ["audit", _noise(tmp_path), "--strict"]).exit_code == 1


def test_exit_1_on_fail_under_not_met(tmp_path):
    r = runner.invoke(app, ["audit", _noise(tmp_path), "--fail-under", "moderate"])
    assert r.exit_code == 1


def test_exit_1_on_diff_regression(tmp_path):
    good = tmp_path / "a.json"
    bad = tmp_path / "b.json"
    good.write_text(runner.invoke(app, ["audit", _clean_win(tmp_path), "--json"]).stdout)
    bad.write_text(runner.invoke(app, ["audit", _noise(tmp_path), "--json"]).stdout)
    assert runner.invoke(app, ["diff", str(good), str(bad)]).exit_code == 1


# --- 2: could not run ----------------------------------------------------------

def test_exit_2_on_missing_file(tmp_path):
    assert runner.invoke(app, ["audit", str(tmp_path / "nope.json")]).exit_code == 2


def test_exit_2_on_unrecognised_format(tmp_path):
    assert runner.invoke(app, ["audit", _write(tmp_path, "x.json", {"nope": 1})]).exit_code == 2


def test_exit_2_on_bad_fail_under_level(tmp_path):
    r = runner.invoke(app, ["audit", _noise(tmp_path), "--fail-under", "banana"])
    assert r.exit_code == 2


def test_exit_2_on_bad_config_path(tmp_path):
    r = runner.invoke(app, ["audit", _noise(tmp_path), "--config", str(tmp_path / "no.toml")])
    assert r.exit_code == 2


def test_exit_2_when_audit_input_is_a_directory(tmp_path):
    # A directory given where a file is expected raises IsADirectoryError (an
    # OSError, not FileNotFoundError); it must still be the "couldn't run" code.
    d = tmp_path / "adir"
    d.mkdir()
    assert runner.invoke(app, ["audit", str(d)]).exit_code == 2


def test_exit_2_when_diff_input_is_a_directory(tmp_path):
    d = tmp_path / "adir"
    d.mkdir()
    good = tmp_path / "a.json"
    good.write_text(runner.invoke(app, ["audit", _clean_win(tmp_path), "--json"]).stdout)
    assert runner.invoke(app, ["diff", str(good), str(d)]).exit_code == 2
