"""Tests for the `evaltrust audit` command."""

import json

from typer.testing import CliRunner

from evaltrust.cli import app

runner = CliRunner()


def write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return str(p)


def clean_win_file(tmp_path):
    raw = {"models": ["A", "B"], "examples": [
        {"id": str(i), "scores": {"A": 0, "B": 1 if i < 180 else 0}}
        for i in range(200)]}
    return write(tmp_path, "win.json", raw)


def noise_file(tmp_path):
    raw = {"models": ["A", "B"], "examples": [
        {"id": str(i), "scores": {"A": i % 2, "B": (i + 1) % 2}}
        for i in range(120)]}
    return write(tmp_path, "noise.json", raw)


def test_audit_prints_report_and_exits_zero(tmp_path):
    result = runner.invoke(app, ["audit", clean_win_file(tmp_path)])
    assert result.exit_code == 0
    assert "EvalTrust" in result.stdout
    assert "High Confidence" in result.stdout


def test_missing_file_exits_nonzero_with_message(tmp_path):
    result = runner.invoke(app, ["audit", str(tmp_path / "nope.json")])
    assert result.exit_code != 0
    assert "No such" in result.stdout or "not" in result.stdout.lower()


def test_unknown_format_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["audit", write(tmp_path, "x.json", {"nope": 1})])
    assert result.exit_code != 0


def test_single_model_exits_with_helpful_message(tmp_path):
    raw = {"models": ["A"], "examples": [{"id": "1", "scores": {"A": 1}}]}
    result = runner.invoke(app, ["audit", write(tmp_path, "one.json", raw)])
    assert result.exit_code != 0
    assert "two models" in result.stdout.lower()


def test_strict_flag_fails_on_low_confidence(tmp_path):
    result = runner.invoke(app, ["audit", noise_file(tmp_path), "--strict"])
    assert result.exit_code == 1


def test_no_strict_still_exits_zero_on_low_confidence(tmp_path):
    result = runner.invoke(app, ["audit", noise_file(tmp_path)])
    assert result.exit_code == 0
    assert "Low Confidence" in result.stdout


def single_model_file(tmp_path, name, model, n, rate):
    raw = {"models": [model], "examples": [
        {"id": str(i), "scores": {model: 1 if i < int(n * rate) else 0}}
        for i in range(n)]}
    return write(tmp_path, name, raw)


def test_two_file_comparison(tmp_path):
    a = single_model_file(tmp_path, "gpt.json", "gpt", 200, 0.60)
    b = single_model_file(tmp_path, "claude.json", "claude", 200, 0.90)
    result = runner.invoke(app, ["audit", a, b])
    assert result.exit_code == 0
    assert "gpt" in result.stdout and "claude" in result.stdout


def test_two_multi_model_files_error_helpfully(tmp_path):
    raw = {"models": ["A", "B"], "examples": [{"id": "1", "scores": {"A": 1, "B": 0}}]}
    f = write(tmp_path, "multi.json", raw)
    result = runner.invoke(app, ["audit", f, f])
    assert result.exit_code == 2
    assert "one model per file" in result.stdout.lower()


def test_three_files_rejected(tmp_path):
    f = single_model_file(tmp_path, "x.json", "m", 10, 0.5)
    result = runner.invoke(app, ["audit", f, f, f])
    assert result.exit_code == 2


def test_explicit_model_selection(tmp_path):
    raw = {"models": ["A", "B", "C"], "examples": [
        {"id": str(i), "scores": {"A": 0, "B": 1, "C": 0}} for i in range(30)]}
    result = runner.invoke(
        app, ["audit", write(tmp_path, "m.json", raw), "--model-a", "A", "--model-b", "C"])
    assert result.exit_code == 0
    assert "A vs C" in result.stdout or "C vs A" in result.stdout


def test_json_output_is_valid_json(tmp_path):
    import json as _json
    result = runner.invoke(app, ["audit", clean_win_file(tmp_path), "--json"])
    assert result.exit_code == 0
    payload = _json.loads(result.stdout)
    assert payload["verdict"]["level"] == "HIGH"
    assert "findings" in payload
    # No rich box characters in JSON mode.
    assert "╭" not in result.stdout


def test_json_output_still_respects_strict(tmp_path):
    result = runner.invoke(app, ["audit", noise_file(tmp_path), "--json", "--strict"])
    assert result.exit_code == 1


def test_plain_output_is_ascii_only(tmp_path):
    result = runner.invoke(app, ["audit", noise_file(tmp_path), "--plain"])
    assert result.exit_code == 0
    assert "[fail]" in result.stdout or "[ok  ]" in result.stdout
    assert result.stdout.isascii()
    assert "╭" not in result.stdout


def multi_metric_file(tmp_path):
    lines = ["id,model,metric,score"]
    for i in range(80):
        # correctness: new clearly better; tone: identical noise
        lines.append(f"q{i},old,correctness,{1 if i % 5 else 0}")
        lines.append(f"q{i},new,correctness,{1 if i % 10 else 0}")
        lines.append(f"q{i},old,tone,{i % 2}")
        lines.append(f"q{i},new,tone,{(i + 1) % 2}")
    p = tmp_path / "suite.csv"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_multi_metric_file_produces_a_suite_report(tmp_path):
    result = runner.invoke(app, ["audit", multi_metric_file(tmp_path)])
    assert result.exit_code == 0
    assert "metrics" in result.stdout
    assert "correctness" in result.stdout and "tone" in result.stdout


def test_multi_metric_json_has_per_metric_results(tmp_path):
    import json as _json
    result = runner.invoke(app, ["audit", multi_metric_file(tmp_path), "--json"])
    payload = _json.loads(result.stdout)
    assert set(payload["metrics"].keys()) == {"correctness", "tone"}
    assert "corrected_alpha" in payload


def test_explain_flag_adds_detail(tmp_path):
    base = runner.invoke(app, ["audit", noise_file(tmp_path)])
    detailed = runner.invoke(app, ["audit", noise_file(tmp_path), "--explain"])
    assert "Detail" not in base.stdout
    assert "Detail" in detailed.stdout
