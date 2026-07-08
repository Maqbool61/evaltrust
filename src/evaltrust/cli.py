"""The `evaltrust` command-line interface.

One command, no configuration:

    evaltrust audit results.json

Reads your existing eval output, audits whether its conclusion is trustworthy,
and prints the verdict. ``--strict`` makes a Low-Confidence verdict fail the
process, so an audit can gate CI the way tests do.
"""

from __future__ import annotations

import json
from typing import List, Optional

import typer
from rich.console import Console

from .adapters.registry import UnknownFormatError
from .audit.runner import run_audit
from .audit.suite import audit_suite
from .audit.verdict import VerdictLevel
from .core.ingest import load_comparison, load_suite
from .report.terminal import (
    print_report,
    print_suite,
    render_plain,
    render_suite_plain,
)

app = typer.Typer(
    add_completion=False,
    help="Check whether an eval's result is real or just noise.")
_err = Console(stderr=False)  # keep errors on stdout so they're easy to capture


@app.callback()
def main() -> None:
    """EvalTrust: you ran an eval and got a score gap between two models. This
    tells you whether that gap is a real improvement or just luck, before you
    ship on it."""


@app.command()
def audit(
    results: List[str] = typer.Argument(
        ..., help="One results file (JSON/CSV), or two single-model files to compare."),
    model_a: Optional[str] = typer.Option(
        None, "--model-a", help="Model to compare (or label for the first file)."),
    model_b: Optional[str] = typer.Option(
        None, "--model-b", help="Model to compare (or label for the second file)."),
    alpha: float = typer.Option(0.05, "--alpha", help="Significance level."),
    equivalence_margin: float = typer.Option(
        0.05, "--equivalence-margin",
        help="Largest score gap considered practically negligible (for equivalence)."),
    seed: int = typer.Option(0, "--seed", help="Seed for reproducible resampling."),
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero if confidence is Low."),
    as_json: bool = typer.Option(
        False, "--json", help="Emit the audit as JSON (for CI and tooling)."),
    plain: bool = typer.Option(
        False, "--plain", help="Plain ASCII output (no colour or Unicode)."),
    explain: bool = typer.Option(
        False, "--explain", help="Also show why each flag matters and how it was measured."),
) -> None:
    """Audit an evaluation and print a confidence verdict.

    Pass one file that already contains multiple models, or two single-model
    files (e.g. two DeepEval runs) to pair into an A-vs-B comparison.
    """
    if len(results) > 2:
        _err.print("[red]Provide at most two files (one, or two to compare).[/red]")
        raise typer.Exit(code=2)

    suite_report = None
    report = None
    try:
        if len(results) == 2:
            data = load_comparison(results, label_a=model_a, label_b=model_b)
            report = run_audit(data, alpha=alpha,
                               equivalence_margin=equivalence_margin, seed=seed)
        else:
            suite = load_suite(results[0])
            if len(suite) > 1:
                suite_report = audit_suite(
                    suite, model_a=model_a, model_b=model_b, alpha=alpha,
                    equivalence_margin=equivalence_margin, seed=seed)
            else:
                data = next(iter(suite.values()))
                report = run_audit(data, model_a=model_a, model_b=model_b,
                                   alpha=alpha, equivalence_margin=equivalence_margin,
                                   seed=seed)
    except FileNotFoundError as e:
        _err.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2)
    except UnknownFormatError as e:
        _err.print(f"[red]Unrecognised evaluation format.[/red]\n{e}")
        raise typer.Exit(code=2)
    except ValueError as e:
        _err.print(f"[red]{e}[/red]")
        raise typer.Exit(code=2)

    if suite_report is not None:
        if as_json:
            typer.echo(json.dumps(suite_report.to_dict(), indent=2))
        elif plain:
            typer.echo(render_suite_plain(suite_report, explain=explain), nl=False)
        else:
            print_suite(suite_report, explain=explain)
        level = suite_report.overall_level
    else:
        if as_json:
            typer.echo(json.dumps(report.to_dict(), indent=2))
        elif plain:
            typer.echo(render_plain(report, explain=explain), nl=False)
        else:
            print_report(report, explain=explain)
        level = report.verdict.level

    if strict and level is VerdictLevel.LOW:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
