"""Render an audit report to the terminal.

Layout follows EvalTrust's report philosophy: a plain-language verdict up top, a
compact status list of every check, then — for anything that isn't a clean pass —
the Golden Rule spelled out: why it matters, how we detected it, how to fix it.
No arbitrary aggregate score anywhere.
"""

from __future__ import annotations

import io

from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from ..audit.runner import AuditReport
from ..audit.verdict import VerdictLevel
from ..core.schema import Finding, Status

_SYMBOL = {
    Status.PASS: ("✓", "green"),
    Status.WARN: ("⚠", "yellow"),
    Status.FAIL: ("✗", "red"),
    Status.SKIP: ("·", "dim"),
}

_VERDICT_STYLE = {
    VerdictLevel.HIGH: "bold green",
    VerdictLevel.MODERATE: "bold yellow",
    VerdictLevel.LOW: "bold red",
}


def _renderable(report: AuditReport):
    verdict = report.verdict
    style = _VERDICT_STYLE[verdict.level]

    header = Text.assemble(
        ("EvalTrust Audit\n", "bold"),
        (f"Comparing {report.model_a} vs {report.model_b}  ", "cyan"),
        (f"· {report.n_examples} examples · source: {report.source_format}", "dim"),
    )
    others = [m for m in report.models_available
              if m not in (report.model_a, report.model_b)]
    if others:
        header.append(
            f"\nCompared the two strongest of {len(report.models_available)} "
            f"models; also present: {', '.join(others)} "
            "(use --model-a/--model-b to pick a different pair)", "dim")

    verdict_panel = Panel(
        Text.assemble((f"{verdict.level.value}\n", style), (verdict.summary, "")),
        border_style=style, title="Verdict", title_align="left",
    )

    # Compact status list of every check.
    checks = Text()
    for f in report.findings:
        sym, color = _SYMBOL[f.status]
        checks.append(f"  {sym} ", style=color)
        checks.append(f"{f.title}\n")
        checks.append(f"      {f.pillar}\n", style="dim")

    # Detailed Golden-Rule blocks for everything that isn't a clean pass.
    problems = [f for f in report.findings if f.status is not Status.PASS]
    detail_blocks = [_detail(f) for f in problems]

    parts = [header, Rule(style="dim"), verdict_panel,
             Rule(" Checks ", style="dim"), checks]
    if detail_blocks:
        parts.append(Rule(" What to address ", style="dim"))
        parts.extend(detail_blocks)
    return Group(*parts)


def _detail(f: Finding) -> Panel:
    sym, color = _SYMBOL[f.status]
    body = Text()
    body.append("Why it matters  ", style="bold")
    body.append(f"{f.why}\n")
    body.append("How we detected ", style="bold")
    body.append(f"{f.how_detected}\n")
    body.append("How to fix      ", style="bold")
    body.append(f"{f.how_to_fix}")
    return Panel(body, title=f"{sym} {f.title}", title_align="left",
                 border_style=color)


def render_report(report: AuditReport, width: int = 100) -> str:
    """Render the report to a plain string (used for tests and piping)."""
    console = Console(record=True, width=width, file=io.StringIO())
    console.print(_renderable(report))
    return console.export_text()


def print_report(report: AuditReport) -> None:
    """Print the report to the real terminal with colour."""
    Console().print(_renderable(report))


_PLAIN_MARK = {
    Status.PASS: "PASS",
    Status.WARN: "WARN",
    Status.FAIL: "FAIL",
    Status.SKIP: "SKIP",
}


def render_plain(report: AuditReport) -> str:
    """Render the report as plain ASCII text.

    No colour, no Unicode, no box drawing — safe for Windows terminals, CI log
    viewers, non-UTF-8 locales, and piping to a file.
    """
    v = report.verdict
    others = [m for m in report.models_available
              if m not in (report.model_a, report.model_b)]
    lines = [
        "EvalTrust Audit",
        f"{report.model_a} vs {report.model_b}  "
        f"({report.n_examples} examples, source: {report.source_format})",
    ]
    if others:
        lines.append(
            f"Compared the two strongest of {len(report.models_available)} models; "
            f"also present: {', '.join(others)}.")
    lines += [
        "",
        f"VERDICT: {v.level.value.upper()}",
        v.summary,
        "",
        "Checks:",
    ]
    for f in report.findings:
        lines.append(f"  [{_PLAIN_MARK[f.status]}] {f.title}  ({f.pillar})")

    problems = [f for f in report.findings if f.status is not Status.PASS]
    if problems:
        lines += ["", "What to address:"]
        for f in problems:
            lines += [
                f"  [{_PLAIN_MARK[f.status]}] {f.title}",
                f"    Why: {f.why}",
                f"    How: {f.how_detected}",
                f"    Fix: {f.how_to_fix}",
                "",
            ]
    return "\n".join(lines).rstrip() + "\n"
