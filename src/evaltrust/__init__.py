"""EvalTrust — is your eval's result real, or just noise?

You ran an eval and got a score gap between two models. EvalTrust does the
statistics that tell you whether that gap is a real improvement or a lucky streak,
before you ship on it. It checks whether the difference is real, big enough to
matter, and backed by enough data, and returns a High / Moderate / Low verdict.

Use the CLI (``evaltrust audit``) or this Python API:

    import evaltrust
    report = evaltrust.audit("results.json")
    print(report.verdict.level)
"""

from .api import audit
from .audit.runner import AuditReport, run_audit
from .audit.verdict import Verdict, VerdictLevel
from .core.schema import EvalData, Example, Finding, Status

__all__ = [
    "audit",
    "run_audit",
    "AuditReport",
    "Verdict",
    "VerdictLevel",
    "EvalData",
    "Example",
    "Finding",
    "Status",
]
