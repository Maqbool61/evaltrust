# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — 2026-07-07

### Added

- **Python API**: `evaltrust.audit(source)` accepts a path, two paths, or an
  `EvalData`; `report.to_dict()` is JSON-serializable.
- **CLI output modes**: `--json` for CI and tooling, `--plain` for ASCII-only
  output that is safe on Windows terminals, CI logs, and pipes.
- **Equivalence testing**: the statistical audit can now conclude two models are
  *statistically equivalent* within a configurable `--equivalence-margin`, instead
  of forcing every non-significant result to look like a failure.
- Proportion-appropriate statistics for pass/fail data: **McNemar's exact test**
  and **Cohen's h** / risk difference.
- **DeepEval adapter**: reads DeepEval's test-results export directly.
- Reports name the models that weren't compared when a file has more than two.

### Changed

- The Statistical Validity pillar now reports **decision / effect size /
  precision**. "Not significant" is reported as *inconclusive* or *equivalent*
  rather than a blunt failure.
- Replaced post-hoc (observed-effect) power with the prospective **minimum
  detectable effect**.
- Verdict summaries reflect the actual outcome (equivalent / inconclusive), not a
  generic "improvement is probably real".
- Documented the tool's assumptions and limitations, and made adapter-coverage
  claims precise.

## [0.1.0] — 2026-07-07

Initial release.

### Added

- `evaltrust audit` command: reads an evaluation results file and prints a
  High/Moderate/Low confidence verdict.
- Four audit pillars:
  - **Statistical Validity** — paired permutation test, bootstrap confidence
    interval, Cohen's *d*, and power analysis.
  - **Benchmark Health** — saturation and discrimination checks.
  - **Repeatability** — rerun stability and measurement noise (when the file
    contains repeated runs).
  - **Judge Reliability** — inter-judge consensus, agreement, and outlier
    detection (when the file contains multiple judges).
- Format auto-detection with adapters for Promptfoo, nested JSON, generic record
  lists, and CSV (long and wide).
- Two-file comparison for single-model tools (e.g. two DeepEval runs), paired by
  example id.
- `--strict` flag to fail CI on a Low-Confidence verdict.
- Deterministic, seeded resampling so audits are reproducible.

[Unreleased]: https://github.com/k-dickinson/evaltrust/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/k-dickinson/evaltrust/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/k-dickinson/evaltrust/releases/tag/v0.1.0
