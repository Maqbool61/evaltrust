<h1 align="center">EvalTrust</h1>

<p align="center">
  <strong>An auditor for LLM evaluations.</strong><br>
  It doesn't tell you how good your model is — it tells you whether you can trust the evaluation you used to decide.
</p>

<p align="center">
  <a href="#installation">Install</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#what-it-checks">What it checks</a> ·
  <a href="docs/">Docs</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/evaltrust/"><img alt="PyPI" src="https://img.shields.io/pypi/v/evaltrust.svg"></a>
  <a href="https://pypi.org/project/evaltrust/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/evaltrust.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/pypi/l/evaltrust.svg"></a>
  <a href="https://github.com/k-dickinson/evaltrust/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/k-dickinson/evaltrust/actions/workflows/ci.yml/badge.svg"></a>
</p>

---

## The problem, in ten seconds

You spent a week evaluating two models. One scored 84.7%, the other 86.2%. Ship
the better one, right?

Maybe. Or maybe that 1.5-point gap is pure luck and the two models are actually
the same. **You can't tell by looking, and neither can anyone else.** People ship
models, publish benchmark numbers, and kill good ideas every day based on gaps
that are just noise.

EvalTrust does the math that tells you. Point it at your eval results and it says,
in plain English, whether the difference is **real**, whether it's **big enough to
matter**, and whether you had **enough data to know** — then gives you one verdict:
**High, Moderate, or Low confidence.** You keep using whatever eval tool you
already have. Think of it as a code reviewer for your eval's conclusion.

## Why you can't just eyeball it

Think about a coin. Flip it 10 times, get 6 heads. Rigged? No — that happens by
luck constantly. Now flip it 1,000 times and get 600 heads. *That's* real. Six
out of ten and six hundred out of a thousand look the same on the surface, but
only one is signal.

"84.7 vs 86.2" is the exact same trap. EvalTrust computes how likely that gap is
to be a lucky streak. If it's likely luck, it tells you straight: don't ship on
this yet.

## Example

```console
$ evaltrust audit gpt4_run.json claude_run.json
```

```
EvalTrust  claude-3 vs gpt-4 · 150 examples · deepeval

● Low Confidence
Not enough data to call a winner. Collect more before deciding.

Statistical Validity
  ✗ Improvement of claude-3 over gpt-4 is inconclusive
  ! Effect size is negligible
  ! Sample size may be too small
Benchmark Health
  ✓ Benchmark has headroom
  ✓ Benchmark discriminates between examples

What to do
  • Don't call a winner yet. Collect more examples first.
  • Collect ~90 more examples (~240 total) to catch a small effect.
```

Two runs at 71% and 74% — a three-point "win" that is actually noise. EvalTrust
catches it before it becomes a shipping decision. (Add `--explain` for the exact
p-values and reasoning behind each line.)

## Why the numbers are real, not our opinion

Fair question for a tool that judges trust. Three things keep it honest:

- **It runs standard, decades-old statistics** — the same significance tests,
  confidence intervals, and power analysis scientists already use. Nothing invented.
- **The math is proven correct.** Every calculation is checked in the test suite
  against the libraries researchers trust (`scipy`, `statsmodels`) and must produce
  the same numbers. So it's not "trust us" — it's "our math matches the reference
  everyone already trusts."
- **It's reproducible.** Same input, same answer, every time. An opinion drifts; a
  calculation doesn't.

One honest limit: EvalTrust checks whether your *numbers* support your conclusion.
If the eval measured the wrong thing to begin with, it catches some of that
(saturated benchmarks, judges that disagree) but not all of it. It's the
statistician auditing your results, not a replacement for a well-designed eval.

## Installation

```bash
pip install evaltrust
```

Requires Python 3.10 or newer. That's the whole setup — no API keys, no config,
no account.

<details>
<summary>Install from source (for development)</summary>

```bash
git clone https://github.com/k-dickinson/evaltrust
cd evaltrust
pip install -e ".[dev]"
pytest
```

</details>

## Quick start

1. Run your evaluation with whatever tool you already use (Promptfoo, DeepEval, or
   anything that can export scores to CSV/JSON) and save the output.
2. Point EvalTrust at it:

   ```bash
   # A file that already compares two or more models (e.g. Promptfoo):
   evaltrust audit results.json

   # Two single-model runs (e.g. two DeepEval runs), paired by example id:
   evaltrust audit gpt4_run.json claude_run.json
   ```

3. Read the verdict. Fix what it flags. Re-run.

Want to see it work before pointing it at your own data? The repo ships sample
files:

```bash
evaltrust audit examples/clean_win.json        # -> High Confidence
evaltrust audit examples/borderline.json       # -> Moderate Confidence
evaltrust audit examples/deepeval_gpt4.json examples/deepeval_claude.json
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--json` | Emit the full audit as JSON, for CI logic and experiment trackers. |
| `--plain` | Plain ASCII output — safe for Windows terminals, CI logs, and piping to a file. |
| `--strict` | Exit with a non-zero status on a Low-Confidence verdict (use it to gate CI). |
| `--model-a`, `--model-b` | Choose which two models to compare, or label the two files. |
| `--alpha` | Significance level (default `0.05`). |
| `--seed` | Seed for the resampling (results are deterministic; change only to stress-test). |

## Use it from Python

The CLI is a thin wrapper over a small API, so you can audit inside a notebook,
a training script, or a CI job:

```python
import evaltrust

report = evaltrust.audit("results.json")           # path, two paths, or an EvalData
print(report.verdict.level)                         # VerdictLevel.HIGH / MODERATE / LOW

if report.verdict.level is evaltrust.VerdictLevel.LOW:
    raise SystemExit("Evaluation isn't trustworthy enough to ship on.")

report.to_dict()          # machine-readable, JSON-serializable — log it, store it, diff it
```

## Gate CI on it

`--strict` returns a non-zero exit code on a Low-Confidence verdict, so an audit
can block a merge the way failing tests do:

```yaml
# .github/workflows/eval.yml
- name: Audit the evaluation
  run: |
    pip install evaltrust
    evaltrust audit results.json --strict --plain
```

## What it checks

EvalTrust audits four pillars of trust and ends in one plain-language verdict —
**High**, **Moderate**, or **Low Confidence**. There is no arbitrary aggregate
score.

| Pillar | The question it answers |
|--------|-------------------------|
| **Statistical Validity** | Is the difference a real improvement, no real difference, or just too little data to tell? Significance (McNemar for pass/fail, paired permutation for continuous), equivalence testing, an interpretable effect size, and the minimum detectable effect. |
| **Benchmark Health** | Can the benchmark even separate these models, or is it saturated / flat? |
| **Repeatability** | If you reran the evaluation, would the winner stay the winner? Uses repeated-run data when the file contains it. |
| **Judge Reliability** | Would a different judge reach the same verdict? Uses multi-judge data when the file contains it. |

Every finding follows the same rule — **why it matters**, **how we detected it**,
and **how to fix it**. Checks that need extra data (repeated runs, multiple
judges) don't guess when it's missing; they tell you how to generate it.

Scoring several metrics per example (correctness, safety, tone…)? Add a `metric`
column and EvalTrust audits each one, corrects the significance threshold for the
number of metrics (so you don't get false wins by testing many), and reports the
suite's confidence as its weakest metric.

See [`docs/checks.md`](docs/checks.md) for the methods and thresholds behind each
one.

## Supported inputs

You never write an EvalTrust-specific format. It reads what your tool already
produced and auto-detects the shape. First-class adapters today:

- **Promptfoo** results (several providers compared across test cases)
- **DeepEval** test-results export (one model per run — pass two files to compare)
- **Nested JSON** — `{"models": [...], "examples": [{"id", "scores": {...}}]}`
- **Record lists** — JSON like `[{"id", "model", "score"}, ...]`
- **CSV** — long (`id,model,score`) or wide (`id,gpt,claude`)

Tools without a dedicated adapter yet (LangSmith, OpenEvals, and others) work by
exporting to CSV or a record list — usually a one-liner. More native adapters are
a top roadmap item; [contributing one](docs/adapters.md) is straightforward.
Details and single-model pairing in [`docs/input-formats.md`](docs/input-formats.md).

## How it works

```
your eval output ──▶ auto-detect + adapter ──▶ canonical model ──▶ audit ──▶ verdict
```

Adapters map every format into one internal representation, so the statistics are
written once and work everywhere. Every statistical method is validated in the
test suite against an independent reference (`scipy` and `statsmodels`), and all
resampling is seeded, so the auditor is itself reproducible. See
[`docs/architecture.md`](docs/architecture.md).

## Roadmap

- **Now:** offline CLI **and Python API**, four pillars, terminal / JSON / plain
  output, equivalence testing, **multi-metric suites** (with multiple-comparison
  correction), adapters for Promptfoo, DeepEval, JSON, and CSV.
- **Next:** native adapters for hosted platforms (LangSmith, Braintrust, ...), an
  optional HTML report, and historical/regression tracking across runs.
- **Later:** opt-in orchestration for the pillars that need to generate evidence
  (robustness perturbations, extra judges) and a provenance/reproducibility check.

## Contributing

Contributions are welcome — new format adapters and additional checks especially.
Start with [`CONTRIBUTING.md`](CONTRIBUTING.md). All participants are expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md). Report security issues per the
[security policy](SECURITY.md).

## License

EvalTrust is released under the [MIT License](LICENSE) — a permissive,
OSI-approved license. Anyone, including companies and organizations, may use,
modify, and distribute it, in commercial or proprietary settings, free of charge.
There is no copyleft obligation and no contributor license agreement to sign.
