# Example results files

Sample inputs you can run EvalTrust against to see each kind of verdict.

| File | What it shows |
|------|---------------|
| `clean_win.json` | A large, unambiguous improvement → **High Confidence**. |
| `borderline.json` | A small, noisy improvement with repeated runs and multiple judges → **Moderate Confidence**. |
| `deepeval_gpt4.json`, `deepeval_claude.json` | Two single-model runs to compare as a pair. |
| `basic.csv` | Long-form CSV with `id`, `model`, and `score` columns. |
| `basic.jsonl` | JSON Lines input with one score record per line. |
| `multi_metric.csv` | A CSV suite containing separate accuracy and safety metrics. |

Try them:

```bash
evaltrust audit examples/clean_win.json
evaltrust audit examples/borderline.json
evaltrust audit examples/deepeval_gpt4.json examples/deepeval_claude.json
evaltrust audit examples/basic.csv
evaltrust audit examples/basic.jsonl
evaltrust audit examples/multi_metric.csv
```
