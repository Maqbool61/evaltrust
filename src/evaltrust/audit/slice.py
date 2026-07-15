"""Per-slice / subgroup comparison audit.

Breaks the model-vs-model comparison down by a per-example attribute (category,
difficulty, language, ...) so an overall improvement can't hide a regression on
an important subset. Each slice's significance is tested at a Bonferroni-corrected
threshold across the number of slices, and any slice whose direction disagrees
with the overall verdict is flagged as a regression.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from ..core.schema import EvalData, Example, Finding, Status
from ..stats.paired import mcnemar_exact
from ..stats.resampling import permutation_test

PILLAR = "Per-slice Comparison"


def _slice_examples(
    data: EvalData, slice_by: str, model_a: str, model_b: str
) -> "OrderedDict[str, list[Example]]":
    groups: "OrderedDict[str, list[Example]]" = OrderedDict()
    for ex in data.examples:
        if model_a not in ex.scores or model_b not in ex.scores:
            continue
        if not ex.attributes:
            continue
        value = ex.attributes.get(slice_by)
        if value is None:
            continue
        groups.setdefault(str(value), []).append(ex)
    return groups


def _is_binary(examples: list[Example], model_a: str, model_b: str) -> bool:
    vals = []
    for ex in examples:
        for m in (model_a, model_b):
            if m in ex.scores:
                vals.append(ex.scores[m])
    return bool(vals) and set(np.unique(vals)).issubset({0.0, 1.0})


def _paired_diffs(
    examples: list[Example], model_a: str, model_b: str
) -> np.ndarray:
    diffs = [ex.scores[model_b] - ex.scores[model_a] for ex in examples]
    return np.asarray(diffs, dtype=float)


def _discordant_counts(
    examples: list[Example], leader: str, trailer: str
) -> tuple[int, int]:
    b_only = a_only = 0
    for ex in examples:
        lead, trail = ex.scores[leader], ex.scores[trailer]
        if lead == 1 and trail == 0:
            b_only += 1
        elif lead == 0 and trail == 1:
            a_only += 1
    return b_only, a_only


def _slice_pvalue(
    examples: list[Example], model_a: str, model_b: str,
    n_resamples: int, seed: int,
) -> tuple[float, float]:
    """Return the p-value and the signed mean of ``score_b - score_a`` for the slice."""
    diffs = _paired_diffs(examples, model_a, model_b)
    mean_diff = float(diffs.mean()) if diffs.size else 0.0
    if diffs.size < 2:
        return 1.0, mean_diff
    if _is_binary(examples, model_a, model_b):
        if mean_diff >= 0:
            leader, trailer = model_b, model_a
        else:
            leader, trailer = model_a, model_b
        b_only, a_only = _discordant_counts(examples, leader, trailer)
        return mcnemar_exact(b_only, a_only), mean_diff
    oriented = diffs if mean_diff >= 0 else -diffs
    return permutation_test(oriented, n_resamples=n_resamples, seed=seed), mean_diff


def audit_slices(
    data: EvalData,
    model_a: str,
    model_b: str,
    slice_by: str,
    alpha: float = 0.05,
    n_resamples: int = 10_000,
    seed: int = 0,
    min_slice_size: int = 5,
    *,
    overall_mean_diff: float | None = None,
) -> list[Finding]:
    """Compare ``model_a`` vs ``model_b`` per slice of ``slice_by``.

    Returns a single :class:`Finding` summarising the per-slice picture. Each
    slice's significance is tested at ``alpha / k`` (Bonferroni across ``k``
    slices) and flagged as a regression when its direction disagrees with the
    overall comparison. If ``overall_mean_diff`` is not supplied it is computed
    from the full paired sample.
    """
    groups = _slice_examples(data, slice_by, model_a, model_b)

    if not groups:
        return [Finding(
            pillar=PILLAR,
            title=f"No examples carry the {slice_by!r} attribute",
            status=Status.SKIP,
            why=(
                "Per-slice comparison needs a per-example attribute to break "
                "the audit down by. Without it, a subgroup regression can hide "
                "inside an overall improvement."
            ),
            how_detected=(f"No example in the paired sample carried an "
                          f"attribute named {slice_by!r}."),
            how_to_fix=(f"Tag examples with an {slice_by!r} value in the input "
                        "file (native adapter: attributes: {\"" + slice_by +
                        "\": \"...\"})."),
            details={"check": "slice_comparison", "slice_by": slice_by,
                     "assessed": False, "reason": "attribute_absent"},
        )]

    if overall_mean_diff is None:
        overall_mean_diff = float(data.differences(model_a, model_b).mean())
    overall_sign = _sign(overall_mean_diff)

    k = len(groups)
    corrected_alpha = alpha / k

    slice_details = []
    regressions: list[str] = []
    for value, examples in groups.items():
        n = len(examples)
        if n < min_slice_size:
            slice_details.append({
                "value": value, "n": n, "assessed": False,
                "reason": "too_few_examples",
            })
            continue
        p, mean_diff = _slice_pvalue(examples, model_a, model_b,
                                     n_resamples=n_resamples, seed=seed)
        significant = p < corrected_alpha
        slice_sign = _sign(mean_diff)
        regresses = (
            significant and overall_sign != 0 and slice_sign != 0
            and slice_sign != overall_sign
        )
        slice_details.append({
            "value": value, "n": n, "assessed": True,
            "p_value": p, "corrected_alpha": corrected_alpha,
            "mean_diff": mean_diff, "significant": significant,
            "regresses": regresses,
        })
        if regresses:
            regressions.append(value)

    if regressions:
        status = Status.WARN
        title = (f"{len(regressions)} of {k} slices regress against the "
                 "overall result")
        listed = ", ".join(repr(v) for v in regressions)
        how = (f"With Bonferroni across {k} slices (alpha/k = "
               f"{corrected_alpha:.4f}), slices {listed} are significant in the "
               "opposite direction to the overall comparison.")
        fix = ("Investigate the flagged slices before shipping: the aggregate "
               "verdict hides a regression on that subset.")
    else:
        status = Status.PASS
        title = "No slice regresses against the overall result"
        how = (f"Compared {k} slices at Bonferroni-corrected alpha/k = "
               f"{corrected_alpha:.4f}; none was significant in a direction "
               "opposite to the overall comparison.")
        fix = "The overall verdict holds across the reported slices."

    return [Finding(
        pillar=PILLAR, title=title, status=status,
        why=(
            "Results audited only in aggregate can hide a regression on an "
            "important subset (category, difficulty, language). A per-slice "
            "breakdown surfaces subgroups where the direction flips."
        ),
        how_detected=how, how_to_fix=fix,
        details={
            "check": "slice_comparison",
            "slice_by": slice_by,
            "assessed": True,
            "n_slices": k,
            "corrected_alpha": corrected_alpha,
            "overall_mean_diff": overall_mean_diff,
            "regressions": regressions,
            "slices": slice_details,
        },
    )]


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0
