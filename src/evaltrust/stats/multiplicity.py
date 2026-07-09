"""Multiplicity corrections for testing several hypotheses at once.

Testing ``k`` metrics at level ``alpha`` inflates the family-wise false-positive
rate: run 20 independent metrics at 0.05 and one will look "significant" by luck.
Bonferroni (``alpha / k`` for every metric) is the blunt fix. Holm-Bonferroni is
a *step-down* refinement that controls the same family-wise error rate while
rejecting at least as many hypotheses — it is uniformly more powerful than
Bonferroni and never rejects fewer.

This module is pure numbers: it maps p-values to reject/keep decisions and knows
nothing about findings or formatting.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def holm_bonferroni(
    pvalues: Sequence[float], alpha: float
) -> tuple[list[bool], list[float]]:
    """Holm-Bonferroni step-down correction.

    Orders the p-values ascending and compares the ``i``-th smallest (0-indexed)
    against ``alpha / (k - i)``, rejecting from the smallest p-value upward and
    stopping at the first that fails — so once a hypothesis is retained, every
    larger one is too. The rejected set is therefore always the block of smallest
    p-values.

    Returns, in the ORIGINAL input order:

    - ``rejected``: whether each hypothesis is rejected at family-wise level
      ``alpha``;
    - ``adjusted_p``: the Holm-adjusted p-values, made monotone across the
      step-down and clipped to ``1.0``, so ``adjusted_p[i] <= alpha`` exactly
      when hypothesis ``i`` is rejected.

    Matches ``statsmodels.stats.multitest.multipletests(pvals, alpha,
    method="holm")`` for both returned arrays, including ties, ``k == 1``, and a
    p-value landing exactly on its threshold.
    """
    p = np.asarray(pvalues, dtype=float)
    k = p.size
    if k == 0:
        return [], []

    order = np.argsort(p, kind="stable")           # indices of ascending p
    sorted_p = p[order]
    multipliers = k - np.arange(k)                 # k, k-1, ..., 1

    # Holm-adjusted p-values: (k - i) * p_(i), made monotone non-decreasing so a
    # later, larger hypothesis can never be "more significant" than an earlier
    # one, then clipped into [0, 1]. Rejection is adjusted_p <= alpha.
    adjusted_sorted = np.maximum.accumulate(multipliers * sorted_p)
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    rejected_sorted = adjusted_sorted <= alpha

    rejected = np.empty(k, dtype=bool)
    adjusted = np.empty(k, dtype=float)
    rejected[order] = rejected_sorted
    adjusted[order] = adjusted_sorted

    return [bool(r) for r in rejected], [float(a) for a in adjusted]
