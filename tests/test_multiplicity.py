"""Tests for the multiplicity correction primitive.

``holm_bonferroni`` is validated against the reference implementation,
``statsmodels.stats.multitest.multipletests(method="holm")`` — never against
itself — on random p-vectors, ties, and every boundary the audit can hit.
"""

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from evaltrust.stats.multiplicity import holm_bonferroni


def _reference(pvals, alpha):
    reject, adjusted, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    return reject.tolist(), adjusted.tolist()


def test_matches_statsmodels_on_random_pvectors():
    # statsmodels.multipletests has notable per-call overhead, so keep the case
    # count modest; the ties / k=1 / boundary tests below cover the corners.
    rng = np.random.default_rng(0)
    for _ in range(15):
        k = int(rng.integers(1, 12))
        pvals = rng.uniform(0.0, 1.0, size=k).tolist()
        for alpha in (0.01, 0.05, 0.1):
            rejected, adjusted = holm_bonferroni(pvals, alpha)
            ref_rejected, ref_adjusted = _reference(pvals, alpha)
            assert rejected == ref_rejected
            assert adjusted == pytest.approx(ref_adjusted)


def test_matches_statsmodels_with_ties():
    pvals = [0.02, 0.02, 0.02, 0.5, 0.5]
    for alpha in (0.05, 0.1):
        rejected, adjusted = holm_bonferroni(pvals, alpha)
        ref_rejected, ref_adjusted = _reference(pvals, alpha)
        assert rejected == ref_rejected
        assert adjusted == pytest.approx(ref_adjusted)


def test_k1_matches_statsmodels():
    for p in (0.001, 0.049, 0.05, 0.2):
        rejected, adjusted = holm_bonferroni([p], 0.05)
        ref_rejected, ref_adjusted = _reference([p], 0.05)
        assert rejected == ref_rejected
        assert adjusted == pytest.approx(ref_adjusted)


def test_all_significant():
    pvals = [0.0001, 0.0002, 0.0003]
    rejected, adjusted = holm_bonferroni(pvals, 0.05)
    assert rejected == [True, True, True]
    assert rejected == _reference(pvals, 0.05)[0]


def test_none_significant():
    pvals = [0.4, 0.6, 0.9]
    rejected, adjusted = holm_bonferroni(pvals, 0.05)
    assert rejected == [False, False, False]
    assert rejected == _reference(pvals, 0.05)[0]


def test_p_equals_alpha_exactly():
    # At k=1 the Holm threshold is alpha itself; statsmodels rejects at p == alpha
    # (uses <=). Our function must agree.
    rejected, adjusted = holm_bonferroni([0.05], 0.05)
    assert rejected == [True]
    assert rejected == _reference([0.05], 0.05)[0]
    # And within a vector, a p landing exactly on its step threshold.
    pvals = [0.01, 0.05, 0.9]  # second step threshold at k=3 is 0.05/2 = 0.025
    rejected, adjusted = holm_bonferroni(pvals, 0.05)
    assert rejected == _reference(pvals, 0.05)[0]
    assert adjusted == pytest.approx(_reference(pvals, 0.05)[1])


def test_rejected_set_is_a_prefix_of_the_sorted_pvalues():
    # Holm is step-down: the rejected hypotheses are exactly the smallest
    # n_rejected p-values, in ANY input order.
    pvals = [0.5, 0.001, 0.2, 0.01, 0.9]
    rejected, _ = holm_bonferroni(pvals, 0.05)
    order = np.argsort(pvals, kind="stable")
    n_rejected = sum(rejected)
    # The rejected originals are precisely the n_rejected smallest.
    expected = [i in set(order[:n_rejected].tolist()) for i in range(len(pvals))]
    assert rejected == expected


def test_empty_input():
    assert holm_bonferroni([], 0.05) == ([], [])


def test_returns_plain_python_types():
    rejected, adjusted = holm_bonferroni([0.01, 0.2], 0.05)
    assert all(isinstance(r, bool) for r in rejected)
    assert all(isinstance(a, float) for a in adjusted)
