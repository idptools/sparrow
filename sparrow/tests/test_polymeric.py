"""Tests for sparrow.polymer.Polymeric.

The SAW and nu-dependent-SAW distribution accessors are regression-pinned here
because they were broken for every input: they called ``end_to_end_distribution``
on the afrc model objects, which expose ``get_end_to_end_distribution``. Both
raised ``AttributeError`` on any sequence. This module was empty, which is why
that went unnoticed - so these assert real output rather than merely not raising.
"""

import numpy as np
import pytest

from sparrow import Protein

SEQ = "MEEEKKKKSSSTTTDDDQQQQNNNNGGGSSSAAAPPPLLL"


@pytest.fixture(scope="module")
def polymeric():
    return Protein(SEQ).polymeric


def _distribution_mean(result):
    """Probability-weighted mean of a (distance, probability) pair.

    The cache-key tests compare this rather than the probability arrays directly:
    the distance grid afrc returns is not prefactor-independent, so two calls can
    legitimately produce different-length arrays and an elementwise comparison
    raises instead of reporting a difference. A scalar summary is both
    shape-independent and closer to what the test actually means.
    """
    r, prob = np.asarray(result[0]), np.asarray(result[1])
    return float(np.sum(r * prob) / np.sum(prob))


def _assert_is_distribution(result):
    """A (distance, probability) pair of equal-length, finite, non-negative arrays."""
    r, p = result
    r, p = np.asarray(r), np.asarray(p)
    assert r.ndim == p.ndim == 1
    assert r.shape == p.shape
    assert r.size > 0
    assert np.all(np.isfinite(r)) and np.all(np.isfinite(p))
    assert np.all(p >= 0)
    assert p.sum() > 0
    # distances must be increasing, so the pair is a usable distribution
    assert np.all(np.diff(r) > 0)


def test_saw_end_to_end_distribution(polymeric):
    _assert_is_distribution(polymeric.get_saw_end_to_end_distribution())


def test_nudep_saw_end_to_end_distribution(polymeric):
    _assert_is_distribution(polymeric.get_nudep_saw_end_to_end_distribution())


def test_saw_prefactor_is_part_of_the_cache_key(polymeric):
    """Two prefactors must not collide in the memoization dict."""
    mean_5 = _distribution_mean(polymeric.get_saw_end_to_end_distribution(prefactor=5.5))
    mean_8 = _distribution_mean(polymeric.get_saw_end_to_end_distribution(prefactor=8.0))
    assert not np.isclose(mean_5, mean_8)


def test_nudep_saw_nu_is_part_of_the_cache_key(polymeric):
    """Likewise for nu, which is the whole point of the nu-dependent model."""
    mean_low = _distribution_mean(polymeric.get_nudep_saw_end_to_end_distribution(nu=0.40))
    mean_high = _distribution_mean(polymeric.get_nudep_saw_end_to_end_distribution(nu=0.60))
    assert not np.isclose(mean_low, mean_high)


def test_afrc_distributions_still_work(polymeric):
    """The afrc-backed siblings share the accessor pattern; pin them too."""
    _assert_is_distribution(polymeric.get_afrc_end_to_end_distribution())
    _assert_is_distribution(polymeric.get_afrc_radius_of_gyration_distribution())
