"""Tests for sparrow.polymer.Polymeric.

The SAW and nu-dependent-SAW distribution accessors are regression-pinned here
because they were broken for every input: they called ``end_to_end_distribution``
on the afrc model objects, which expose ``get_end_to_end_distribution``. Both
raised ``AttributeError`` on any sequence. This module was empty, which is why
that went unnoticed - so these assert real output rather than merely not raising.
"""

import warnings

import numpy as np
import pytest

from sparrow import Protein
from sparrow.polymer import analytical_frc, nudep_saw, saw, zheng2020

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
    _assert_is_distribution(polymeric.saw_end_to_end_distribution())


def test_nudep_saw_end_to_end_distribution(polymeric):
    _assert_is_distribution(polymeric.nudep_saw_end_to_end_distribution())


def test_saw_prefactor_is_part_of_the_cache_key(polymeric):
    """Two prefactors must not collide in the memoization dict."""
    mean_5 = _distribution_mean(polymeric.saw_end_to_end_distribution(prefactor=5.5))
    mean_8 = _distribution_mean(polymeric.saw_end_to_end_distribution(prefactor=8.0))
    assert not np.isclose(mean_5, mean_8)


def test_nudep_saw_nu_is_part_of_the_cache_key(polymeric):
    """Likewise for nu, which is the whole point of the nu-dependent model."""
    mean_low = _distribution_mean(polymeric.nudep_saw_end_to_end_distribution(nu=0.40))
    mean_high = _distribution_mean(polymeric.nudep_saw_end_to_end_distribution(nu=0.60))
    assert not np.isclose(mean_low, mean_high)


def test_afrc_distributions_still_work(polymeric):
    """The afrc-backed siblings share the accessor pattern; pin them too."""
    _assert_is_distribution(polymeric.afrc_end_to_end_distribution())
    _assert_is_distribution(polymeric.afrc_radius_of_gyration_distribution())


# afrc's fixed grid spacing, in Angstroms
AFRC_GRID_SPACING_A = 0.05


@pytest.mark.parametrize("length", [40, 400])
def test_afrc_distributions_use_the_fixed_grid(length):
    """The AFRC distributions are on afrc's fixed 0.05 A grid at any length.

    sparrow used to pass its bin width as AnalyticalFRC's second argument, which
    is adaptable_P_res, so the grid was 3.7N/500 A instead - 0.3 A at 40 residues
    and 3 A at 400.
    """
    polymeric = Protein((SEQ * 10)[:length]).polymeric
    for distances, _ in (
        polymeric.afrc_end_to_end_distribution(),
        polymeric.afrc_radius_of_gyration_distribution(),
    ):
        np.testing.assert_allclose(np.diff(distances), AFRC_GRID_SPACING_A)


def test_each_model_module_gives_what_polymeric_gives(polymeric):
    """Polymeric calls the model modules; a direct call gives the same values."""
    assert analytical_frc.mean_radius_of_gyration(SEQ) == polymeric.mean_afrc_radius_of_gyration()
    assert analytical_frc.mean_end_to_end_distance(SEQ, "distribution") == polymeric.mean_afrc_end_to_end_distance(
        mode="distribution"
    )
    np.testing.assert_array_equal(analytical_frc.internal_scaling(SEQ), polymeric.afrc_internal_scaling())
    assert saw.mean_radius_of_gyration(SEQ, prefactor=6.0) == polymeric.mean_saw_radius_of_gyration(prefactor=6.0)
    assert nudep_saw.mean_end_to_end_distance(SEQ, nu=0.45) == polymeric.mean_nudep_saw_end_to_end_distance(nu=0.45)
    assert zheng2020.scaling_exponent(SEQ) == polymeric.empirical_nu()
    assert zheng2020.radius_of_gyration(SEQ) == polymeric.empirical_radius_of_gyration()


def test_a_bin_width_reaches_the_saw_models():
    sequence_length_only = "A" * 100
    fine = Protein(sequence_length_only).polymeric.saw_end_to_end_distribution()
    coarse = saw.end_to_end_distribution(sequence_length_only, p_of_r_resolution=0.5)
    np.testing.assert_allclose(np.diff(coarse[0]), 0.5)
    assert len(coarse[0]) < len(fine[0])


@pytest.mark.parametrize(
    ("old_name", "new_name"),
    [
        ("get_mean_afrc_radius_of_gyration", "mean_afrc_radius_of_gyration"),
        ("get_saw_end_to_end_distribution", "saw_end_to_end_distribution"),
        ("get_mean_nudep_saw_radius_of_gyration", "mean_nudep_saw_radius_of_gyration"),
        ("get_predicted_nu_dep_end_to_end_distribution", "predicted_nu_dep_end_to_end_distribution"),
    ],
)
def test_an_old_get_name_still_works_with_a_deprecation_warning(polymeric, old_name, new_name):
    with pytest.warns(
        DeprecationWarning, match=f"Polymeric.{old_name} has been renamed Polymeric.{new_name}"
    ) as caught:
        old_method = getattr(polymeric, old_name)
    # the warning points at the line that used the old name, not inside sparrow
    assert caught[0].filename == __file__
    assert old_method == getattr(polymeric, new_name)


def test_an_unknown_attribute_still_raises(polymeric):
    with pytest.raises(AttributeError, match="get_nothing"):
        polymeric.get_nothing  # noqa: B018


def test_the_new_names_do_not_warn(polymeric):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        polymeric.mean_afrc_radius_of_gyration()


def test_the_old_scaling_parameters_names_still_work():
    from sparrow.polymer import scaling_parameters

    with pytest.warns(DeprecationWarning, match="zheng2020.scaling_exponent"):
        assert scaling_parameters.compute_nu_zheng2020(SEQ) == zheng2020.scaling_exponent(SEQ)
    with pytest.warns(DeprecationWarning, match="zheng2020.radius_of_gyration"):
        assert scaling_parameters.compute_rg_zheng2020(SEQ) == zheng2020.radius_of_gyration(SEQ)
