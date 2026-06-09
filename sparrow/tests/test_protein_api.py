"""Comprehensive tests for Protein methods, properties, caching, and errors."""

import pytest

from sparrow import Protein
from sparrow.calculate_parameters import calculate_seg_complexity
from sparrow.patterning import kappa as kappa_module
from sparrow.sequence_analysis import physical_properties
from sparrow.sparrow_exceptions import ProteinException, SparrowException

SEQ = "MEEEKKKKSSSTTTDDDQQQQNNNNGGGGRRRRDDDD"


@pytest.fixture
def protein():
    return Protein(SEQ)


# -- scalar properties match their functional implementations --------------

def test_molecular_weight_matches_function(protein):
    assert protein.molecular_weight == pytest.approx(
        physical_properties.calculate_molecular_weight(SEQ)
    )


def test_complexity_matches_function(protein):
    assert protein.complexity == pytest.approx(calculate_seg_complexity(SEQ))


def test_ncpr_is_pos_minus_neg(protein):
    assert protein.NCPR == pytest.approx(
        protein.fraction_positive - protein.fraction_negative
    )


def test_fcr_is_sum_of_charged_fractions(protein):
    expected = sum(protein.amino_acid_fractions[r] for r in "EDRK")
    assert protein.FCR == pytest.approx(expected)


def test_fraction_proline_matches_dict(protein):
    assert protein.fraction_proline == protein.amino_acid_fractions["P"]


# -- kappa family ----------------------------------------------------------

def test_kappa_in_unit_range(protein):
    assert 0.0 <= protein.kappa <= 1.0


def test_kappa_default_matches_window_average(protein):
    k5 = kappa_module.kappa_x(SEQ, ["R", "K"], ["E", "D"], 5, 1)
    k6 = kappa_module.kappa_x(SEQ, ["R", "K"], ["E", "D"], 6, 1)
    assert protein.kappa == pytest.approx((k5 + k6) / 2)


def test_kappa_minus_one_for_short_sequence():
    assert Protein("EK").kappa == -1


def test_compute_kappa_x_group_order_normalized(protein):
    assert protein.compute_kappa_x("ED", "RK", 6) == protein.compute_kappa_x("DE", "KR", 6)


def test_compute_kappa_x_invalid_residue_raises(protein):
    with pytest.raises(ProteinException):
        protein.compute_kappa_x("EZ", "RK")


# -- IWD -------------------------------------------------------------------

def test_compute_iwd_accepts_string_or_list():
    p = Protein("ILVAMILVAMGGGGILVAM")
    assert p.compute_iwd("ILVAM") == p.compute_iwd(["I", "L", "V", "A", "M"])


def test_compute_iwd_charged_weighted_invalid_charge_raises(protein):
    with pytest.raises(ProteinException):
        protein.compute_iwd_charged_weighted(charge="x")


def test_compute_iwd_charged_weighted_signs(protein):
    assert protein.compute_iwd_charged_weighted("-") >= 0
    assert protein.compute_iwd_charged_weighted("+") >= 0


# -- caching ---------------------------------------------------------------

def test_amino_acid_fractions_cached(protein):
    assert protein.amino_acid_fractions is protein.amino_acid_fractions


def test_linear_profile_cached(protein):
    a = protein.linear_sequence_profile("NCPR", window_size=6)
    b = protein.linear_sequence_profile("NCPR", window_size=6)
    assert a is b


def test_accessors_are_reused(protein):
    assert protein.predictor is protein.predictor
    assert protein.polymeric is protein.polymeric
    assert protein.plugin is protein.plugin


# -- validation / normalization --------------------------------------------

def test_sequence_is_uppercased():
    assert Protein("acdef").sequence == "ACDEF"


def test_validation_converts_nonstandard_residues():
    # B->N, U->C, X->G, Z->Q
    assert Protein("MBUXZ", validate=True).sequence == "MNCGQ"


def test_validation_strips_stars_and_dashes():
    assert Protein("MA*-K", validate=True).sequence == "MAK"


def test_validation_unfixable_raises():
    # a digit cannot be converted to a valid residue
    with pytest.raises(SparrowException):
        Protein("M1234", validate=True)


# -- dunders ---------------------------------------------------------------

def test_len(protein):
    assert len(protein) == len(SEQ)


def test_repr_contains_length_and_prefix(protein):
    r = repr(protein)
    assert str(len(SEQ)) in r
    assert SEQ[:5] in r
