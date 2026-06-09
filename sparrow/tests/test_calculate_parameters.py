"""Tests for sparrow.calculate_parameters."""


import numpy as np
import pytest

from sparrow import calculate_parameters as cp
from sparrow.data import amino_acids
from sparrow.sparrow_exceptions import CalculationException


# -- calculate_aa_fractions ------------------------------------------------

def test_aa_fractions_sum_to_one():
    fr = cp.calculate_aa_fractions("ACDEFGHIKLMNPQRSTVWYAA")
    assert sum(fr.values()) == pytest.approx(1.0)


def test_aa_fractions_known_values():
    fr = cp.calculate_aa_fractions("ACAA")
    assert fr["A"] == pytest.approx(0.75)
    assert fr["C"] == pytest.approx(0.25)
    assert fr["D"] == 0.0


def test_aa_fractions_has_twenty_keys():
    assert set(cp.calculate_aa_fractions("AAA")) == set(amino_acids.VALID_AMINO_ACIDS)


# -- calculate_seg_complexity ----------------------------------------------

def test_seg_complexity_homopolymer_is_zero():
    assert cp.calculate_seg_complexity("AAAAAAAA") == pytest.approx(0.0)


def test_seg_complexity_positive_for_diverse():
    assert cp.calculate_seg_complexity("ACDEFGHIKL") > 0.0


def test_seg_complexity_increases_with_diversity():
    low = cp.calculate_seg_complexity("AAAAAAAC")
    high = cp.calculate_seg_complexity("ACDEFGHI")
    assert high > low


# -- hydrophobicity --------------------------------------------------------

def test_linear_hydrophobicity_known_kd_values():
    # values from the shipped KD table
    vals = cp.calculate_linear_hydrophobicity("ACI")
    assert vals == [
        amino_acids.AA_hydro_KD["A"],
        amino_acids.AA_hydro_KD["C"],
        amino_acids.AA_hydro_KD["I"],
    ]


def test_linear_hydrophobicity_normalized_range():
    vals = cp.calculate_linear_hydrophobicity("ACDEFGHIKLMNPQRSTVWY", normalize=True)
    assert min(vals) >= 0.0 and max(vals) <= 1.0


def test_mean_hydrophobicity_matches_mean_of_linear():
    seq = "ACDEFGHIKL"
    assert cp.calculate_hydrophobicity(seq) == pytest.approx(
        np.mean(cp.calculate_linear_hydrophobicity(seq))
    )


def test_hydrophobicity_invalid_residue_raises():
    with pytest.raises(CalculationException):
        cp.calculate_linear_hydrophobicity("ACBX")


def test_hydrophobicity_invalid_mode_raises():
    with pytest.raises(CalculationException):
        cp.calculate_linear_hydrophobicity("ACDE", mode="nope")
