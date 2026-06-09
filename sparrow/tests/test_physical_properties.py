"""Tests for sparrow.sequence_analysis.physical_properties."""

import pytest

from sparrow import Protein
from sparrow.data import amino_acids
from sparrow.sequence_analysis import physical_properties as pp


def test_single_residue_weight_is_free_amino_acid():
    assert pp.calculate_molecular_weight("G") == pytest.approx(
        amino_acids.AA_MOLECULAR_WEIGHT["G"]
    )


def test_dipeptide_loses_one_water():
    g = amino_acids.AA_MOLECULAR_WEIGHT["G"]
    assert pp.calculate_molecular_weight("GG") == pytest.approx(2 * g - 18)


def test_tripeptide_loses_two_waters():
    seq = "GAS"
    expected = sum(amino_acids.AA_MOLECULAR_WEIGHT[r] for r in seq) - 18 * 2
    assert pp.calculate_molecular_weight(seq) == pytest.approx(expected)


def test_protein_molecular_weight_property_matches_function():
    seq = "MEEEKKKKSSSTTTDDD"
    assert Protein(seq).molecular_weight == pytest.approx(
        pp.calculate_molecular_weight(seq)
    )
