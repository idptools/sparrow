"""Tests for sparrow.properties.composition."""

import numpy as np
import pytest

from sparrow import Protein
from sparrow.data import amino_acids
from sparrow.properties import composition, encoding
from sparrow.sparrow_exceptions import SparrowException

ALL_TWENTY = "ACDEFGHIKLMNPQRSTVWY"


def column(residue):
    return encoding.RESIDUES.index(residue)


# -- counting ------------------------------------------------------------------


def test_count_residues_counts_every_residue_of_every_sequence():
    counts = composition.count_residues(["ACAA", "", "YY"])
    assert counts.lengths.tolist() == [4, 0, 2]
    assert counts.counts[0, column("A")] == 3
    assert counts.counts[0, column("C")] == 1
    assert counts.counts[1].sum() == 0
    assert counts.counts[2, column("Y")] == 2
    assert counts.counts.sum(axis=1).tolist() == counts.lengths.tolist()


def test_count_residues_upper_cases():
    assert np.array_equal(composition.count_residues("meek").counts, composition.count_residues("MEEK").counts)


def test_count_residues_of_an_empty_batch():
    counts = composition.count_residues([])
    assert counts.counts.shape == (0, encoding.NUM_RESIDUES)
    assert len(counts) == 0


def test_count_residues_says_which_sequence_is_bad():
    with pytest.raises(SparrowException, match="Sequence 1 of 3"):
        composition.count_residues(["MEEK", "MEXK", "GS"])


def test_count_residues_rejects_non_strings():
    with pytest.raises(SparrowException, match="item 1"):
        composition.count_residues(["MEEK", 7])


def test_counts_can_be_passed_in_place_of_sequences():
    sequences = ["MEEEKKKKSSS", "GSGSDDDE"]
    counts = composition.count_residues(sequences)
    assert np.array_equal(composition.fraction_charged(counts), composition.fraction_charged(sequences))


# -- fractions -------------------------------------------------------------------


def test_amino_acid_fractions_sum_to_one():
    assert composition.amino_acid_fractions(ALL_TWENTY + "AA")[0].sum() == pytest.approx(1.0)


def test_amino_acid_fractions_known_values():
    fractions = composition.amino_acid_fractions("ACAA")[0]
    assert fractions[column("A")] == 0.75
    assert fractions[column("C")] == 0.25
    assert fractions[column("D")] == 0.0


def test_fraction_in_set_ignores_order_and_repeats():
    """A fraction is one integer count over the length, whatever order the set is in."""
    sequence = "MFWYYGSWFFEK" * 7
    reference = composition.fraction_in_set(sequence, "FWY")
    for residues in ("YWF", "WFY", "FFWY", ["Y", "W", "F"]):
        assert np.array_equal(composition.fraction_in_set(sequence, residues), reference)


def test_fraction_in_set_rejects_a_non_standard_residue():
    with pytest.raises(SparrowException, match="'X'"):
        composition.fraction_in_set("MEEK", "EX")


def test_named_fractions_use_the_documented_sets():
    sequence = ALL_TWENTY * 2
    for function, residues in [
        (composition.fraction_positive, amino_acids.POS),
        (composition.fraction_negative, amino_acids.NEG),
        (composition.fraction_charged, amino_acids.CHARGE),
        (composition.fraction_aromatic, amino_acids.ARO),
        (composition.fraction_aliphatic, amino_acids.ALI),
        (composition.fraction_polar, amino_acids.POLAR),
        (composition.fraction_chain_expanding, amino_acids.EXPANDING),
        (composition.fraction_disorder_promoting, amino_acids.DISORDER_PROMOTING),
    ]:
        assert function(sequence)[0] == len(residues) / len(ALL_TWENTY)


def test_ncpr_of_a_charge_balanced_sequence_is_exactly_zero():
    """Positive minus negative fraction used to leave a 1e-16 residue here."""
    for sequence in ["RKKEEDAAAAAAAAAAAAAAAAAAAAA", "RRREEDDDKKAAAAAAAAAAAAAAAAA" * 3, "KEDRGGGGGGG"]:
        positive = sum(sequence.count(residue) for residue in "RK")
        negative = sum(sequence.count(residue) for residue in "ED")
        assert positive == negative
        assert composition.net_charge_per_residue(sequence)[0] == 0.0


def test_empty_sequence_fractions_are_zero():
    assert composition.fraction_charged("")[0] == 0.0
    assert composition.amino_acid_fractions("")[0].sum() == 0.0
    assert composition.net_charge_per_residue("")[0] == 0.0


# -- scales ----------------------------------------------------------------------


def test_scale_values_are_the_scale_entries():
    values = encoding.scale_values("ACI", amino_acids.AA_hydro_KD)
    assert values.tolist() == [amino_acids.AA_hydro_KD[r] for r in "ACI"]


def test_normalized_kyte_doolittle_values_are_between_zero_and_one():
    values = encoding.scale_values(ALL_TWENTY, amino_acids.AA_hydro_KD_normalized)
    assert values.min() >= 0.0 and values.max() <= 1.0


def test_hydrophobicity_is_the_mean_of_the_per_residue_values():
    sequence = "ACDEFGHIKL"
    expected = np.mean(encoding.scale_values(sequence, amino_acids.AA_hydro_KD))
    assert composition.hydrophobicity(sequence)[0] == pytest.approx(expected)


def test_mean_scale_raises_only_for_residues_the_sequence_contains():
    scale = {residue: 1.0 for residue in ALL_TWENTY if residue != "W"}
    assert composition.mean_scale("MEEK", scale)[0] == 1.0
    with pytest.raises(SparrowException, match="'W'"):
        composition.mean_scale("MEWK", scale)


def test_mean_scale_of_an_empty_sequence_is_nan():
    assert np.isnan(composition.hydrophobicity("")[0])
    assert np.isnan(composition.polyproline_II_propensity("")[0])


def test_hydrophobicity_rejects_a_non_standard_residue():
    with pytest.raises(SparrowException):
        composition.hydrophobicity("ACBX")


# -- molecular weight ------------------------------------------------------------


def test_single_residue_weight_is_the_free_amino_acid():
    assert composition.molecular_weight("G")[0] == pytest.approx(amino_acids.AA_MOLECULAR_WEIGHT["G"])


def test_each_peptide_bond_loses_one_water():
    glycine = amino_acids.AA_MOLECULAR_WEIGHT["G"]
    assert composition.molecular_weight("GG")[0] == pytest.approx(2 * glycine - composition.WATER_MASS_DA)
    expected = sum(amino_acids.AA_MOLECULAR_WEIGHT[r] for r in "GAS") - 2 * composition.WATER_MASS_DA
    assert composition.molecular_weight("GAS")[0] == pytest.approx(expected)


def test_empty_sequence_weighs_nothing():
    assert composition.molecular_weight("")[0] == 0.0


def test_protein_molecular_weight_is_the_same_number():
    sequence = "MEEEKKKKSSSTTTDDD"
    assert Protein(sequence).molecular_weight == composition.molecular_weight(sequence)[0]


# -- SEG complexity --------------------------------------------------------------


def test_seg_complexity_of_a_homopolymer_is_zero():
    assert composition.seg_complexity("AAAAAAAA")[0] == 0.0


def test_seg_complexity_increases_with_diversity():
    low, high = composition.seg_complexity(["AAAAAAAC", "ACDEFGHI"])
    assert 0.0 < low < high


def test_seg_complexity_of_all_twenty_equally_is_one():
    assert composition.seg_complexity(ALL_TWENTY)[0] == pytest.approx(1.0)


def test_seg_complexity_of_an_empty_sequence_is_zero():
    assert composition.seg_complexity("")[0] == 0.0


def test_seg_complexity_needs_at_least_two_residues_in_the_alphabet():
    with pytest.raises(SparrowException, match="at least two"):
        composition.seg_complexity("AAAA", alphabet="A")


# -- ratios ----------------------------------------------------------------------


def test_log_residue_ratio_uses_a_pseudo_count():
    assert composition.log_residue_ratio("RRRGGG", "R", "K")[0] == pytest.approx(np.log10(4))
    assert composition.log_residue_ratio("GGG", "R", "K")[0] == 0.0


def test_log_residue_ratio_takes_single_residues():
    with pytest.raises(SparrowException, match="single residues"):
        composition.log_residue_ratio("RRK", "RK", "E")
