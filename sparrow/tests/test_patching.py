"""Tests for sparrow.sequence_analysis.patching."""

import pytest

from sparrow.sequence_analysis import patching
from sparrow.sparrow_exceptions import CalculationException


def test_patch_fraction_simple_block():
    # 4 contiguous A's out of 8 residues, min_target_count default 4
    frac = patching.patch_fraction("AAAAGGGG", "A", min_target_count=4)
    assert frac == pytest.approx(0.5)


def test_patch_fraction_no_hits_is_zero():
    assert patching.patch_fraction("GGGGGGGG", "A") == 0.0


def test_patch_fraction_empty_sequence_is_zero():
    assert patching.patch_fraction("", "A") == 0.0


def test_patch_fraction_min_target_count_filters_small_patches():
    # a single A cannot meet min_target_count=4
    assert patching.patch_fraction("GGAGG", "A", min_target_count=4) == 0.0


def test_patch_fraction_interruption_bridges_gap():
    # with interruption=2, the two A-blocks separated by 2 G's bridge into one patch
    bridged = patching.patch_fraction("AAGGAA", "A", interruption=2, min_target_count=4)
    unbridged = patching.patch_fraction("AAGGAA", "A", interruption=0, min_target_count=4)
    assert bridged > unbridged


def test_patch_fraction_motif_requirement():
    seq = "RGRGRGTTTT"
    with_motif = patching.patch_fraction(
        seq, "RG", min_target_count=None, adjacent_pair_pattern="RG", min_adjacent_pair_count=2
    )
    assert with_motif > 0.0


def test_rg_patch_fraction_matches_generic_call():
    seq = "AAAAQQRGRGTTTAAAQQ"
    assert patching.rg_patch_fraction(seq) == pytest.approx(
        patching.patch_fraction(
            seq,
            "RG",
            min_target_count=None,
            adjacent_pair_pattern="RG",
            min_adjacent_pair_count=2,
        )
    )


def test_patch_fraction_negative_interruption_raises():
    with pytest.raises(CalculationException):
        patching.patch_fraction("AAAA", "A", interruption=-1)


def test_patch_fraction_invalid_residue_raises():
    with pytest.raises(CalculationException):
        patching.patch_fraction("AAAA", "Z")
