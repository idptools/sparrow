"""Tests for sparrow.properties.patches."""

import random

import numpy as np
import pytest

from sparrow.properties import patches
from sparrow.sparrow_exceptions import CalculationException, SparrowException

ORACLE_SEED = 4
ORACLE_BATCHES = 300
ORACLE_ALPHABET = "AGQSRKEDNP"


def oracle_patch_fraction(sequence, selector, interruption, min_target_count, motif, min_pairs):
    """Patch fraction of one sequence, one patch at a time - the original algorithm."""
    if not sequence:
        return 0.0
    hit = np.array([residue in selector for residue in sequence])
    positions = list(np.flatnonzero(hit))
    if not positions:
        return 0.0

    # bridge gaps of at most `interruption` residues between consecutive hits
    covered = set(positions)
    for left, right in zip(positions, positions[1:]):
        if 1 < right - left <= interruption + 1:
            covered.update(range(left + 1, right))

    span = 0
    start = None
    for position in range(len(sequence) + 1):
        inside = position in covered
        if inside and start is None:
            start = position
        if not inside and start is not None:
            region = sequence[start:position]
            keep = True
            if min_target_count is not None and int(hit[start:position].sum()) < min_target_count:
                keep = False
            if motif is not None:
                pairs = sum(1 for a, b in zip(region, region[1:]) if a + b == motif)
                if len(region) < 2 or pairs < min_pairs:
                    keep = False
            if keep:
                span += position - start
            start = None
    return span / len(sequence)


def test_patch_fraction_simple_block():
    # 4 contiguous A's out of 8 residues, min_target_count default 4
    assert patches.patch_fraction("AAAAGGGG", "A")[0] == 0.5


def test_patch_fraction_no_hits_is_zero():
    assert patches.patch_fraction("GGGGGGGG", "A")[0] == 0.0


def test_patch_fraction_empty_sequence_is_zero():
    assert patches.patch_fraction("", "A")[0] == 0.0


def test_patch_fraction_min_target_count_filters_small_patches():
    # a single A cannot meet min_target_count=4
    assert patches.patch_fraction("GGAGG", "A", min_target_count=4)[0] == 0.0


def test_patch_fraction_interruption_bridges_gap():
    # with interruption=2, the two A-blocks separated by 2 G's bridge into one patch
    bridged = patches.patch_fraction("AAGGAA", "A", interruption=2)[0]
    unbridged = patches.patch_fraction("AAGGAA", "A", interruption=0)[0]
    assert bridged == 1.0
    assert unbridged == 0.0


def test_patch_fraction_motif_requirement():
    sequence = "RGRGRGTTTT"
    with_motif = patches.patch_fraction(
        sequence, "RG", min_target_count=None, adjacent_pair_pattern="RG", min_adjacent_pair_count=2
    )
    assert with_motif[0] == 0.6


def test_patches_do_not_cross_from_one_sequence_into_the_next():
    """Laid end to end, AAA + A would bridge into a patch of four; it must not."""
    batch = ["GGAAA", "AGGGG"]
    assert patches.patch_fraction(batch, "A").tolist() == [0.0, 0.0]
    assert patches.patch_fraction("GGAAAAGGGG", "A")[0] == 0.4


def test_patch_fraction_negative_interruption_raises():
    with pytest.raises(CalculationException):
        patches.patch_fraction("AAAA", "A", interruption=-1)


def test_patch_fraction_invalid_selector_residue_raises():
    with pytest.raises(CalculationException):
        patches.patch_fraction("AAAA", "Z")


def test_patch_fraction_non_standard_sequence_raises():
    with pytest.raises(SparrowException, match="non-standard"):
        patches.patch_fraction("AAXA", "A")


def test_patch_fraction_matches_the_original_algorithm():
    """Random batches and parameters, against a transcription of the original."""
    rng = random.Random(ORACLE_SEED)
    for _ in range(ORACLE_BATCHES):
        batch = [
            "".join(rng.choice(ORACLE_ALPHABET) for _ in range(rng.randint(0, 40))) for _ in range(rng.randint(1, 10))
        ]
        selector = "".join(sorted(rng.sample(ORACLE_ALPHABET, rng.randint(1, 3))))
        interruption = rng.randint(0, 4)
        min_target_count = rng.choice([None, 1, 2, 3, 4, 6])
        motif = rng.choice([None, "RG", "QQ", "GA"])
        min_pairs = rng.randint(0, 3)

        got = patches.patch_fraction(
            batch,
            selector,
            interruption=interruption,
            min_target_count=min_target_count,
            adjacent_pair_pattern=motif,
            min_adjacent_pair_count=min_pairs,
        )
        want = [
            oracle_patch_fraction(sequence, selector, interruption, min_target_count, motif, min_pairs)
            for sequence in batch
        ]
        assert got.tolist() == want, (batch, selector, interruption, min_target_count, motif, min_pairs)
