"""Tests for sparrow.properties.charge."""

import random

import numpy as np
import pytest

from sparrow import Protein
from sparrow.data import amino_acids
from sparrow.properties import charge

ORACLE_SEED = 31
ORACLE_SEQUENCES = 300


def oracle_charge(sequence, pH):
    """Mean titratable charge, one residue at a time - the original implementation."""
    total = 0.0
    titratable = 0
    for residue in sequence:
        if residue in amino_acids.TITRATABLE_BASES:
            total += 1.0 / (1.0 + 10.0 ** (pH - amino_acids.AA_PKA[residue]))
            titratable += 1
        elif residue in amino_acids.TITRATABLE_ACIDS:
            total += -1.0 / (1.0 + 10.0 ** (amino_acids.AA_PKA[residue] - pH))
            titratable += 1
    return total / titratable if titratable else 0.0


def oracle_isoelectric_point(sequence, threshold=charge.PI_THRESHOLD):
    """localCIDER's bisection for one sequence, as sparrow used to compute it."""
    low, high = charge.PI_SEARCH_RANGE
    current = 0.0
    steps = 0
    widenings = 0
    while True:
        steps += 1
        if steps == charge.PI_STEPS_PER_WIDENING:
            if widenings == charge.PI_MAX_WIDENINGS:
                return 0.5 * (high + low)
            widenings += 1
            steps = 0
            if current > 0:
                high += 1
            else:
                low -= 1
        middle = 0.5 * (high + low)
        current = oracle_charge(sequence, middle)
        if current > threshold:
            low = middle
        elif current < -threshold:
            high = middle
        else:
            return middle


def oracle_corpus():
    rng = random.Random(ORACLE_SEED)
    residues = amino_acids.VALID_AMINO_ACIDS
    sequences = ["".join(rng.choice(residues) for _ in range(rng.randint(5, 300))) for _ in range(ORACLE_SEQUENCES)]
    # pIs pushed outside the starting range, so the widening path is exercised
    sequences += ["R" * 40, "RRRRRRRRRK", "D" * 40, "DDDDDDDDDE", "C" * 12, "Y" * 12, "H" * 12]
    return sequences


def test_charge_at_ph_extremes():
    # at very low pH the acidic groups are protonated and the bases are charged
    assert charge.charge_at_pH("KKKKEEEE", pH=0.0)[0] > 0
    # at very high pH the reverse
    assert charge.charge_at_pH("KKKKEEEE", pH=14.0)[0] < 0
    # no titratable residues -> zero
    assert charge.charge_at_pH("AAAAGGGG")[0] == 0.0


def test_charge_at_ph_agrees_with_the_one_residue_at_a_time_sum():
    sequences = oracle_corpus()
    for pH in (2.0, 7.4, 11.0):
        batch = charge.charge_at_pH(sequences, pH=pH)
        reference = np.array([oracle_charge(sequence, pH) for sequence in sequences])
        np.testing.assert_allclose(batch, reference, rtol=0, atol=1e-14)


def test_summed_charge_is_the_mean_times_the_number_of_titratable_residues():
    sequence = "MEEEKKKSSSTTTDDDRRRHHHYYYCCC"
    titratable = sum(sequence.count(r) for r in amino_acids.AA_PKA)
    summed = charge.charge_at_pH(sequence, normalize=False)[0]
    assert summed == pytest.approx(titratable * charge.charge_at_pH(sequence)[0])


def test_isoelectric_point_sanity():
    # a strongly basic sequence has a high pI, an acidic one a low pI
    assert charge.isoelectric_point("KKKKKKRRRRRR")[0] > 10
    assert charge.isoelectric_point("EEEEEEDDDDDD")[0] < 5


def test_isoelectric_point_without_titratable_residues_is_seven():
    assert charge.isoelectric_point(["AAAAGGGG", ""]).tolist() == [7.0, 7.0]


def test_isoelectric_point_matches_the_original_bisection_exactly():
    """Bisecting the whole batch in step must give each sequence its own answer.

    The batch sums the charge per residue type rather than per residue, so the
    charges differ from the original's in the last bits; the bisection only
    compares them against the threshold, and across this corpus (and the 24,508
    human IDRs) that never changes a single step.
    """
    sequences = oracle_corpus()
    batch = charge.isoelectric_point(sequences)
    reference = np.array([oracle_isoelectric_point(sequence) for sequence in sequences])
    assert np.array_equal(batch, reference)


def test_isoelectric_point_is_where_the_charge_crosses_zero():
    for sequence in ["MEEEKKKSSSTTT", "EEEEDDDD", "KKKKRRRR", "GSGSGSGS"]:
        pI = charge.isoelectric_point(sequence)[0]
        assert abs(charge.charge_at_pH(sequence, pH=pI)[0]) <= charge.PI_THRESHOLD


def test_protein_properties_are_the_same_numbers():
    sequence = "MEEEKKKSSSTTTDDDRRRHHHYYYCCC"
    protein = Protein(sequence)
    assert protein.isoelectric_point == charge.isoelectric_point(sequence)[0]
    assert protein.charge_at_pH(5.0, normalize=False) == charge.charge_at_pH(sequence, 5.0, normalize=False)[0]
