"""Tests for the PLAAC entry points: score_sequence and score_sequences must refuse
the same inputs, and agree on what they score."""

import numpy as np
import pytest

from sparrow.sequence_analysis.plaac.plaac import score_sequence, score_sequences

PRION_LIKE = "MFKSGNQNN" * 20
ORDINARY = "ACDEFGHIKLMNPQRSTVWY" * 10

# a custom background, which only counts when alpha is below 1
UNIFORM_BACKGROUND = dict.fromkeys("ACDEFGHIKLMNPQRSTVWY", 0.05)
MIXED_ALPHA = 0.5


def test_score_sequences_matches_score_sequence():
    batch = score_sequences({"prion_like": PRION_LIKE, "ordinary": ORDINARY})
    for name, sequence in (("prion_like", PRION_LIKE), ("ordinary", ORDINARY)):
        single = score_sequence(sequence)
        # assert_equal treats NaN as equal to NaN: a sequence with no prion-like
        # core has a NaN core score
        np.testing.assert_equal(
            (batch[name].core_score, batch[name].core_start, batch[name].core_end),
            (single.core_score, single.core_start, single.core_end),
        )


@pytest.mark.parametrize("empty", ["", "*", "**"])
def test_an_empty_sequence_raises_in_both(empty):
    with pytest.raises(ValueError, match="Empty sequence"):
        score_sequence(empty)
    with pytest.raises(ValueError, match="'blank'"):
        score_sequences({"fine": PRION_LIKE, "blank": empty})


def test_score_sequences_names_every_empty_sequence():
    with pytest.raises(ValueError, match="'first'.*'second'"):
        score_sequences({"first": "", "fine": PRION_LIKE, "second": "*"})


def test_a_background_alpha_ignores_raises_in_both():
    with pytest.raises(ValueError, match="alpha=1.0"):
        score_sequence(PRION_LIKE, bg_freqs=UNIFORM_BACKGROUND)
    with pytest.raises(ValueError, match="alpha=1.0"):
        score_sequences({"prion_like": PRION_LIKE}, bg_freqs=UNIFORM_BACKGROUND)


def test_a_background_with_a_lower_alpha_is_used_by_both():
    single = score_sequence(PRION_LIKE, alpha=MIXED_ALPHA, bg_freqs=UNIFORM_BACKGROUND)
    batch = score_sequences({"prion_like": PRION_LIKE}, alpha=MIXED_ALPHA, bg_freqs=UNIFORM_BACKGROUND)
    assert batch["prion_like"].core_score == single.core_score
