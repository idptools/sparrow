"""Tests for sparrow.properties.profiles, the linear profiles.

The central check is that each point of a profile is exactly what the property
gives for that window as a sequence of its own - to the last bit, because the
fractions and the net charge are an integer window sum over the window size,
just as the composition functions compute them.
"""

import random

import numpy as np
import pytest
from scipy.signal import savgol_filter

from sparrow import Protein
from sparrow.data import amino_acids
from sparrow.properties import composition, patterning, profiles
from sparrow.sparrow_exceptions import SparrowException

SEQ = "ACDEFGHIKLMNPQRSTVWY"  # length 20, all standard residues
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
WINDOW = 8
N_RANDOM_CASES = 40


def _windows(sequence, window_size):
    """Every window of a sequence, as strings."""
    return [sequence[i : i + window_size] for i in range(len(sequence) - window_size + 1)]


def _random_cases(seed):
    """Random (sequence, window size) pairs, the window never longer than the sequence."""
    rng = random.Random(seed)
    cases = []
    for _ in range(N_RANDOM_CASES):
        length = rng.randint(1, 200)
        sequence = "".join(rng.choice(AMINO_ACIDS) for _ in range(length))
        cases.append((sequence, rng.randint(1, length)))
    return cases


def _ones(windows):
    return np.ones(len(windows))


# -- end modes -------------------------------------------------------------


def test_extend_ends_length_matches_sequence():
    profile = profiles.window_profile(SEQ, _ones, WINDOW, end_mode="extend-ends")[0]
    assert len(profile) == len(SEQ)


def test_zero_ends_length_matches_sequence():
    # regression: zero-ends previously produced a track shorter than the sequence
    profile = profiles.window_profile(SEQ, _ones, WINDOW, end_mode="zero-ends")[0]
    assert len(profile) == len(SEQ)


def test_no_ends_gives_one_value_per_window():
    profile = profiles.window_profile(SEQ, _ones, WINDOW, end_mode="")[0]
    assert len(profile) == len(SEQ) - WINDOW + 1


def test_zero_ends_pads_with_zeros():
    profile = profiles.window_profile(SEQ, _ones, WINDOW, end_mode="zero-ends")[0]
    front = WINDOW // 2
    back = len(SEQ) - front - (len(SEQ) - WINDOW + 1)
    assert np.all(profile[:front] == 0.0)
    assert np.all(profile[len(SEQ) - back :] == 0.0)
    assert np.all(profile[front : len(SEQ) - back] == 1.0)


def test_extend_ends_centres_each_window():
    sequence = "EEEEEEEEGGGGGGGGKKKKKKKK"
    per_window = profiles.net_charge_profile(sequence, WINDOW, end_mode="")[0]
    profile = profiles.net_charge_profile(sequence, WINDOW, end_mode="extend-ends")[0]
    front = WINDOW // 2
    assert np.all(profile[:front] == per_window[0])
    np.testing.assert_array_equal(profile[front : front + per_window.size], per_window)
    assert np.all(profile[front + per_window.size :] == per_window[-1])


# -- arguments -------------------------------------------------------------


def test_window_larger_than_sequence_raises():
    with pytest.raises(SparrowException, match="larger than sequence"):
        profiles.window_profile("ACDE", _ones, window_size=10)


def test_batch_error_names_the_short_sequence():
    with pytest.raises(SparrowException, match="Sequence 1 of 2"):
        profiles.fraction_profile([SEQ, "ACDE"], ["A"], window_size=10)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(smooth=True),
        dict(smooth=5.0),
        dict(window_size=0),
        dict(window_size=2.5),
        dict(end_mode="bogus"),
    ],
)
def test_invalid_arguments_raise(kwargs):
    arguments = dict(window_size=5, end_mode="extend-ends", smooth=None) | kwargs
    with pytest.raises(SparrowException):
        profiles.window_profile(SEQ, _ones, **arguments)


@pytest.mark.parametrize("smooth", [2, 99])
def test_an_unusable_smoothing_window_raises(smooth):
    # the filter needs more points than its cubic, and no more than the profile
    with pytest.raises(SparrowException, match="Cannot smooth"):
        profiles.net_charge_profile(SEQ, 5, smooth=smooth)


def test_property_function_must_return_one_value_per_window():
    with pytest.raises(SparrowException, match="one value per window"):
        profiles.window_profile(SEQ, composition.amino_acid_fractions, 5)


def test_non_standard_residue_raises():
    with pytest.raises(SparrowException):
        profiles.net_charge_profile("ACDEXGHIK", 3)


# -- each window as a sequence of its own -----------------------------------


def test_fraction_profile_is_the_fraction_of_each_window():
    rng = random.Random(5)
    for sequence, window_size in _random_cases(1):
        residues = rng.sample(list(AMINO_ACIDS), rng.randint(1, 6))
        profile = profiles.fraction_profile(sequence, residues, window_size, end_mode="")[0]
        expected = composition.fraction_in_set(_windows(sequence, window_size), residues)
        np.testing.assert_array_equal(profile, expected)


def test_net_charge_profile_is_the_ncpr_of_each_window():
    for sequence, window_size in _random_cases(2):
        profile = profiles.net_charge_profile(sequence, window_size, end_mode="")[0]
        expected = composition.net_charge_per_residue(_windows(sequence, window_size))
        np.testing.assert_array_equal(profile, expected)


def test_scale_profile_is_the_mean_of_each_window():
    for sequence, window_size in _random_cases(3):
        profile = profiles.scale_profile(sequence, amino_acids.AA_hydro_KD, window_size, end_mode="")[0]
        expected = composition.mean_scale(_windows(sequence, window_size), amino_acids.AA_hydro_KD)
        np.testing.assert_array_equal(profile, expected)


def test_kappa_profile_is_the_kappa_of_each_window():
    for sequence, window_size in _random_cases(4):
        profile = profiles.window_profile(sequence, patterning.kappa, window_size, end_mode="")[0]
        np.testing.assert_array_equal(profile, patterning.kappa(_windows(sequence, window_size)))


def test_the_direct_and_general_net_charge_profiles_agree():
    for sequence, window_size in _random_cases(5):
        direct = profiles.net_charge_profile(sequence, window_size)[0]
        general = profiles.window_profile(sequence, composition.net_charge_per_residue, window_size)[0]
        np.testing.assert_array_equal(direct, general)


def test_fraction_profile_of_no_residues_is_zero():
    assert np.all(profiles.fraction_profile(SEQ, [], 5)[0] == 0.0)


def test_smoothing_is_a_cubic_savgol_filter_of_the_profile():
    sequence = "ACDEFGHIKLMNPQRSTVWY" * 3
    unsmoothed = profiles.net_charge_profile(sequence, 5)[0]
    smoothed = profiles.net_charge_profile(sequence, 5, smooth=7)[0]
    np.testing.assert_array_equal(smoothed, savgol_filter(unsmoothed, 7, profiles.SAVGOL_POLYNOMIAL_ORDER))


def test_splitting_the_windows_into_several_calls_changes_nothing(monkeypatch):
    rng = random.Random(6)
    batch = ["".join(rng.choice(AMINO_ACIDS) for _ in range(rng.randint(10, 80))) for _ in range(30)]
    whole = profiles.window_profile(batch, patterning.kappa, 10)
    # 7 windows of 10 residues per call: the calls split sequences as well as
    # falling between them
    monkeypatch.setattr(profiles, "_WINDOW_RESIDUES_PER_CALL", 70)
    split = profiles.window_profile(batch, patterning.kappa, 10)
    for one, other in zip(whole, split):
        np.testing.assert_array_equal(one, other)


# -- Protein's linear profiles ---------------------------------------------


@pytest.mark.parametrize(
    "mode",
    [
        "FCR",
        "NCPR",
        "aromatic",
        "aliphatic",
        "polar",
        "proline",
        "positive",
        "negative",
        "hydrophobicity",
        "seg-complexity",
        "kappa",
    ],
)
def test_linear_sequence_profile_length(mode):
    p = Protein(SEQ)
    track = p.linear_sequence_profile(mode, window_size=8, end_mode="extend-ends")
    assert len(track) == len(SEQ)


def test_linear_sequence_profile_is_the_window_profile():
    sequence = "MEEEKKKKSSSTTTDDDQQQQNNNNGGGKEKEKEKE"
    p = Protein(sequence)
    np.testing.assert_array_equal(
        p.linear_sequence_profile("kappa", window_size=10),
        profiles.window_profile(sequence, patterning.kappa, 10)[0],
    )


def test_linear_sequence_profile_invalid_mode_raises():
    p = Protein(SEQ)
    with pytest.raises(SparrowException):
        p.linear_sequence_profile("not-a-mode")


def test_linear_composition_profile_values_in_unit_range():
    p = Protein(SEQ)
    track = p.linear_composition_profile(["A", "C"], window_size=5)
    assert len(track) == len(SEQ)
    assert track.min() >= 0.0 and track.max() <= 1.0


def test_linear_composition_profile_ignores_non_standard_residues():
    p = Protein(SEQ)
    np.testing.assert_array_equal(
        p.linear_composition_profile(["A", "X"], window_size=5),
        p.linear_composition_profile(["A"], window_size=5),
    )


def test_linear_sequence_profile_smoothing_runs():
    p = Protein("ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY")
    smoothed = p.linear_sequence_profile("NCPR", window_size=5, smooth=5)
    assert len(smoothed) == len(p)
