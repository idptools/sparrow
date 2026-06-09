"""Tests for sparrow.tools.track_tools (linear track construction)."""

import numpy as np
import pytest

from sparrow import Protein
from sparrow.sparrow_exceptions import SparrowException
from sparrow.tools import track_tools

SEQ = "ACDEFGHIKLMNPQRSTVWY"  # length 20, all standard residues


def _const(_frag):
    return 1.0


# -- build_track end modes -------------------------------------------------

def test_extend_ends_length_matches_sequence():
    track = track_tools.build_track(SEQ, _const, window_size=8, end_mode="extend-ends")
    assert len(track) == len(SEQ)


def test_zero_ends_length_matches_sequence():
    # regression: zero-ends previously produced a track shorter than the sequence
    track = track_tools.build_track(SEQ, _const, window_size=8, end_mode="zero-ends")
    assert len(track) == len(SEQ)


def test_empty_end_mode_shorter_than_sequence():
    w = 8
    track = track_tools.build_track(SEQ, _const, window_size=w, end_mode="")
    assert len(track) == len(SEQ) - w + 1


def test_zero_ends_pads_with_zeros():
    track = track_tools.build_track(SEQ, _const, window_size=8, end_mode="zero-ends")
    # leading/trailing values are zero, middle values are the constant
    assert track[0] == 0
    assert track[-1] == 0
    assert np.any(track == 1.0)


def test_window_larger_than_sequence_raises():
    with pytest.raises(SparrowException):
        track_tools.build_track("ACDE", _const, window_size=10)


def test_smooth_bool_rejected():
    with pytest.raises(SparrowException):
        track_tools.build_track(SEQ, _const, window_size=5, smooth=True)


def test_invalid_end_mode_raises():
    with pytest.raises(SparrowException):
        track_tools.build_track(SEQ, _const, window_size=5, end_mode="bogus")


# -- linear_track_property -------------------------------------------------

def test_linear_track_property_window_mean():
    value_map = {r: float(i) for i, r in enumerate(SEQ)}
    w = 4
    track = track_tools.linear_track_property(SEQ, value_map, w, "")
    expected_first = np.mean([value_map[r] for r in SEQ[:w]])
    assert track[0] == pytest.approx(expected_first)
    assert len(track) == len(SEQ) - w + 1


# -- predefined_linear_track via Protein -----------------------------------

@pytest.mark.parametrize(
    "mode",
    ["FCR", "NCPR", "aromatic", "aliphatic", "polar", "proline",
     "positive", "negative", "hydrophobicity", "seg-complexity"],
)
def test_linear_sequence_profile_length(mode):
    p = Protein(SEQ)
    track = p.linear_sequence_profile(mode, window_size=8, end_mode="extend-ends")
    assert len(track) == len(SEQ)


def test_linear_sequence_profile_invalid_mode_raises():
    p = Protein(SEQ)
    with pytest.raises(SparrowException):
        p.linear_sequence_profile("not-a-mode")


def test_linear_composition_profile_values_in_unit_range():
    p = Protein(SEQ)
    track = p.linear_composition_profile(["A", "C"], window_size=5)
    assert len(track) == len(SEQ)
    assert track.min() >= 0.0 and track.max() <= 1.0


def test_linear_sequence_profile_smoothing_runs():
    p = Protein("ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY")
    smoothed = p.linear_sequence_profile("NCPR", window_size=5, smooth=5)
    assert len(smoothed) == len(p)
