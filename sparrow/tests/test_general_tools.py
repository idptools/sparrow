"""Tests for sparrow.tools.general_tools."""

import pytest

from sparrow.tools import general_tools as gt


# -- compare_sequence ------------------------------------------------------


def test_compare_sequence_counts_differences():
    assert gt.compare_sequence("ACDE", "AGDE") == 1


def test_compare_sequence_returns_positions():
    assert gt.compare_sequence("ACDE", "AGDF", return_positions=True) == [1, 3]


def test_compare_sequence_ignore_gaps():
    # the '-' positions are skipped, only the genuine mismatch counts
    assert gt.compare_sequence("AC-E", "A-DE", ignore_gaps=True) == 0


def test_compare_sequence_length_mismatch_raises():
    with pytest.raises(ValueError):
        gt.compare_sequence("ACDE", "ACD")
