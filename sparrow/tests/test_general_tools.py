"""Tests for sparrow.tools.general_tools."""

import pytest

from sparrow.sparrow_exceptions import SparrowException
from sparrow.tools import general_tools as gt


# -- validate_protein_sequence ---------------------------------------------

def test_validate_uppercases_by_default():
    assert gt.validate_protein_sequence("acdef") == "ACDEF"


def test_validate_preserves_case_when_disabled():
    # lowercase residues are not valid amino acids when uppercase=False
    with pytest.raises(ValueError):
        gt.validate_protein_sequence("acdef", uppercase=False)


def test_validate_rejects_non_string():
    with pytest.raises(ValueError):
        gt.validate_protein_sequence(123)


def test_validate_empty_allowed_by_default():
    assert gt.validate_protein_sequence("") == ""


def test_validate_empty_rejected_when_disallowed():
    with pytest.raises(ValueError):
        gt.validate_protein_sequence("", allow_empty=False)


def test_validate_reports_invalid_residues():
    with pytest.raises(ValueError) as exc:
        gt.validate_protein_sequence("ACBXZ")
    # B and Z are invalid; X is invalid too
    assert "B" in str(exc.value)


def test_validate_custom_exception_class():
    with pytest.raises(SparrowException):
        gt.validate_protein_sequence("123", exception_cls=SparrowException)


# -- normalize_residue_selector --------------------------------------------

def test_normalize_string_to_list():
    assert gt.normalize_residue_selector("ED") == ["E", "D"]


def test_normalize_iterable_input():
    assert gt.normalize_residue_selector(["e", "d"]) == ["E", "D"]


def test_normalize_unique_preserves_first_seen_order():
    assert gt.normalize_residue_selector("EDE", unique=True) == ["E", "D"]


def test_normalize_unique_sorted():
    assert gt.normalize_residue_selector("KED", unique=True, sort_unique=True) == [
        "D",
        "E",
        "K",
    ]


def test_normalize_return_type_str():
    assert gt.normalize_residue_selector(["E", "D"], return_type="str") == "ED"


def test_normalize_expected_length_enforced():
    with pytest.raises(ValueError):
        gt.normalize_residue_selector("E", expected_length=2)


def test_normalize_rejects_invalid_residue():
    with pytest.raises(ValueError):
        gt.normalize_residue_selector("EZ")


def test_normalize_require_nonempty():
    with pytest.raises(ValueError):
        gt.normalize_residue_selector("", require_nonempty=True)


# -- is_valid_protein_sequence ---------------------------------------------

@pytest.mark.parametrize(
    "seq,expected",
    [("ACDEFGHIKLMNPQRSTVWY", True), ("ACBX", False), ("", True)],
)
def test_is_valid_protein_sequence(seq, expected):
    assert gt.is_valid_protein_sequence(seq) is expected


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
