"""Tests for sparrow.properties.encoding."""

import numpy as np
import pytest

from sparrow.properties import encoding
from sparrow.sparrow_exceptions import SparrowException


# -- validate_protein_sequence ---------------------------------------------


def test_validate_uppercases_by_default():
    assert encoding.validate_protein_sequence("acdef") == "ACDEF"


def test_validate_preserves_case_when_disabled():
    # lowercase residues are not valid amino acids when uppercase=False
    with pytest.raises(ValueError):
        encoding.validate_protein_sequence("acdef", uppercase=False)


def test_validate_rejects_non_string():
    with pytest.raises(ValueError):
        encoding.validate_protein_sequence(123)


def test_validate_empty_allowed_by_default():
    assert encoding.validate_protein_sequence("") == ""


def test_validate_empty_rejected_when_disallowed():
    with pytest.raises(ValueError):
        encoding.validate_protein_sequence("", allow_empty=False)


def test_validate_reports_invalid_residues():
    with pytest.raises(ValueError) as exc:
        encoding.validate_protein_sequence("ACBXZ")
    # B and Z are invalid; X is invalid too
    assert "B" in str(exc.value)


def test_validate_custom_exception_class():
    with pytest.raises(SparrowException):
        encoding.validate_protein_sequence("123", exception_cls=SparrowException)


# -- normalize_residue_selector --------------------------------------------


def test_normalize_string_to_list():
    assert encoding.normalize_residue_selector("ED") == ["E", "D"]


def test_normalize_iterable_input():
    assert encoding.normalize_residue_selector(["e", "d"]) == ["E", "D"]


def test_normalize_unique_preserves_first_seen_order():
    assert encoding.normalize_residue_selector("EDE", unique=True) == ["E", "D"]


def test_normalize_unique_sorted():
    assert encoding.normalize_residue_selector("KED", unique=True, sort_unique=True) == [
        "D",
        "E",
        "K",
    ]


def test_normalize_return_type_str():
    assert encoding.normalize_residue_selector(["E", "D"], return_type="str") == "ED"


def test_normalize_expected_length_enforced():
    with pytest.raises(ValueError):
        encoding.normalize_residue_selector("E", expected_length=2)


def test_normalize_rejects_invalid_residue():
    with pytest.raises(ValueError):
        encoding.normalize_residue_selector("EZ")


def test_normalize_require_nonempty():
    with pytest.raises(ValueError):
        encoding.normalize_residue_selector("", require_nonempty=True)


# -- is_valid_protein_sequence ---------------------------------------------


@pytest.mark.parametrize(
    "seq,expected",
    [("ACDEFGHIKLMNPQRSTVWY", True), ("ACBX", False), ("", True)],
)
def test_is_valid_protein_sequence(seq, expected):
    assert encoding.is_valid_protein_sequence(seq) is expected


# -- residue_codes ------------------------------------------------------------


def test_residue_codes_follow_the_residue_order():
    codes = encoding.residue_codes("".join(encoding.RESIDUES))
    assert codes.dtype == np.int8
    assert codes.tolist() == list(range(encoding.NUM_RESIDUES))


def test_residue_codes_of_an_empty_sequence_are_empty():
    assert encoding.residue_codes("").size == 0


@pytest.mark.parametrize("sequence,bad", [("MEEKXKK", "X"), ("MEEK\u00c5KK", "\u00c5"), ("meek", "e")])
def test_residue_codes_name_every_non_standard_residue(sequence, bad):
    """Non-ASCII input must get sparrow's error, not a UnicodeEncodeError."""
    with pytest.raises(SparrowException, match="non-standard residue") as error:
        encoding.residue_codes(sequence)
    assert repr(bad) in str(error.value)


# -- ascii_codes and membership_lookup ----------------------------------------


def test_membership_lookup_marks_exactly_the_residues():
    table = encoding.membership_lookup("DE")
    assert table.shape == (encoding.ASCII_TABLE_SIZE,)
    assert table[encoding.ascii_codes("MDEK")].tolist() == [False, True, True, False]
    assert int(table.sum()) == 2


# -- ternarize_winged ----------------------------------------------------------


def test_ternarize_winged_marks_groups_and_pads_with_zeros():
    ternary = encoding.ternarize_winged("EKGE", ["E"], ["K"], 2)
    assert ternary.tolist() == [0, 0, 1, -1, 0, 1, 0, 0]


def test_ternarize_winged_with_no_second_group_is_binary():
    ternary = encoding.ternarize_winged("EKGE", ["E"], [], 1)
    assert ternary.tolist() == [0, 1, -1, -1, 1, 0]


def test_ternarize_winged_batch_matches_one_at_a_time():
    sequences = ["EKGEDR", "RRKKEE", "GSGSGS"]
    batch = encoding.ternarize_winged_batch(
        np.stack([encoding.ascii_codes(s) for s in sequences]), ["E", "D"], ["R", "K"], 3
    )
    for row, sequence in zip(batch, sequences):
        assert row.tolist() == encoding.ternarize_winged(sequence, ["E", "D"], ["R", "K"], 3).tolist()
