"""Tests for sparrow.sequence_analysis.elm: finding ELM matches, and comparing a
sequence's matches with those of a point mutant."""

import numpy as np
import pytest

from sparrow.sequence_analysis.elm import (
    ELM,
    compute_gained_elms,
    compute_lost_elms,
    compute_retained_elms,
    find_all_elms,
    parse_hgvs,
)
from sparrow.sparrow_exceptions import SparrowException

# S7 sits in several overlapping kinase motifs, which the old non-overlapping
# search hid behind one another
KINASE_SITE_SEQUENCE = "MKKRPKSPSGSPTSRRPLL"
KINASE_SITE_SUBSTITUTION = "p.S7A"
KINASE_SITE_POSITION = 6

REPEAT_SEQUENCE = "KRRARKRRARKRRARKRRAR"


def make_elm(start, end, sequence, regex="[KR]R.", site="site"):
    return ELM(regex, "IDENTIFIER", site, "description", 0.01, start, end, sequence)


def test_overlapping_matches_of_one_class_are_all_found():
    # KRR at 0 and RRA at 1 overlap; the old search reported only the first
    spans = {
        (motif.start, motif.end) for motif in find_all_elms(REPEAT_SEQUENCE) if motif.identifier == "CLV_PCSK_KEX2_1"
    }
    assert {(0, 3), (1, 4)} <= spans


def test_a_substitution_gains_no_motif_the_wild_type_already_has():
    assert compute_gained_elms(KINASE_SITE_SEQUENCE, KINASE_SITE_SUBSTITUTION) == set()


def test_every_motif_lost_or_gained_covers_the_substituted_residue():
    lost = compute_lost_elms(KINASE_SITE_SEQUENCE, KINASE_SITE_SUBSTITUTION)
    gained = compute_gained_elms(KINASE_SITE_SEQUENCE, "p.P8K")
    assert lost
    assert all(motif.start <= KINASE_SITE_POSITION < motif.end for motif in lost)
    assert all(motif.start <= KINASE_SITE_POSITION + 1 < motif.end for motif in gained)


def test_lost_and_retained_partition_the_wild_type_matches():
    wild_type = find_all_elms(KINASE_SITE_SEQUENCE)
    lost = compute_lost_elms(KINASE_SITE_SEQUENCE, KINASE_SITE_SUBSTITUTION)
    retained = compute_retained_elms(KINASE_SITE_SEQUENCE, KINASE_SITE_SUBSTITUTION)
    assert lost | retained == wild_type
    assert not lost & retained


def test_equal_matches_have_equal_hashes():
    # same class and start, different extent: the same motif
    first = make_elm(3, 6, "KRR")
    second = make_elm(3, 8, "KRRAR")
    assert first == second
    assert hash(first) == hash(second)


def test_overlapping_matches_that_start_apart_are_different_motifs():
    assert make_elm(0, 3, "KRR") != make_elm(1, 4, "RRA")
    assert make_elm(0, 3, "KRR") != make_elm(0, 3, "KRR", site="another site")
    assert len({make_elm(0, 3, "KRR"), make_elm(1, 4, "RRA")}) == 2


@pytest.mark.parametrize(
    ("notation", "expected"),
    [
        ("p.S7A", (6, "A")),
        ("p.Ser7Ala", (6, "A")),
        ("p.Ser7Val", (6, "V")),
        ("p.(Ser7Val)", (6, "V")),
        ("p.s7a", (6, "A")),
        ("p.Lys120Arg", (119, "R")),
    ],
)
def test_parse_hgvs_reads_one_and_three_letter_codes(notation, expected):
    assert parse_hgvs(notation) == expected


@pytest.mark.parametrize(
    "notation",
    ["S7A", "p.7A", "p.S7", "p.S0A", "p.Ser7Ter", "p.S7*", "p.S7=", "p.Xyz7Ala", "p.S7X", "p.(Ser7Val", "p.S7A extra"],
)
def test_parse_hgvs_rejects_anything_but_a_standard_substitution(notation):
    with pytest.raises(SparrowException):
        parse_hgvs(notation)


def test_the_wild_type_residue_must_match_the_sequence():
    with pytest.raises(SparrowException, match="expects K at position 1"):
        compute_lost_elms("MKKK", "p.K1R")


@pytest.mark.parametrize(
    "query",
    [(4, "R"), (-1, "R"), (1.0, "R"), (True, "R"), (1, "X"), (1, "RR"), (1, 5), "p.K5R", (1,), 3],
)
def test_a_substitution_outside_the_sequence_or_unreadable_raises(query):
    with pytest.raises(SparrowException):
        compute_lost_elms("MKKK", query)


def test_a_position_pair_can_be_any_two_item_sequence():
    expected = compute_lost_elms("MKKK", (1, "R"))
    assert compute_lost_elms("MKKK", [1, "R"]) == expected
    assert compute_lost_elms("MKKK", (np.int64(1), "r")) == expected
