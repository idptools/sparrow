"""Tests for sparrow.sequence_analysis.sequence_complexity (LCD extraction)."""

import pytest

import sparrow.batching
from sparrow import Protein
from sparrow.sequence_analysis import sequence_complexity as sc
from sparrow.sparrow_exceptions import SparrowException

# a Q-rich block flanked by alanine
SEQ = "AAAAA" + "Q" * 16 + "AAAAA"

# Qs interrupted by one A, flanked by S: one domain, from the first Q to the
# last, with the A inside it
INTERRUPTED = "SSQQAQQQSS"
INTERRUPTED_MASK = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]

# a run of Qs and an interrupted run, split by more alanines than the default
# max_interruption allows, so two domains
TWO_DOMAINS = "AAAAA" + "Q" * 12 + "A" * 8 + "QQAQQAQQAQQAQQ" + "AAA"


def test_holt_finds_q_rich_domain():
    lcds = sc.low_complexity_domains_holt(SEQ, "Q", minimum_length=10)
    assert len(lcds) == 1
    domain, start, end = lcds[0]
    # 0-indexed start, exclusive end; slice reproduces the domain
    assert SEQ[start:end] == domain
    assert set(domain) == {"Q"}


def test_holt_no_hits_returns_empty():
    assert sc.low_complexity_domains_holt("AAAAAAAAAA", "Q") == []


def test_holt_minimum_length_filter():
    # a short Q-run below minimum_length should not be reported
    assert sc.low_complexity_domains_holt("AAQQQAA", "Q", minimum_length=10) == []


def test_permissive_forwards_fractional_threshold():
    # regression: the permissive variant previously dropped fractional_threshold.
    # With an impossible-to-satisfy threshold it must return no domains; with a
    # lenient one it must find the Q block -> the argument is actually honored.
    strict = sc.low_complexity_domains_holt_permissive(SEQ, "Q", minimum_length=10, fractional_threshold=1.01)
    lenient = sc.low_complexity_domains_holt_permissive(SEQ, "Q", minimum_length=10, fractional_threshold=0.1)
    assert strict == []
    assert len(lenient) >= 1


def test_protein_low_complexity_domains_wrapper():
    p = Protein(SEQ)
    lcds = p.low_complexity_domains(mode="holt", residue_selector="Q", minimum_length=10)
    assert len(lcds) == 1


def test_protein_low_complexity_domains_permissive_mode():
    p = Protein(SEQ)
    lcds = p.low_complexity_domains(mode="holt-permissive", residue_selector="Q", minimum_length=10)
    assert isinstance(lcds, list)


def test_protein_low_complexity_domains_invalid_mode_raises():
    p = Protein(SEQ)
    with pytest.raises(Exception):
        p.low_complexity_domains(mode="not-a-mode", residue_selector="Q")


def test_the_mask_marks_each_residue_inside_a_domain_including_interruptions():
    assert Protein(INTERRUPTED).low_complexity_mask(residue_selector="Q", minimum_length=5) == INTERRUPTED_MASK


@pytest.mark.parametrize("mode", ["holt", "holt-permissive"])
def test_the_mask_is_one_exactly_inside_the_domains(mode):
    protein = Protein(TWO_DOMAINS)
    domains = protein.low_complexity_domains(mode=mode, residue_selector="Q", minimum_length=10)
    mask = protein.low_complexity_mask(mode=mode, residue_selector="Q", minimum_length=10)
    assert len(domains) == 2
    inside = {position for _, start, end in domains for position in range(start, end)}
    assert mask == [1 if position in inside else 0 for position in range(len(TWO_DOMAINS))]


def test_the_mask_of_a_sequence_without_domains_is_all_zeros():
    assert Protein("AAAAAAAAAA").low_complexity_mask(residue_selector="Q") == [0] * 10


def test_the_mask_refuses_an_unknown_mode():
    with pytest.raises(SparrowException, match="mode"):
        Protein(SEQ).low_complexity_mask(mode="not-a-mode", residue_selector="Q")


def test_the_old_predictor_name_gives_the_mask_with_a_deprecation_warning():
    protein = Protein(INTERRUPTED)
    with pytest.warns(DeprecationWarning, match="Protein.low_complexity_mask") as caught:
        # the old method's arguments, by position
        mask = protein.predictor.low_complexity_domains("Q", 5)
    # the warning points at the line that used the old name, not inside sparrow
    assert caught[0].filename == __file__
    assert mask == INTERRUPTED_MASK


def test_the_old_predictor_name_is_not_listed_among_the_predictions():
    predictor = Protein(SEQ).predictor
    assert "low_complexity_domains" not in dir(predictor)
    assert "predictor.low_complexity_domains" not in sparrow.batching._available_names()
    with pytest.raises(AttributeError, match="no_such_prediction"):
        predictor.no_such_prediction
