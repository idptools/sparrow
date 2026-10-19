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
# max_interruption allows: two domains in 'holt', and one in 'holt-permissive',
# which has no max_interruption and finds the whole region 65% Q
TWO_DOMAINS = "AAAAA" + "Q" * 12 + "A" * 8 + "QQAQQAQQAQQAQQ" + "AAA"

# two runs of eight Qs, split by eight As
SPLIT_RUNS = "Q" * 8 + "A" * 8 + "Q" * 8


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


def test_holt_splits_at_a_run_longer_than_max_interruption():
    assert sc.low_complexity_domains_holt(SPLIT_RUNS, "Q", minimum_length=5) == [
        ["QQQQQQQQ", 0, 8],
        ["QQQQQQQQ", 16, 24],
    ]


def test_holt_moves_on_when_nothing_passes_from_a_stretch_s_first_residue():
    # the lone Q is in the run's stretch, and nothing from it is 80% Q
    assert sc.low_complexity_domains_holt(
        "Q" + "AAAAA" + "Q" * 6, "Q", minimum_length=5, fractional_threshold=0.8
    ) == [["QQQQQQ", 6, 12]]


def test_holt_searches_the_rest_of_a_stretch_after_a_domain():
    # the lone Q between the runs puts both in one stretch
    sequence = "Q" * 6 + "AAAAA" + "Q" + "AAAAA" + "Q" * 6
    assert sc.low_complexity_domains_holt(sequence, "Q", minimum_length=5, fractional_threshold=0.8) == [
        ["QQQQQQ", 0, 6],
        ["QQQQQQ", 17, 23],
    ]


def test_holt_refuses_a_negative_max_interruption():
    with pytest.raises(SparrowException, match="max_interruption must be 0 or more"):
        sc.low_complexity_domains_holt(SEQ, "Q", max_interruption=-1)


def test_permissive_runs_across_any_interruption_the_threshold_allows():
    # 16 Q in 24 residues is 67% Q, above the default threshold of 0.25
    assert sc.low_complexity_domains_holt_permissive(SPLIT_RUNS, "Q", minimum_length=5) == [[SPLIT_RUNS, 0, 24]]


def test_permissive_applies_the_fractional_threshold():
    # 67% Q is below 0.7, so the runs stay apart
    assert sc.low_complexity_domains_holt_permissive(SPLIT_RUNS, "Q", minimum_length=5, fractional_threshold=0.7) == [
        ["QQQQQQQQ", 0, 8],
        ["QQQQQQQQ", 16, 24],
    ]


def test_permissive_tries_the_next_chosen_residue_when_nothing_passes():
    # nothing from the first Q is 90% Q, but the run of eight after it is
    sequence = "Q" + "AAAA" + "Q" * 8
    assert sc.low_complexity_domains_holt_permissive(sequence, "Q", minimum_length=5, fractional_threshold=0.9) == [
        ["QQQQQQQQ", 5, 13]
    ]


@pytest.mark.parametrize("mode", ["holt", "holt-permissive"])
def test_a_lone_chosen_residue_is_never_a_domain(mode):
    protein = Protein("AAQAA")
    assert (
        protein.low_complexity_domains(mode=mode, residue_selector="Q", minimum_length=1, fractional_threshold=0) == []
    )


def test_permissive_refuses_max_interruption_by_name_or_position():
    with pytest.raises(SparrowException, match="no max_interruption"):
        sc.low_complexity_domains_holt_permissive(SPLIT_RUNS, "Q", max_interruption=5)
    with pytest.raises(SparrowException, match="no max_interruption"):
        sc.low_complexity_domains_holt_permissive(SPLIT_RUNS, "Q", 5, 5)
    with pytest.raises(SparrowException, match="no max_interruption"):
        Protein(SPLIT_RUNS).low_complexity_mask(mode="holt-permissive", residue_selector="Q", max_interruption=5)


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


@pytest.mark.parametrize(("mode", "number_of_domains"), [("holt", 2), ("holt-permissive", 1)])
def test_the_mask_is_one_exactly_inside_the_domains(mode, number_of_domains):
    protein = Protein(TWO_DOMAINS)
    domains = protein.low_complexity_domains(mode=mode, residue_selector="Q", minimum_length=10)
    mask = protein.low_complexity_mask(mode=mode, residue_selector="Q", minimum_length=10)
    assert len(domains) == number_of_domains
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
