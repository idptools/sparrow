"""Tests for sparrow.sequence_analysis.sequence_complexity (LCD extraction)."""

import pytest

from sparrow import Protein
from sparrow.sequence_analysis import sequence_complexity as sc

# a Q-rich block flanked by alanine
SEQ = "AAAAA" + "Q" * 16 + "AAAAA"


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
    strict = sc.low_complexity_domains_holt_permissive(
        SEQ, "Q", minimum_length=10, fractional_threshold=1.01
    )
    lenient = sc.low_complexity_domains_holt_permissive(
        SEQ, "Q", minimum_length=10, fractional_threshold=0.1
    )
    assert strict == []
    assert len(lenient) >= 1


def test_protein_low_complexity_domains_wrapper():
    p = Protein(SEQ)
    lcds = p.low_complexity_domains(mode="holt", residue_selector="Q", minimum_length=10)
    assert len(lcds) == 1


def test_protein_low_complexity_domains_permissive_mode():
    p = Protein(SEQ)
    lcds = p.low_complexity_domains(
        mode="holt-permissive", residue_selector="Q", minimum_length=10
    )
    assert isinstance(lcds, list)


def test_protein_low_complexity_domains_invalid_mode_raises():
    p = Protein(SEQ)
    with pytest.raises(Exception):
        p.low_complexity_domains(mode="not-a-mode", residue_selector="Q")
