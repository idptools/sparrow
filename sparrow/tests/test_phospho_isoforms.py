"""Tests for phosphoisoform generation."""

import pytest

from sparrow import Protein
from sparrow.sequence_analysis import phospho_isoforms

SEQ = "MSRSTPYK"  # contains S, T, Y phosphosites


def test_get_phosphoisoforms_all_mode_returns_list():
    p = Protein(SEQ)
    isoforms = phospho_isoforms.get_phosphoisoforms(p, mode="all")
    assert isinstance(isoforms, list)
    assert SEQ in isoforms  # the unmodified sequence is one isoform


def test_phosphomimetic_is_glutamate():
    p = Protein(SEQ)
    isoforms = phospho_isoforms.get_phosphoisoforms(p, mode="all")
    # every isoform differs from wild-type only by S/T/Y -> E substitutions
    for iso in isoforms:
        assert len(iso) == len(SEQ)
        for wt, mod in zip(SEQ, iso):
            if wt != mod:
                assert wt in "STY" and mod == "E"


def test_protein_wrapper_all_mode():
    # regression: Protein.generate_phosphoisoforms previously passed a string
    # instead of the Protein object and raised AttributeError.
    p = Protein(SEQ)
    isoforms = p.generate_phosphoisoforms(mode="all")
    assert isinstance(isoforms, list)
    assert len(isoforms) > 1


def test_protein_wrapper_custom_mode():
    p = Protein(SEQ)
    # phosphosite at index 1 (the S) -> isoforms include the E substitution there
    isoforms = p.generate_phosphoisoforms(mode="custom", phosphosites=[1])
    assert any(iso[1] == "E" for iso in isoforms)


def test_custom_mode_requires_phosphosites():
    p = Protein(SEQ)
    with pytest.raises(Exception):
        p.generate_phosphoisoforms(mode="custom", phosphosites=None)


def test_invalid_mode_raises():
    p = Protein(SEQ)
    with pytest.raises(Exception):
        p.generate_phosphoisoforms(mode="not-a-mode")
