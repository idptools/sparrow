"""Tests for sparrow.visualize.sequence_visuals.show_sequence."""

import pytest

from sparrow import Protein
from sparrow.sparrow_exceptions import SparrowException
from sparrow.visualize.sequence_visuals import show_sequence

SEQ = "ACDEFGHIKL"


def test_returns_html_string():
    html = show_sequence(SEQ, return_raw_string=True)
    assert isinstance(html, str)
    assert html.startswith("<p")
    assert html.endswith("</p>")
    assert "<span" in html


def test_header_is_included():
    html = show_sequence(SEQ, header="my header", return_raw_string=True)
    assert "my header" in html
    # the malformed stray '>' before the header bold tag must not be present
    assert '">><b>' not in html


def test_blocksize_below_one_raises():
    with pytest.raises(SparrowException):
        show_sequence(SEQ, blocksize=0, return_raw_string=True)


def test_mutable_default_arguments_do_not_leak():
    # calling with defaults must not accumulate state across calls
    first = show_sequence(SEQ, return_raw_string=True)
    second = show_sequence(SEQ, return_raw_string=True)
    assert first == second


def test_bold_residues_render_bold():
    html = show_sequence("EEEE", bold_residues=["E"], return_raw_string=True)
    assert "<b>E</b>" in html


def test_protein_show_sequence_wrapper():
    html = Protein(SEQ).show_sequence(return_raw_string=True)
    assert isinstance(html, str) and "<span" in html
