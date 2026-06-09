"""Tests for the sparrow.data.aaindex registry."""

import pytest

from sparrow.data import aaindex
from sparrow.sparrow_exceptions import SparrowException


def test_lists_all_indices_with_unique_identifiers():
    rows = aaindex.list_property_indices()
    assert len(rows) > 500
    identifiers = [r[0] for r in rows]
    accessions = [r[1] for r in rows]
    assert len(set(identifiers)) == len(identifiers)
    assert len(set(accessions)) == len(accessions)


def test_resolve_slug_and_accession():
    assert aaindex.resolve_identifier("hydropathy-kyte-1982") == "KYTJ820101"
    assert aaindex.resolve_identifier("KYTJ820101") == "KYTJ820101"


def test_resolve_accession_case_insensitive():
    assert aaindex.resolve_identifier("kytj820101") == "KYTJ820101"


def test_slug_format_is_meaning_author_year():
    meta = aaindex.get_property_metadata("hydropathy-kyte-1982")
    assert meta["identifier"] == "hydropathy-kyte-1982"
    assert meta["accession"] == "KYTJ820101"
    assert "Kyte" in meta["authors"]
    assert meta["description"]


def test_get_values_has_twenty_amino_acids():
    values = aaindex.get_property_values("hydropathy-kyte-1982")
    assert set(values) == set("ACDEFGHIKLMNPQRSTVWY")
    assert values["A"] == 1.8  # classic Kyte-Doolittle alanine value


def test_collision_group_is_numbered():
    identifiers = {r[0] for r in aaindex.list_property_indices()}
    # the Aurora 1998 "positional" indices share meaning+author+year
    assert "positional-aurora-1998-1" in identifiers
    assert "positional-aurora-1998-2" in identifiers


def test_unknown_identifier_raises():
    with pytest.raises(SparrowException):
        aaindex.resolve_identifier("totally-made-up-9999")


def test_metadata_keys_present():
    meta = aaindex.get_property_metadata("KYTJ820101")
    for key in ("identifier", "accession", "description", "authors", "reference"):
        assert key in meta
