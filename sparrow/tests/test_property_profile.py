import numpy as np
import pytest

from sparrow import Protein
from sparrow.data import aaindex
from sparrow.sparrow_exceptions import ProteinException, SparrowException

SEQ = "ACDEFGHIKLMNPQRSTVWY" * 3  # length 60, all standard residues


def test_registry_unique_and_resolves():
    indices = aaindex.list_property_indices()
    assert len(indices) > 500
    identifiers = [i[0] for i in indices]
    # every identifier is unique
    assert len(set(identifiers)) == len(identifiers)
    # known index resolves by slug and by accession
    assert aaindex.resolve_identifier("hydropathy-kyte-1982") == "KYTJ820101"
    assert aaindex.resolve_identifier("KYTJ820101") == "KYTJ820101"
    # classic Kyte-Doolittle alanine value
    assert aaindex.get_property_values("hydropathy-kyte-1982")["A"] == 1.8


def test_collision_increment():
    # the Aurora 1998 positional indices share meaning+author+year -> numbered
    ids = [i[0] for i in aaindex.list_property_indices()
           if i[0].startswith("positional-aurora-1998")]
    assert "positional-aurora-1998-1" in ids
    assert len(ids) == len(set(ids)) and len(ids) > 1


def test_profile_matches_manual_window_mean():
    p = Protein(SEQ)
    values = aaindex.get_property_values("hydropathy-kyte-1982")
    window = 9

    raw = [
        np.mean([values[r] for r in SEQ[i:i + window]])
        for i in range(len(SEQ) - window + 1)
    ]
    raw = np.array(raw)
    front = window // 2
    expected = np.concatenate([[raw[0]] * front, raw])
    expected = np.concatenate([expected, [expected[-1]] * (len(SEQ) - len(expected))])

    profile = p.linear_property_profile("hydropathy-kyte-1982", window_size=window)
    assert len(profile) == len(SEQ)
    assert np.allclose(profile, expected)


def test_slug_and_accession_equivalent_and_cached():
    p = Protein(SEQ)
    by_slug = p.linear_property_profile("hydropathy-kyte-1982", window_size=9)
    by_accession = p.linear_property_profile("KYTJ820101", window_size=9)
    assert np.array_equal(by_slug, by_accession)


@pytest.mark.parametrize(
    "end_mode,expected_len",
    [("extend-ends", 60), ("zero-ends", 60), ("", 53)],
)
def test_end_mode_lengths(end_mode, expected_len):
    p = Protein(SEQ)
    profile = p.linear_property_profile(
        "hydropathy-kyte-1982", window_size=8, end_mode=end_mode
    )
    assert len(profile) == expected_len


def test_unknown_identifier_raises():
    p = Protein(SEQ)
    with pytest.raises(SparrowException):
        p.linear_property_profile("does-not-exist-2099")


def test_missing_value_index_raises():
    # find an index that has a None value for at least one residue
    miss_id = None
    for identifier, _accession, _desc in aaindex.list_property_indices():
        values = aaindex.get_property_values(identifier)
        if any(v is None for v in values.values()):
            miss_id = identifier
            break
    assert miss_id is not None
    p = Protein(SEQ)
    with pytest.raises(ProteinException):
        p.linear_property_profile(miss_id)
