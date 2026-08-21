"""Tests for sparrow.sequence_analysis.alignment.

There were none, which is why `SequenceAlignment(...)` shipped raising
``ValueError`` on construction with default arguments: sparrow defaulted
``medoid_threshold`` to 0, and pyFAMSA rejects any threshold below its
``medoid_seeds`` (100). Nothing exercised the constructor, so nothing caught it.

pyfamsa is a hard dependency, so these are not guarded by importorskip.
"""

import pytest

from sparrow import Protein
from sparrow.sequence_analysis.alignment import SequenceAlignment

SEQUENCES = {
    "alpha": "MEEEKKKKSSSTTTDDDQQQ",
    "beta": "MEEEKKKASSSTTTDDDQQQ",
    "gamma": "MEEKKKKSSSTTTDD",
}


@pytest.fixture
def proteins():
    return {name: Protein(seq) for name, seq in SEQUENCES.items()}


def test_constructs_with_default_arguments(proteins):
    """The regression: defaults alone used to raise ValueError."""
    assert SequenceAlignment(proteins).aligner is not None


def test_default_leaves_pyfamsa_medoid_threshold_alone(proteins):
    """None must mean 'do not pass it', not 'pass 0'."""
    from pyfamsa import Aligner

    assert SequenceAlignment(proteins).aligner.medoid_threshold == Aligner().medoid_threshold


def test_explicit_medoid_threshold_is_forwarded(proteins):
    assert SequenceAlignment(proteins, medoid_threshold=500).aligner.medoid_threshold == 500


def test_msa_aligns_and_pads_to_equal_length(proteins):
    """Every aligned row must share one length, and gaps must appear."""
    msa = SequenceAlignment(proteins).construct_msa()
    rows = {seq.id.decode(): seq.sequence.decode() for seq in msa}
    assert set(rows) == set(SEQUENCES)
    assert len({len(r) for r in rows.values()}) == 1
    # gamma is the shortest input, so it must have been gapped
    assert "-" in rows["gamma"]
    # gaps aside, each row must still be its original sequence
    for name, row in rows.items():
        assert row.replace("-", "") == SEQUENCES[name]


def test_alignment_property_is_cached(proteins):
    sa = SequenceAlignment(proteins)
    assert sa.alignment is sa.alignment is sa.construct_msa()


def test_guide_tree_and_tree_heuristic_accept_documented_values(proteins):
    """The annotations must match what pyFAMSA actually accepts at runtime."""
    for tree in ("sl", "slink", "upgma", "nj"):
        assert SequenceAlignment(proteins, guide_tree=tree).aligner.guide_tree == tree
    for heuristic in (None, "medoid", "parttree"):
        sa = SequenceAlignment(proteins, medoid_threshold=200, tree_heuristic=heuristic)
        assert sa.aligner.tree_heuristic == heuristic


def test_save_msa_round_trips(proteins, tmp_path):
    import protfasta

    target = tmp_path / "msa.fasta"
    SequenceAlignment(proteins).save_msa(str(target))
    # alignment=True is required: the file contains gap characters, which
    # protfasta rejects under its default invalid_sequence_action="fail"
    written = protfasta.read_fasta(str(target), alignment=True)
    assert set(written) == set(SEQUENCES)
    assert len({len(v) for v in written.values()}) == 1
