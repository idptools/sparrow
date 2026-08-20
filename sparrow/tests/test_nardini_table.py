"""Tests for the labelled containers and the Protein.nardini accessor."""

import numpy as np
import pytest

from sparrow import Protein
from sparrow.nardini import (
    FeatureTable,
    FeatureVector,
    gin_cluster_definitions,
    load_reference_background,
    nardini_vectors,
)
from sparrow.sparrow_exceptions import SparrowException

HUMAN_IDR = "MENDPRVRKFASDGAQWAIKWQKKGWSTLTSRQKQTARAAMGIKLSPVAQPVQKVTRLSAPVALAYREVSTQPRVSTARD"
POLY_Q = "Q" * 120

# The synthetic fixtures here are deliberately small, so build_background warns
# that they are weak. That warning is asserted directly in the dedicated tests
# below (pytest.warns overrides this filter); everywhere else it is just noise.
pytestmark = pytest.mark.filterwarnings("ignore::sparrow.sparrow_exceptions.BackgroundQualityWarning")


# =====================================================================
# FeatureVector
# =====================================================================


def test_feature_vector_basics():
    v = FeatureVector([1.0, -2.0, 0.5], ["a", "b", "c"], label="seq")

    assert len(v) == 3
    assert v["b"] == -2.0
    assert v[0] == 1.0
    assert "a" in v and "z" not in v
    assert dict(v) == {"a": 1.0, "b": -2.0, "c": 0.5}
    assert v.to_dict() == {"a": 1.0, "b": -2.0, "c": 0.5}
    assert np.array_equal(v.to_numpy(), np.array([1.0, -2.0, 0.5]))
    assert "seq" in repr(v)


def test_feature_vector_get_and_dict():
    v = FeatureVector([1.0, 2.0, 3.0], ["a", "b", "c"])

    assert v.to_dict() == {"a": 1.0, "b": 2.0, "c": 3.0}
    assert np.array_equal(v.to_numpy(), np.array([1.0, 2.0, 3.0]))
    # the raw array stays reachable for anything sparrow does not do
    assert v.values.mean() == 2.0


def test_feature_vector_unknown_name_raises():
    v = FeatureVector([1.0], ["a"])
    with pytest.raises(SparrowException, match="Unknown feature"):
        v["nope"]


def test_feature_vector_validates_shape():
    with pytest.raises(SparrowException, match="1D"):
        FeatureVector([[1.0, 2.0]], ["a", "b"])
    with pytest.raises(SparrowException, match="values but"):
        FeatureVector([1.0, 2.0], ["a"])


# =====================================================================
# FeatureTable
# =====================================================================


@pytest.fixture
def table():
    return FeatureTable(
        [[1.0, -2.0, 0.0], [3.0, 0.5, -4.0]],
        ["s1", "s2"],
        ["a", "b", "c"],
    )


def test_feature_table_basics(table):
    assert table.shape == (2, 3)
    assert len(table) == 2
    assert np.array_equal(table["a"], np.array([1.0, 3.0]))
    assert table["s2", "c"] == -4.0
    assert "b" in table


def test_feature_table_row(table):
    row = table.row("s2")
    assert isinstance(row, FeatureVector)
    assert row.label == "s2"
    assert row["a"] == 3.0
    # positional access agrees
    assert np.array_equal(table.row(1).values, row.values)


def test_feature_table_iterates_rows(table):
    labels = [row.label for row in table]
    assert labels == ["s1", "s2"]


def test_feature_table_validates_shape():
    with pytest.raises(SparrowException, match="does not match"):
        FeatureTable([[1.0, 2.0]], ["s1"], ["a", "b", "c"])


def test_feature_table_to_dict_and_numpy(table):
    assert table.to_dict()["s1"]["b"] == -2.0
    assert table.to_numpy().shape == (2, 3)
    # the raw array stays reachable for anything sparrow does not do
    assert table.values.max() == 3.0


def test_feature_table_bad_index(table):
    with pytest.raises(SparrowException, match="Index a FeatureTable"):
        table[3.5]
    with pytest.raises(SparrowException, match="Unknown row"):
        table.row("nope")


# =====================================================================
# NardiniResult views
# =====================================================================


def test_nardini_result_to_table(table):
    result = nardini_vectors(
        {"a": HUMAN_IDR},
        background=load_reference_background("human"),
        num_scrambles=100,
        seed=1,
    )
    output = result.to_table()

    assert isinstance(output, FeatureTable)
    assert output.shape == (1, 90)
    assert output.row_names == ("a",)
    assert output.column_names[0] == "pol-pol"
    assert output.column_names[-1] == "RG Frac"


def test_nardini_result_vector_and_raw():
    result = nardini_vectors(
        {"a": HUMAN_IDR},
        background=load_reference_background("human"),
        num_scrambles=100,
        seed=1,
    )
    z = result.vector("a")
    raw = result.vector("a", raw=True)

    assert isinstance(z, FeatureVector)
    assert len(z) == 90 and len(raw) == 90
    # raw patch fractions are bounded 0-1; their z-scores are not
    assert 0.0 <= raw["RG Frac"] <= 1.0


def test_gin_cluster_definitions_default_is_plain_python():
    definitions = gin_cluster_definitions()
    assert isinstance(definitions, list)
    assert isinstance(definitions[0], dict)


def test_to_dataframe_is_available():
    """pandas is the opt-in view; the labelled table is the default."""
    result = nardini_vectors(
        {"a": HUMAN_IDR},
        background=load_reference_background("human"),
        num_scrambles=100,
        seed=1,
    )
    frame = result.to_dataframe()
    assert frame.shape == (1, 90)
    assert list(frame.index) == ["a"]


# =====================================================================
# Protein.nardini
# =====================================================================


def test_protein_nardini_accessor_is_cached():
    p = Protein(HUMAN_IDR)
    assert p.nardini is p.nardini
    assert "ProteinNardini" in repr(p.nardini)


def test_protein_nardini_vector():
    p = Protein(HUMAN_IDR)
    v = p.nardini.vector(seed=1, num_scrambles=200)

    assert isinstance(v, FeatureVector)
    assert len(v) == 90
    assert v.to_numpy().shape == (90,)
    # both blocks are present and addressable by name
    assert "pol-neg" in v
    assert "Frac K+R" in v


def test_protein_nardini_vector_without_background_is_patterning_only():
    p = Protein(HUMAN_IDR)
    v = p.nardini.vector(background=None, seed=1, num_scrambles=200)
    assert len(v) == 36


def test_protein_nardini_patterning_needs_no_background():
    p = Protein(HUMAN_IDR)
    pat = p.nardini.patterning(seed=1, num_scrambles=200)

    assert len(pat) == 36
    # and matches the patterning block of the full vector
    full = p.nardini.vector(seed=1, num_scrambles=200)
    assert np.allclose(pat.to_numpy(), full.to_numpy()[:36])


def test_protein_nardini_composition_raw_and_z():
    p = Protein(HUMAN_IDR)
    z = p.nardini.composition()
    raw = p.nardini.composition(raw=True)

    assert len(z) == 54 and len(raw) == 54
    assert 0.0 <= raw["FCR"] <= 1.0
    assert z["FCR"] != raw["FCR"]
    # background=None is equivalent to raw
    assert np.allclose(p.nardini.composition(background=None).to_numpy(), raw.to_numpy())


def test_protein_nardini_results_are_memoized():
    p = Protein(HUMAN_IDR)
    first = p.nardini.vector(seed=1, num_scrambles=200)
    second = p.nardini.vector(seed=1, num_scrambles=200)
    assert np.array_equal(first.to_numpy(), second.to_numpy())


def test_protein_nardini_accepts_species_and_background_object():
    p = Protein(HUMAN_IDR)
    by_name = p.nardini.composition(background="human")
    by_object = p.nardini.composition(background=load_reference_background("human"))
    assert np.allclose(by_name.to_numpy(), by_object.to_numpy())

    # a different species gives different composition z-scores
    yeast = p.nardini.composition(background="yeast")
    assert not np.allclose(by_name.to_numpy(), yeast.to_numpy())


def test_protein_nardini_rejects_bad_background():
    p = Protein(HUMAN_IDR)
    with pytest.raises(SparrowException, match="background must be"):
        p.nardini.composition(background=12345)


# =====================================================================
# Protein.nardini GIN, and the typicality diagnostics
# =====================================================================


def test_protein_gin_cluster():
    p = Protein(HUMAN_IDR)
    record = p.nardini.gin_cluster(seed=42, num_scrambles=500)

    assert set(record) == {
        "name",
        "cluster",
        "description",
        "distance",
        "separation",
        "n_saturated",
        "composition_rms_z",
        "is_typical",
    }
    assert 0 <= record["cluster"] < 30
    assert isinstance(record["description"], str)


def test_protein_gin_cluster_requires_a_background():
    p = Protein(HUMAN_IDR)
    with pytest.raises(SparrowException, match="background is required"):
        p.nardini.gin_cluster(background=None)


def test_human_like_sequence_is_flagged_typical():
    """A real human IDR should pass the typicality screen."""
    record = Protein(HUMAN_IDR).nardini.gin_cluster(seed=42, num_scrambles=500)

    assert record["composition_rms_z"] < 2.0
    assert record["n_saturated"] == 0
    assert record["is_typical"] is True


def test_extreme_sequence_is_flagged_atypical():
    """A 120-residue poly-Q is far outside the human IDRome."""
    record = Protein(POLY_Q).nardini.gin_cluster(seed=42, num_scrambles=500)

    assert record["composition_rms_z"] > 2.0
    assert record["is_typical"] is False
    # the nearest cluster is still semantically sensible - it is the confidence
    # that is the problem, not the label
    assert record["description"] == "Q-tracts"


def test_composition_rms_z_separates_idr_like_from_extreme():
    """The diagnostic must actually discriminate, not just exist."""
    import random

    rng = random.Random(0)
    idr_alphabet = "SSSTTTGGGQQQNNNPPPEEEDDDKKKRRRAAALLLVVVMIYFWHC"

    idr_like = ["".join(rng.choice(idr_alphabet) for _ in range(150)) for _ in range(4)]
    extreme = ["Q" * 120, "E" * 120, "W" * 120, "LLLIIIVVVMMM" * 10]

    def rms(sequence):
        return Protein(sequence).nardini.gin_cluster(seed=1, num_scrambles=100)["composition_rms_z"]

    idr_scores = [rms(s) for s in idr_like]
    extreme_scores = [rms(s) for s in extreme]

    assert max(idr_scores) < 2.0
    assert min(extreme_scores) > 2.0


def test_gin_report_mentions_the_caveat_for_extreme_sequences():
    report = Protein(POLY_Q).nardini.gin_report(seed=42, num_scrambles=500)

    assert "CAUTION" in report
    assert "extrapolation" in report
    # and it should say the patterning half is still fine
    assert "patterning" in report


def test_gin_report_is_clean_for_typical_sequences():
    report = Protein(HUMAN_IDR).nardini.gin_report(seed=42, num_scrambles=500)

    assert "CAUTION" not in report
    assert "label is usable" in report
    assert "nearest alternatives" in report


# =====================================================================
# the Protein.nardini memoization key
# =====================================================================


def _background_from(letters, seed, length=60, count=20):
    """Build a small background over sequences drawn from `letters`."""
    from sparrow.nardini import build_background

    rng = np.random.default_rng(seed)
    return build_background({f"s{i}": "".join(rng.choice(list(letters), length)) for i in range(count)})


def test_background_cache_key_is_content_based():
    """Two backgrounds with identical statistics must share a cache entry."""
    from sparrow.nardini.protein_interface import _background_key

    same_a = _background_from("STGQNPEDKRA", 99)
    same_b = _background_from("STGQNPEDKRA", 99)
    different = _background_from("ILMVFWYAC", 99)

    assert _background_key(same_a) == _background_key(same_b)
    assert _background_key(same_a) != _background_key(different)
    # species names and None keep keying on themselves
    assert _background_key("human") == "human"
    assert _background_key(None) is None


def test_background_cache_key_normalizes_pathlike():
    from pathlib import Path

    from sparrow.nardini.protein_interface import _background_key

    assert _background_key(Path("/tmp/bg.npz")) == _background_key("/tmp/bg.npz")


def test_protein_nardini_cache_survives_background_id_reuse():
    """A collected background must not poison the cache for the next one.

    The key used to be ``id(background)``. Because the cache stores only the key
    and holds no reference to the background, dropping the caller's reference frees
    the object and CPython reallocates the next background at the same address -
    silently returning the previous background's z-scores. This reproduced within a
    handful of allocations and was observed reporting +19.8 where the correct value
    was -4.7.

    The loop deliberately churns allocations to provoke address reuse. If the
    platform never reuses an address the test still passes, which is why
    test_background_cache_key_is_content_based carries the real weight.
    """
    import gc

    from sparrow.nardini import nardini_vectors

    protein = Protein(HUMAN_IDR)
    seen = set()
    reused = 0

    for attempt in range(40):
        letters = "STGQNPEDKRA" if attempt % 2 == 0 else "ILMVFWYAC"
        background = _background_from(letters, attempt)

        got = protein.nardini.vector(background=background, seed=1, num_scrambles=100).to_numpy()
        expected = nardini_vectors(
            {"sequence": HUMAN_IDR},
            background=background,
            num_scrambles=100,
            seed=1,
        ).zscores[0]

        reused += id(background) in seen
        seen.add(id(background))

        assert np.allclose(got, expected), f"stale cache hit on attempt {attempt} (address reuse #{reused})"

        del background
        gc.collect()


# =====================================================================
# the patterning half is background-independent, and cached as such
# =====================================================================


def test_patterning_block_is_independent_of_the_background():
    """The premise behind reusing a cached full result for patterning()."""
    p = Protein(HUMAN_IDR)
    options = dict(seed=1, num_scrambles=300)

    human = p.nardini.vector(background="human", **options).to_numpy()[:36]
    yeast = p.nardini.vector(background="yeast", **options).to_numpy()[:36]
    alone = p.nardini.patterning(**options).to_numpy()

    assert np.array_equal(human, yeast)
    assert np.array_equal(human, alone)


def test_patterning_reuses_a_cached_full_result(monkeypatch):
    """patterning() must not repeat the scrambling already done for a full vector.

    Its cache key uses background=None, so before this it missed whenever the
    cached result had been computed with a real background - re-running the
    scrambling, which is essentially the whole cost. That is exactly the situation
    after nardini_proteins() has populated a proteome.
    """
    p = Protein(HUMAN_IDR)
    options = dict(seed=1, num_scrambles=300)

    expected = p.nardini.vector(background="human", **options).to_numpy()[:36]

    def explode(*args, **kwargs):
        raise AssertionError("patterning() recomputed instead of reusing the cache")

    monkeypatch.setattr("sparrow.nardini.vectors.nardini_vectors", explode)

    assert np.array_equal(p.nardini.patterning(**options).to_numpy(), expected)


def test_patterning_still_recomputes_for_different_options(monkeypatch):
    """The reuse must key on the patterning options, not ignore them."""
    p = Protein(HUMAN_IDR)
    p.nardini.vector(background="human", seed=1, num_scrambles=300)

    calls = []
    original = nardini_vectors

    def counted(*args, **kwargs):
        calls.append(kwargs.get("num_scrambles"))
        return original(*args, **kwargs)

    monkeypatch.setattr("sparrow.nardini.vectors.nardini_vectors", counted)

    p.nardini.patterning(seed=1, num_scrambles=400)
    assert calls == [400], "a different scramble count must not reuse the cache"
