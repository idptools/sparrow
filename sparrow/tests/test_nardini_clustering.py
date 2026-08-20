"""Tests for building your own GIN-style clusters.

The headline test is ``test_recovers_planted_grammars``: sequences are generated
with four deliberately different grammars, and the clustering has to put them back
into four groups. Everything else checks the guard rails.
"""

import random
import warnings

import numpy as np
import pytest

from sparrow.nardini import (
    assign_gin_clusters,
    build_gin_reference,
    compare_gin_references,
    describe_centroid,
    load_gin_reference,
    load_reference_background,
    nardini_vectors,
)
from sparrow.nardini.gin import GINReference
from sparrow.sparrow_exceptions import ClusteringQualityWarning, SparrowException

# these fixtures are small on purpose, so the "too few per cluster" advisory is
# expected noise; the tests that care about it assert it directly
pytestmark = pytest.mark.filterwarnings("ignore::sparrow.sparrow_exceptions.ClusteringQualityWarning")

PLANTED_GRAMMARS = {
    "DE_tract": lambda rng: "GS" + "DE" * rng.randint(30, 45) + "GSGS" * rng.randint(3, 8),
    "polyQ": lambda rng: "GS" + "Q" * rng.randint(50, 80) + "PSPS" * rng.randint(3, 8),
    "S_rich": lambda rng: "".join(rng.choice("SSSSTGSP") for _ in range(rng.randint(120, 180))),
    "K_blocks": lambda rng: "GS" + ("K" * 12 + "GSGS") * rng.randint(4, 7),
}


@pytest.fixture(scope="module")
def planted():
    """(NardiniResult, list of planted labels) for four distinct grammars."""
    rng = random.Random(3)
    sequences, labels = {}, []
    for label, maker in PLANTED_GRAMMARS.items():
        for index in range(12):
            sequences[f"{label}_{index}"] = maker(rng)
            labels.append(label)

    result = nardini_vectors(
        sequences,
        background=load_reference_background("human"),
        num_scrambles=300,
        seed=1,
    )
    return result, labels


@pytest.fixture(scope="module")
def diverse():
    """A compositionally bland set - no planted structure."""
    rng = random.Random(7)
    sequences = {
        f"s{i}": "".join(rng.choice("SSTTGGQQNNPPEEDDKKRRAAILVM") for _ in range(rng.randint(120, 260)))
        for i in range(40)
    }
    return nardini_vectors(
        sequences,
        background=load_reference_background("human"),
        num_scrambles=200,
        seed=1,
    )


# =====================================================================
# does it actually cluster?
# =====================================================================


def test_recovers_planted_grammars(planted):
    """Four deliberately different grammars must come back as four clusters."""
    result, labels = planted
    reference = build_gin_reference(result, k=4, seed=0)

    assignment = assign_gin_clusters(result, reference=reference)

    # every planted group should be dominated by a single cluster, and each group
    # should claim a different one
    dominant = {}
    for label in PLANTED_GRAMMARS:
        members = [
            int(cluster) for cluster, planted_label in zip(assignment.clusters, labels) if planted_label == label
        ]
        counts = np.bincount(members, minlength=4)
        dominant[label] = int(np.argmax(counts))
        assert counts.max() == len(members), f"{label} was split across clusters: {counts.tolist()}"

    assert len(set(dominant.values())) == 4, f"grammars collided: {dominant}"


def test_assignment_reproduces_the_clustering_labels(planted):
    """Nearest-centroid assignment must agree with the k-means partition."""
    result, _ = planted
    reference = build_gin_reference(result, k=4, seed=0)

    sizes = [reference.num_idrs[cluster] for cluster in range(4)]
    assignment = assign_gin_clusters(result, reference=reference)
    from_assignment = np.bincount(assignment.clusters, minlength=4)

    assert sizes == from_assignment.tolist()


def test_clusters_are_ordered_largest_first(planted):
    result, _ = planted
    reference = build_gin_reference(result, k=3, seed=0)

    sizes = [reference.num_idrs[cluster] for cluster in range(3)]
    assert sizes == sorted(sizes, reverse=True)


def test_seed_makes_it_reproducible(planted):
    result, _ = planted
    first = build_gin_reference(result, k=4, seed=11)
    second = build_gin_reference(result, k=4, seed=11)

    assert np.allclose(first.centroids, second.centroids)
    assert first.descriptions == second.descriptions


def test_metadata_records_how_it_was_built(planted):
    result, _ = planted
    reference = build_gin_reference(result, k=4, seed=5, clip=2.5)

    metadata = reference.metadata
    assert metadata["derived_from"] == "build_gin_reference"
    assert metadata["k"] == 4
    assert metadata["clip"] == 2.5
    assert metadata["seed"] == 5
    assert metadata["backend"] == "nardini"
    assert metadata["num_sequences"] == len(result)
    assert metadata["inertia"] > 0


# =====================================================================
# input validation
# =====================================================================


@pytest.mark.parametrize("k", [0, 1, -3])
def test_k_must_be_at_least_two(planted, k):
    result, _ = planted
    with pytest.raises(SparrowException, match="at least 2"):
        build_gin_reference(result, k=k)


def test_k_cannot_exceed_the_number_of_sequences(planted):
    result, _ = planted
    with pytest.raises(SparrowException, match="Cannot fit"):
        build_gin_reference(result, k=len(result) + 1)


def test_bare_array_needs_feature_names(planted):
    result, _ = planted
    with pytest.raises(SparrowException, match="feature_names"):
        build_gin_reference(result.zscores, k=3)


def test_bare_array_works_with_feature_names(planted):
    result, _ = planted
    reference = build_gin_reference(result.zscores, k=3, seed=0, feature_names=result.feature_names)
    assert len(reference) == 3
    assert reference.feature_names == tuple(result.feature_names)


def test_non_finite_vectors_are_rejected(planted):
    result, _ = planted
    broken = np.array(result.zscores, dtype=np.float64)
    broken[0, 0] = np.nan

    with pytest.raises(SparrowException, match="non-finite"):
        build_gin_reference(broken, k=3, feature_names=result.feature_names)


def test_mismatched_feature_names_rejected(planted):
    result, _ = planted
    with pytest.raises(SparrowException, match="feature names"):
        build_gin_reference(result.zscores, k=3, feature_names=["a", "b"])


# =====================================================================
# backend spaces must not be mixed
# =====================================================================


def test_reference_records_its_backend(diverse):
    kappa = nardini_vectors(
        {name: sequence for name, sequence in zip(diverse.names, diverse.sequences)},
        background=load_reference_background("human"),
        num_scrambles=200,
        seed=1,
        backend="kappa",
    )
    reference = build_gin_reference(kappa, k=3, seed=0)
    assert reference.backend == "kappa"


def test_own_backend_assigns_happily(diverse):
    """A kappa reference must accept kappa vectors - this is the point."""
    kappa = nardini_vectors(
        dict(zip(diverse.names, diverse.sequences)),
        background=load_reference_background("human"),
        num_scrambles=200,
        seed=1,
        backend="kappa",
    )
    reference = build_gin_reference(kappa, k=3, seed=0)

    assignment = assign_gin_clusters(kappa, reference=reference)
    assert len(assignment.clusters) == len(kappa)


def test_kappa_vectors_against_published_centroids_refused(diverse):
    kappa = nardini_vectors(
        dict(zip(diverse.names, diverse.sequences)),
        background=load_reference_background("human"),
        num_scrambles=200,
        seed=1,
        backend="kappa",
    )
    with pytest.raises(SparrowException, match="different scales"):
        assign_gin_clusters(kappa)


def test_nardini_vectors_against_a_kappa_reference_refused(diverse):
    kappa = nardini_vectors(
        dict(zip(diverse.names, diverse.sequences)),
        background=load_reference_background("human"),
        num_scrambles=200,
        seed=1,
        backend="kappa",
    )
    kappa_reference = build_gin_reference(kappa, k=3, seed=0)

    with pytest.raises(SparrowException, match="different scales"):
        assign_gin_clusters(diverse, reference=kappa_reference)


def test_published_reference_still_works(diverse):
    """The default path must be unchanged by the relaxed guard."""
    assignment = assign_gin_clusters(diverse)
    assert len(assignment.clusters) == len(diverse)
    assert np.all((assignment.clusters >= 0) & (assignment.clusters < 30))


# =====================================================================
# descriptions
# =====================================================================


def test_describe_centroid_uses_the_right_language():
    names = ["neg-neg", "pol-gly", "Frac E", "Frac K", "Q Patch"]

    assert "blocky neg-neg" in describe_centroid([2.5, 0, 0, 0, 0], names)
    assert "well-mixed pol-gly" in describe_centroid([0, -2.5, 0, 0, 0], names)
    assert "high Frac E" in describe_centroid([0, 0, 2.5, 0, 0], names)
    assert "low Frac K" in describe_centroid([0, 0, 0, -2.5, 0], names)
    # patch features read as a plural noun, not "high Q Patch"
    assert "Q patches" == describe_centroid([0, 0, 0, 0, 2.5], names)
    assert "few Q patches" == describe_centroid([0, 0, 0, 0, -2.5], names)


def test_describe_centroid_orders_by_magnitude_and_truncates():
    names = ["a-a", "Frac E", "Frac K"]
    described = describe_centroid([0.6, 3.0, -1.5], names, n=2)

    assert described.startswith("high Frac E")
    assert "low Frac K" in described
    assert "a-a" not in described


def test_describe_centroid_says_so_when_nothing_stands_out():
    assert describe_centroid([0.1, -0.2], ["a-a", "Frac E"]) == "no distinctive features"


def test_describe_centroid_validates_shape():
    with pytest.raises(SparrowException, match="1D centroid"):
        describe_centroid([[1.0, 2.0]], ["a", "b"])


def test_supplied_descriptions_win(planted):
    result, _ = planted
    reference = build_gin_reference(result, k=3, seed=0, descriptions={1: "my own label"})

    assert reference.describe(1) == "my own label"
    # the others are still derived
    assert reference.describe(0) != "my own label"


# =====================================================================
# comparing two references
# =====================================================================


def test_comparing_a_reference_to_itself_is_exact(planted):
    result, _ = planted
    reference = build_gin_reference(result, k=4, seed=0)

    report = compare_gin_references(reference, reference)

    assert report["num_pairs"] == 4
    assert report["mean_distance"] == pytest.approx(0.0, abs=1e-9)
    assert report["max_distance"] == pytest.approx(0.0, abs=1e-9)
    # matched one-to-one onto itself
    for pair in report["pairs"]:
        assert pair["cluster"] == pair["matched_cluster"]


def test_comparison_distances_share_the_module_convention(planted):
    """Distances must be squared Euclidean, like separation_scale and assignment.

    Mixing the two conventions would make mean_distance silently incomparable to
    separation_scale, which is the only thing that makes it interpretable.
    """
    result, _ = planted
    first = build_gin_reference(result, k=3, seed=0)
    second = build_gin_reference(result, k=3, seed=99)

    report = compare_gin_references(first, second)

    expected = np.sum((first.centroids[0] - second.centroids[0]) ** 2)
    distances = [pair["distance"] for pair in report["pairs"]]
    # the pairing is optimal, so cluster 0's distance is at most the naive pairing
    assert min(distances) <= expected + 1e-9


def test_comparison_reports_both_separation_scales(planted):
    result, _ = planted
    reference = build_gin_reference(result, k=4, seed=0)

    report = compare_gin_references(reference)

    assert report["separation_scale"] == pytest.approx(load_gin_reference().separation_scale)
    assert report["reference_separation_scale"] == pytest.approx(reference.separation_scale)
    # 4 of my clusters paired against 30 published ones leaves 26 unmatched
    assert report["num_pairs"] == 4
    assert report["unmatched"] == 26


def test_comparison_rejects_different_feature_spaces(planted):
    result, _ = planted
    reference = build_gin_reference(result, k=3, seed=0)
    narrow = GINReference(
        feature_names=["a", "b"],
        clusters=[0, 1],
        centroids=[[0.0, 0.0], [1.0, 1.0]],
    )

    with pytest.raises(SparrowException, match="feature spaces"):
        compare_gin_references(reference, narrow)


def test_comparison_rejects_different_backends(planted):
    result, _ = planted
    reference = build_gin_reference(result, k=3, seed=0)
    other = build_gin_reference(result, k=3, seed=0)
    other.backend = "kappa"

    with pytest.raises(SparrowException, match="different scales"):
        compare_gin_references(reference, other)


# =====================================================================
# persistence
# =====================================================================


def test_reference_save_load_roundtrip(tmp_path, planted):
    result, _ = planted
    reference = build_gin_reference(result, k=4, seed=0)

    path = reference.save(tmp_path / "mine")
    loaded = GINReference.load(path)

    assert np.array_equal(reference.centroids, loaded.centroids)
    assert np.array_equal(reference.clusters, loaded.clusters)
    assert reference.descriptions == loaded.descriptions
    assert reference.num_idrs == loaded.num_idrs
    assert reference.metadata == loaded.metadata
    assert reference.backend == loaded.backend
    assert reference.feature_names == loaded.feature_names


def test_saved_reference_is_still_usable(tmp_path, planted):
    result, _ = planted
    reference = build_gin_reference(result, k=4, seed=0)
    loaded = GINReference.load(reference.save(tmp_path / "mine"))

    first = assign_gin_clusters(result, reference=reference)
    second = assign_gin_clusters(result, reference=loaded)

    assert np.array_equal(first.clusters, second.clusters)
    assert np.allclose(first.distances, second.distances)


def test_save_adds_the_extension(tmp_path, planted):
    result, _ = planted
    reference = build_gin_reference(result, k=3, seed=0)

    assert reference.save(tmp_path / "noext").endswith(".npz")


def test_load_rejects_a_missing_file():
    with pytest.raises(SparrowException, match="No such GIN reference"):
        GINReference.load("/nonexistent/reference.npz")


def test_load_rejects_an_unrelated_npz(tmp_path):
    path = tmp_path / "wrong.npz"
    np.savez(path, something_else=np.arange(3))

    with pytest.raises(SparrowException, match="saved GIN reference"):
        GINReference.load(path)


# =====================================================================
# the quality advisory
# =====================================================================


def test_warns_when_clusters_would_be_nearly_empty(planted):
    """48 sequences into 8 clusters is 6 each, below the floor of 10."""
    result, _ = planted

    with pytest.warns(ClusteringQualityWarning, match="per cluster"):
        build_gin_reference(result, k=8, seed=0)


def test_unfittable_k_raises_rather_than_warning(planted):
    """k so large that every restart leaves an empty cluster is an error."""
    result, _ = planted

    with pytest.raises(SparrowException, match="could not fit"):
        build_gin_reference(result, k=len(result) - 1, seed=0)


def test_does_not_warn_with_enough_sequences_per_cluster(planted):
    result, _ = planted

    with warnings.catch_warnings():
        warnings.simplefilter("error", ClusteringQualityWarning)
        build_gin_reference(result, k=4, seed=0)
