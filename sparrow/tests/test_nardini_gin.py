"""Tests for GIN cluster assignment and the bundled NARDINI reference data.

As in ``test_nardini.py``, the assignment algorithm from the reference notebook is
transcribed here (``_reference_assign``) and used as an oracle, so any change to
the production implementation that would alter a published cluster call fails
loudly.
"""

import numpy as np
import pandas as pd
import pytest

from sparrow.nardini import (
    GIN_CLIP,
    REFERENCE_SPECIES,
    assign_gin_clusters,
    composition_feature_names,
    gin_cluster_definitions,
    load_gin_reference,
    load_reference_background,
    nardini_vectors,
    patterning_feature_names,
)
from sparrow.sparrow_exceptions import SparrowException

# The synthetic fixtures here are deliberately small, so build_background warns
# that they are weak. That warning is asserted directly in the dedicated tests
# below (pytest.warns overrides this filter); everywhere else it is just noise.
pytestmark = pytest.mark.filterwarnings("ignore::sparrow.sparrow_exceptions.BackgroundQualityWarning")


# sequences designed to land in specific published clusters
NOTEBOOK_IDR = "MENDPRVRKFASDGAQWAIKWQKKGWSTLTSRQKQTARAAMGIKLSPVAQPVQKVTRLSAPVALAYREVSTQPRVSTARD"
DE_TRACT = "EEEEEEDDDDEEEEEEDDDDEEEEEEDDDDEEEEEEDDDD"
S_PATCH = "SSSSSSSSSSGGSSSSSSSSSSGGSSSSSSSSSSGGSSSS"
Q_TRACT = "QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ"
K_BLOCK = "KKKKKKKKKKSSSSKKKKKKKKKKSSSSKKKKKKKKKKSS"


# =====================================================================
# Reference implementation, transcribed from notebook cell 8
# =====================================================================


def _reference_assign(zvec, centroids):
    """Return ``(cluster, min_inter_cluster_distance, distances)``."""
    currzvecaa = list(zvec)

    pos = [i for i, ltr in enumerate(currzvecaa) if ltr > 3]
    for p in pos:
        currzvecaa[p] = 3
    pos = [i for i, ltr in enumerate(currzvecaa) if ltr < -3]
    for p in pos:
        currzvecaa[p] = -3

    distances = []
    for c in range(0, len(centroids)):
        distances.append(np.nansum((centroids[c] - currzvecaa) ** 2))

    minval = min(distances)
    tmpdiff = []
    for m in range(0, len(distances)):
        if distances[m] != minval:
            tmpdiff.append(distances[m] - minval)
        else:
            tmpdiff.append(100)

    return pd.Series(distances).idxmin(), min(tmpdiff), np.array(distances)


# =====================================================================
# bundled reference backgrounds
# =====================================================================


@pytest.mark.parametrize("species", sorted(REFERENCE_SPECIES))
def test_reference_backgrounds_load(species):
    background = load_reference_background(species)

    assert len(background) == 54
    assert tuple(background.feature_names) == composition_feature_names()
    assert np.all(np.isfinite(background.mean))
    assert np.all(np.isfinite(background.std))
    assert np.all(background.std >= 0)
    # the published tables do not record how many IDRs went into them
    assert background.num_sequences is None
    assert background.metadata["species"] == REFERENCE_SPECIES[species]
    assert "citation" in background.metadata


def test_reference_background_accepts_latin_name():
    by_key = load_reference_background("human")
    by_latin = load_reference_background("Homo sapiens")
    assert np.array_equal(by_key.mean, by_latin.mean)


def test_reference_background_is_cached():
    assert load_reference_background("human") is load_reference_background("human")


def test_reference_background_unknown_species_raises():
    with pytest.raises(SparrowException, match="No bundled NARDINI background"):
        load_reference_background("tardigrade")


def test_reference_background_zero_variance_features_are_safe():
    """Yeast and worm genuinely have no C patches, so std is exactly 0 there."""
    from sparrow.nardini import composition_values

    for species in ("yeast", "worm"):
        background = load_reference_background(species)
        index = list(background.feature_names).index("C Patch")
        assert background.std[index] == 0.0

        z = background.zscores(composition_values(NOTEBOOK_IDR))
        assert np.all(np.isfinite(z))
        assert z[index] == 0.0


def test_reference_background_works_end_to_end():
    result = nardini_vectors(
        {"a": NOTEBOOK_IDR},
        background=load_reference_background("human"),
        num_scrambles=200,
        seed=1,
    )
    assert result.zscores.shape == (1, 90)
    assert np.all(np.isfinite(result.zscores))


# =====================================================================
# GIN reference data
# =====================================================================


def test_gin_reference_shape_and_metadata():
    reference = load_gin_reference()

    assert len(reference) == 30
    assert reference.centroids.shape == (30, 90)
    assert list(reference.clusters) == list(range(30))
    assert reference.metadata["species"] == "Homo sapiens"
    assert reference.metadata["clip"] == 3.0
    assert "Ruff" in reference.metadata["citation"]


def test_gin_reference_feature_order_matches_sparrow():
    """The centroids are only meaningful if our feature order matches theirs."""
    reference = load_gin_reference()
    expected = list(patterning_feature_names("8")) + list(composition_feature_names())
    assert list(reference.feature_names) == expected


def test_gin_reference_is_cached():
    assert load_gin_reference() is load_gin_reference()


def test_gin_cluster_definitions():
    """The default return is plain dictionaries - no pandas required."""
    definitions = gin_cluster_definitions()

    assert isinstance(definitions, list)
    assert len(definitions) == 30
    assert [d["cluster"] for d in definitions] == list(range(30))
    assert definitions[7]["description"] == "D/E-tracts"
    assert definitions[6]["description"] == "S patches"
    assert definitions[11]["description"] == "Q-tracts"
    # the published analysis assigned 24,508 human IDRs across the 30 clusters
    assert sum(d["num_idrs"] for d in definitions) == 24508


def test_gin_cluster_definitions_as_dataframe():
    frame = gin_cluster_definitions(as_dataframe=True)

    assert list(frame.index) == list(range(30))
    assert frame.loc[7, "description"] == "D/E-tracts"
    assert frame["num_idrs"].sum() == 24508


def test_gin_reference_describe():
    reference = load_gin_reference()
    assert reference.describe(7) == "D/E-tracts"
    assert "no description" in reference.describe(999)


# =====================================================================
# assignment -- agreement with the reference implementation
# =====================================================================


@pytest.fixture(scope="module")
def gin_result():
    """NARDINI+ vectors for a spread of designed sequences, human background."""
    sequences = {
        "notebook_idr": NOTEBOOK_IDR,
        "de_tract": DE_TRACT,
        "s_patch": S_PATCH,
        "q_tract": Q_TRACT,
        "k_block": K_BLOCK,
        "p_patch": "PPPPPPPPPPAAPPPPPPPPPPAAPPPPPPPPPPAAPPPP",
        "aromatic": "YYYYYFFFFFWWWWWYYYYYFFFFFGGGGGSSSSSTTTTT",
    }
    return nardini_vectors(
        sequences,
        background=load_reference_background("human"),
        num_scrambles=2000,
        seed=42,
    )


def test_assignment_matches_reference_implementation(gin_result):
    reference = load_gin_reference()
    assignment = assign_gin_clusters(gin_result)

    for index in range(len(gin_result)):
        expected_cluster, expected_sep, expected_distances = _reference_assign(
            gin_result.zscores[index], reference.centroids
        )

        assert int(assignment.clusters[index]) == int(expected_cluster)
        assert np.allclose(assignment.all_distances[index], expected_distances, rtol=1e-12)
        # separations agree unless the reference hit its 100 sentinel, which does
        # not happen for real z-vectors
        if expected_sep < 100:
            assert assignment.separations[index] == pytest.approx(expected_sep)


def test_assignment_places_designed_sequences_in_matching_clusters(gin_result):
    """A D/E tract should land in 'D/E-tracts', a Q tract in 'Q-tracts', etc."""
    frame = assign_gin_clusters(gin_result).to_dataframe()

    assert frame.loc["de_tract", "description"] == "D/E-tracts"
    assert frame.loc["s_patch", "description"] == "S patches"
    assert frame.loc["q_tract", "description"] == "Q-tracts"
    assert frame.loc["k_block", "description"] == "K blocks"


# =====================================================================
# assignment -- behaviour
# =====================================================================


def test_assignment_of_a_centroid_recovers_its_own_cluster():
    """Assigning a centroid must return that cluster at distance zero."""
    reference = load_gin_reference()

    assignment = assign_gin_clusters(reference.centroids)

    assert list(assignment.clusters) == list(reference.clusters)
    assert np.allclose(assignment.distances, 0.0)
    assert np.all(assignment.separations > 0)


def test_assignment_accepts_a_single_vector():
    reference = load_gin_reference()
    assignment = assign_gin_clusters(reference.centroids[7])

    assert len(assignment) == 1
    assert int(assignment.clusters[0]) == 7
    assert assignment.descriptions[0] == "D/E-tracts"


def test_assignment_records(gin_result):
    """The default view is plain dictionaries - no pandas required."""
    records = assign_gin_clusters(gin_result).records()

    assert [r["name"] for r in records] == list(gin_result.names)
    assert set(records[0]) == {
        "name",
        "cluster",
        "description",
        "distance",
        "separation",
        "n_saturated",
        "composition_rms_z",
        "is_typical",
    }
    assert all(r["distance"] >= 0 for r in records)
    assert all(r["separation"] >= 0 for r in records)


def test_assignment_dataframe(gin_result):
    frame = assign_gin_clusters(gin_result).to_dataframe()

    assert list(frame.index) == list(gin_result.names)
    assert "composition_rms_z" in frame.columns
    assert np.all(frame["distance"] >= 0)


def test_assignment_via_result_convenience_method(gin_result):
    direct = assign_gin_clusters(gin_result)
    convenience = gin_result.gin_clusters()

    assert list(direct.clusters) == list(convenience.clusters)
    assert np.allclose(direct.separations, convenience.separations)


def test_ranked_clusters(gin_result):
    assignment = assign_gin_clusters(gin_result)
    ranked = assignment.ranked_clusters("de_tract", n=5)

    assert len(ranked) == 5
    distances = [d for _, _, d in ranked]
    assert distances == sorted(distances)
    # the best entry must agree with the assignment itself
    assert ranked[0][0] == int(assignment.clusters[list(gin_result.names).index("de_tract")])


def test_ranked_clusters_by_index_and_unknown_name(gin_result):
    assignment = assign_gin_clusters(gin_result)
    assert assignment.ranked_clusters(0, n=3) == assignment.ranked_clusters(gin_result.names[0], n=3)

    with pytest.raises(SparrowException):
        assignment.ranked_clusters("not-a-sequence")


def test_assignment_requires_composition_block():
    patterning_only = nardini_vectors({"a": NOTEBOOK_IDR}, num_scrambles=100, seed=1)
    assert not patterning_only.has_composition

    with pytest.raises(SparrowException, match="full 90-feature"):
        assign_gin_clusters(patterning_only)

    with pytest.raises(SparrowException, match="full 90-feature"):
        patterning_only.gin_clusters()


def test_assignment_rejects_wrong_width():
    with pytest.raises(SparrowException, match="90 NARDINI"):
        assign_gin_clusters(np.zeros((2, 36)))


@pytest.mark.parametrize("backend", ["kappa", "iwd"])
def test_assignment_rejects_non_nardini_backend(backend):
    """The published centroids live in NARDINI omega/kappa space and nothing else.

    Assigning a kappa_x or IWD vector against them would silently return
    plausible-looking but meaningless cluster labels, so it must be refused.

    The guard compares the vector's backend against the *reference's* rather than
    hard-coding 'nardini', so that a reference you built yourself in kappa space
    accepts kappa vectors - see test_nardini_clustering.py. The published
    reference is 'nardini', so this case is still refused.
    """
    result = nardini_vectors(
        {"a": NOTEBOOK_IDR},
        background=load_reference_background("human"),
        num_scrambles=100,
        seed=1,
        backend=backend,
    )
    assert result.has_composition  # the failure must be about the backend

    for call in (lambda: assign_gin_clusters(result), result.gin_clusters):
        with pytest.raises(SparrowException, match="different scales") as info:
            call()
        message = str(info.value)
        assert f"'{backend}' vector" in message
        assert "'nardini' space" in message
        # and it says how to proceed, either way
        assert "build_gin_reference()" in message


def test_assignment_accepts_the_nardini_backend_explicitly():
    result = nardini_vectors(
        {"a": NOTEBOOK_IDR},
        background=load_reference_background("human"),
        num_scrambles=100,
        seed=1,
        backend="nardini",
    )
    assert len(result.gin_clusters()) == 1


def test_clipping_is_applied_by_default():
    """An extreme feature must not be allowed to dominate the distance."""
    reference = load_gin_reference()

    # start from a centroid, then blow one feature far out of range
    vector = reference.centroids[7].copy()
    vector[40] = 500.0

    clipped = assign_gin_clusters(vector)
    unclipped = assign_gin_clusters(vector, clip=None)

    # with clipping the assignment survives one crazy feature
    assert int(clipped.clusters[0]) == 7
    # without it, the distance explodes
    assert unclipped.distances[0] > clipped.distances[0] * 100


def test_clip_value_default_matches_published_procedure():
    assert GIN_CLIP == 3.0


def test_assignment_repr(gin_result):
    assert "sequences" in repr(assign_gin_clusters(gin_result))
    assert "clusters" in repr(load_gin_reference())


def test_assignment_is_deterministic_given_a_seed():
    kwargs = dict(background=load_reference_background("human"), num_scrambles=200, seed=5)
    a = nardini_vectors({"x": NOTEBOOK_IDR}, **kwargs).gin_clusters()
    b = nardini_vectors({"x": NOTEBOOK_IDR}, **kwargs).gin_clusters()

    assert list(a.clusters) == list(b.clusters)
    assert np.allclose(a.distances, b.distances)


def test_all_distances_shape(gin_result):
    assignment = assign_gin_clusters(gin_result)
    assert assignment.all_distances.shape == (len(gin_result), 30)


def test_custom_reference_is_used():
    """A caller-supplied reference must override the bundled one."""
    from sparrow.nardini.gin import GINReference

    reference = load_gin_reference()
    # a two-cluster reference built from two of the real centroids
    custom = GINReference(
        feature_names=reference.feature_names,
        clusters=np.array([0, 1]),
        centroids=reference.centroids[[7, 11]],
        descriptions={0: "fake DE", 1: "fake Q"},
    )

    assignment = assign_gin_clusters(reference.centroids[11], reference=custom)
    assert int(assignment.clusters[0]) == 1
    assert assignment.descriptions[0] == "fake Q"


def test_gin_reference_rejects_inconsistent_shape():
    from sparrow.nardini.gin import GINReference

    with pytest.raises(SparrowException):
        GINReference(
            feature_names=("a", "b", "c"),
            clusters=np.array([0, 1]),
            centroids=np.zeros((2, 5)),
        )
