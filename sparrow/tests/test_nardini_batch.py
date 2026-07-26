"""Tests for nardini_proteins - parallel, whole-proteome NARDINI+.

The property that matters most is that parallelism changes nothing: the same
vectors come out regardless of process count or chunking, and they equal what each
Protein would compute for itself. That is only true because per-sequence seeds
derive from the sequence rather than its position in a batch, so these tests are
also the regression guard for that.
"""

import random
import warnings

import numpy as np
import pytest

from sparrow import Protein
from sparrow.nardini import load_reference_background, nardini_proteins, nardini_vectors
from sparrow.sparrow_exceptions import SparrowException, SparrowWarning

# Small synthetic sets, so the background advisory is expected noise here. The
# fork warning is also expected and deliberate: CPython warns when a multi-threaded
# process forks, pytest itself is multi-threaded, and fork is the start method
# sparrow chooses on purpose (see _default_start_method). Filtered here only so our
# own suite stays readable - it is left visible for library users, for whom it is
# genuine information about their process.
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore::sparrow.sparrow_exceptions.BackgroundQualityWarning"
    ),
    pytest.mark.filterwarnings("ignore:This process .* is multi-threaded"),
]

OPTIONS = dict(num_scrambles=200, seed=42)


def _sequences(count, length=110, seed=0):
    rng = random.Random(seed)
    return {
        f"idr{i:03d}": "".join(
            rng.choice("SSTTGGQQNNPPEEDDKKRRAAILVM") for _ in range(length)
        )
        for i in range(count)
    }


@pytest.fixture(scope="module")
def small():
    return _sequences(6)


@pytest.fixture(scope="module")
def enough_to_parallelize():
    """More than the internal threshold, so a real pool gets used."""
    return _sequences(20)


# =====================================================================
# input and output shapes
# =====================================================================


def test_dict_in_dict_out(small):
    out = nardini_proteins(small, **OPTIONS)

    assert isinstance(out, dict)
    assert list(out) == list(small)
    assert all(isinstance(protein, Protein) for protein in out.values())
    assert out["idr000"].sequence == small["idr000"]


def test_list_in_list_out(small):
    out = nardini_proteins(list(small.values()), **OPTIONS)

    assert isinstance(out, list)
    assert [protein.sequence for protein in out] == list(small.values())


def test_protein_objects_are_populated_in_place(small):
    originals = [Protein(sequence) for sequence in small.values()]
    returned = nardini_proteins(originals, **OPTIONS)

    for original, given_back in zip(originals, returned):
        assert original is given_back
    # the vector is available on the object the caller already held
    assert len(originals[0].nardini.vector(**OPTIONS)) == 90


def test_mixed_sequences_and_proteins(small):
    values = list(small.values())
    mixed = {"a": values[0], "b": Protein(values[1]), "c": values[2]}

    out = nardini_proteins(mixed, **OPTIONS)

    assert set(out) == {"a", "b", "c"}
    assert out["b"] is mixed["b"]
    assert all(len(p.nardini.vector(**OPTIONS)) == 90 for p in out.values())


def test_background_none_gives_patterning_only(small):
    out = nardini_proteins(small, background=None, **OPTIONS)
    assert len(out["idr000"].nardini.vector(background=None, **OPTIONS)) == 36


# =====================================================================
# the results must be the real ones
# =====================================================================


def test_attached_result_equals_what_the_protein_would_compute(small):
    """The whole design rests on this: the cached result is not an approximation."""
    out = nardini_proteins(small, **OPTIONS)

    for name, sequence in small.items():
        attached = out[name].nardini.vector(**OPTIONS).to_numpy()
        computed = Protein(sequence).nardini.vector(**OPTIONS).to_numpy()
        assert np.allclose(attached, computed), name


def test_attached_result_equals_nardini_vectors(small):
    out = nardini_proteins(small, **OPTIONS)
    reference = nardini_vectors(
        small, background=load_reference_background("human"), **OPTIONS
    )

    stacked = np.vstack(
        [out[name].nardini.vector(**OPTIONS).to_numpy() for name in small]
    )
    assert np.allclose(stacked, reference.zscores)


def test_result_is_cached_not_recomputed(small, monkeypatch):
    """Reading the vector afterwards must not recompute it."""
    out = nardini_proteins(small, **OPTIONS)

    def explode(*args, **kwargs):
        raise AssertionError("nardini_vectors was called again - cache missed")

    monkeypatch.setattr("sparrow.nardini.vectors.nardini_vectors", explode)

    # same options -> served from the cache the batch run populated
    assert len(out["idr000"].nardini.vector(**OPTIONS)) == 90
    assert len(out["idr000"].nardini.patterning(**OPTIONS)) == 36


def test_gin_assignment_works_off_the_attached_result(small):
    out = nardini_proteins(small, **OPTIONS)
    record = out["idr000"].nardini.gin_cluster(**OPTIONS)

    assert 0 <= record["cluster"] < 30
    assert isinstance(record["description"], str)


def test_different_seeds_give_different_vectors(small):
    a = nardini_proteins(small, num_scrambles=200, seed=1)
    b = nardini_proteins(small, num_scrambles=200, seed=2)

    first = a["idr000"].nardini.vector(num_scrambles=200, seed=1).to_numpy()
    second = b["idr000"].nardini.vector(num_scrambles=200, seed=2).to_numpy()
    assert not np.allclose(first, second)


# =====================================================================
# parallelism must not change anything
# =====================================================================


def test_parallel_matches_serial(enough_to_parallelize):
    """Two processes and one must agree exactly."""
    serial = nardini_proteins(enough_to_parallelize, n_processes=1, **OPTIONS)
    parallel = nardini_proteins(enough_to_parallelize, n_processes=2, **OPTIONS)

    for name in enough_to_parallelize:
        assert np.allclose(
            serial[name].nardini.vector(**OPTIONS).to_numpy(),
            parallel[name].nardini.vector(**OPTIONS).to_numpy(),
        ), name


def test_chunksize_does_not_change_anything(enough_to_parallelize):
    fine = nardini_proteins(
        enough_to_parallelize, n_processes=2, chunksize=1, **OPTIONS
    )
    coarse = nardini_proteins(
        enough_to_parallelize, n_processes=2, chunksize=100, **OPTIONS
    )

    for name in enough_to_parallelize:
        assert np.allclose(
            fine[name].nardini.vector(**OPTIONS).to_numpy(),
            coarse[name].nardini.vector(**OPTIONS).to_numpy(),
        ), name


def test_start_method_does_not_change_anything(enough_to_parallelize):
    """fork and spawn must produce the same numbers."""
    forked = nardini_proteins(
        enough_to_parallelize, n_processes=2, start_method="fork", **OPTIONS
    )
    spawned = nardini_proteins(
        enough_to_parallelize, n_processes=2, start_method="spawn", **OPTIONS
    )

    for name in enough_to_parallelize:
        assert np.allclose(
            forked[name].nardini.vector(**OPTIONS).to_numpy(),
            spawned[name].nardini.vector(**OPTIONS).to_numpy(),
        ), name


# =====================================================================
# error handling
# =====================================================================


def test_bad_sequence_raises_by_default(small):
    broken = dict(small)
    broken["bad"] = "MEEEKKKXXXSSSTTT"  # X has no NARDINI type

    with pytest.raises(SparrowException, match="Could not compute"):
        nardini_proteins(broken, **OPTIONS)


def test_error_message_names_the_sequence_and_the_way_out(small):
    broken = dict(small)
    broken["bad"] = "MEEEKKKXXXSSSTTT"

    with pytest.raises(SparrowException) as info:
        nardini_proteins(broken, **OPTIONS)

    message = str(info.value)
    assert "'bad'" in message
    assert "on_error='skip'" in message


def test_skip_drops_the_bad_one_and_keeps_the_rest(small):
    broken = dict(small)
    broken["bad"] = "MEEEKKKXXXSSSTTT"

    with pytest.warns(SparrowWarning, match="Skipped 1 of"):
        out = nardini_proteins(broken, on_error="skip", **OPTIONS)

    assert "bad" not in out
    assert set(out) == set(small)
    assert len(out["idr000"].nardini.vector(**OPTIONS)) == 90


def test_skip_still_works_in_parallel(enough_to_parallelize):
    broken = dict(enough_to_parallelize)
    broken["bad1"] = "MEEEKKKXXXSSSTTT"
    broken["bad2"] = "QQQ"  # too short for the window

    with pytest.warns(SparrowWarning, match="Skipped 2 of"):
        out = nardini_proteins(broken, on_error="skip", n_processes=2, **OPTIONS)

    assert "bad1" not in out and "bad2" not in out
    assert set(out) == set(enough_to_parallelize)


def test_unknown_on_error_rejected(small):
    with pytest.raises(SparrowException, match="on_error must be"):
        nardini_proteins(small, on_error="ignore", **OPTIONS)


def test_a_bare_string_is_rejected_helpfully():
    with pytest.raises(SparrowException, match="not a single sequence string"):
        nardini_proteins("MEEEKKKSSSTTTQQQNNN", **OPTIONS)


def test_empty_input_rejected():
    with pytest.raises(SparrowException, match="No sequences"):
        nardini_proteins({}, **OPTIONS)


def test_non_sequence_entry_rejected():
    with pytest.raises(SparrowException, match="takes amino acid strings"):
        nardini_proteins({"a": 12345}, **OPTIONS)


# =====================================================================
# the recursion backstop
# =====================================================================


def test_worker_processes_run_serially(monkeypatch, enough_to_parallelize):
    """Inside a worker, no nested pool may be created.

    This is what turns a spawn-platform script with no ``__main__`` guard from a
    recursively-spawning fork bomb into a slow-but-correct serial run.
    """
    import multiprocessing as mp

    monkeypatch.setattr(
        "sparrow.nardini.batch._in_worker_process", lambda: True, raising=True
    )

    def no_pools(*args, **kwargs):
        raise AssertionError("a nested pool was created inside a worker")

    monkeypatch.setattr(mp.context.BaseContext, "Pool", no_pools, raising=True)

    out = nardini_proteins(enough_to_parallelize, n_processes=4, **OPTIONS)
    assert len(out) == len(enough_to_parallelize)


def test_default_start_method_prefers_fork_where_available():
    import multiprocessing as mp

    from sparrow.nardini.batch import _default_start_method

    expected = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    assert _default_start_method() == expected


def test_progress_bar_runs(small):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = nardini_proteins(small, show_progress=True, **OPTIONS)
    assert len(out) == len(small)
