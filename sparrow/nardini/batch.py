"""Compute NARDINI+ vectors for a whole proteome in parallel.

:func:`nardini_proteins` takes sequences or :class:`~sparrow.protein.Protein`
objects, computes every NARDINI+ vector across multiple processes, and hands back
Protein objects with the results already attached - so the expensive work happens
once, up front, and afterwards ``p.nardini.vector(...)`` is free::

    from sparrow.nardini import nardini_proteins

    proteins = nardini_proteins(idr_sequences, num_scrambles=10000, seed=42,
                                show_progress=True)

    proteins["IDR1"].nardini.vector(seed=42, num_scrambles=10000)   # cached, instant
    proteins["IDR1"].nardini.gin_cluster(seed=42, num_scrambles=10000)

Input and output shapes match: pass a dict and you get a dict back with the same
keys, pass a list and you get a list in the same order. Sequences become new
Protein objects; Protein objects you pass in are populated in place and returned.

Why this is safe to parallelize
-------------------------------
The work is split by chunking the input and running the ordinary
:func:`~sparrow.nardini.vectors.nardini_vectors` on each chunk in a worker
process. Nothing about the calculation is reimplemented here, which matters
because the per-sequence seeds derive from the sequence itself rather than its
position in a batch. A vector therefore does not depend on how the input was
chunked, how many processes ran, or what order they finished in - so the
parallel result is *identical* to the serial one, and identical to what each
Protein would compute for itself. That is asserted directly in
``test_nardini_batch.py``.

Only the patterning half is farmed out, because that is where the cost is: at
1,000 scrambles it runs about 150x the cost of the compositional half per
sequence, and the compositional half needs no scrambles at all.

Cost
----
Roughly 110 ms per IDR at 1,000 scrambles and 560 ms at 10,000 (200-residue IDR)
on one core, so a 25,000-IDR proteome is about 45 minutes or 4 hours serially -
divided by however many cores you give it. Worker start-up costs about a second
each because sparrow has to be imported in every process, which is why this is
worth using for hundreds of sequences and not for five.
"""

import os
import warnings

from sparrow.nardini.patterning import (
    DEFAULT_BACKEND,
    DEFAULT_BLOB_SIZE,
    DEFAULT_MIN_FRACTION,
    DEFAULT_NUM_SCRAMBLES,
)
from sparrow.nardini.protein_interface import DEFAULT_REFERENCE
from sparrow.sparrow_exceptions import SparrowException, SparrowWarning

__all__ = ["nardini_proteins", "score_in_parallel"]

#: Target number of chunks per worker. More than one keeps workers busy when
#: sequences differ in length (cost scales with length), at the price of a little
#: more per-chunk overhead.
_CHUNKS_PER_WORKER = 4

#: Below this many sequences, parallelism costs more than it saves - each worker
#: spends about a second importing sparrow before it does any work.
_MIN_FOR_PARALLEL = 16


def _default_start_method():
    """Prefer ``fork`` where it exists, because ``spawn`` re-imports ``__main__``.

    This is not a micro-optimization, it is a correctness issue for the way people
    actually run this. Under ``spawn`` (the default on macOS and Windows) each
    worker re-imports the parent's ``__main__`` module. A script that calls
    ``nardini_proteins`` at module level therefore calls it again in every worker,
    which spawns more workers - a fork bomb, not an error message. The standard
    remedy is an ``if __name__ == "__main__":`` guard, but that cannot be enforced,
    and it does not exist at all in a Jupyter notebook or an interactive session,
    which is where a lot of this work happens.

    ``fork`` does not re-import anything, so none of that applies. It is available
    on Linux and macOS. Where it is not (Windows), ``spawn`` is used and the
    ``__main__`` guard genuinely is required - see :func:`nardini_proteins`.

    ``forkserver`` looks like it should be the best of both and is not: it was
    measured re-importing ``__main__`` exactly as ``spawn`` does, so it recurses in
    the same way. Do not switch to it expecting otherwise.

    The tradeoff is real: forking a process that holds threaded state can deadlock
    in the child, and CPython emits a DeprecationWarning about it when the parent is
    multi-threaded. The NARDINI path is NumPy and SciPy only and never touches the
    predictor networks, and each child does its numpy work and exits, so this is
    safe in practice. If your parent process has live threads you care about - or
    you would simply rather not rely on that argument - pass
    ``start_method='spawn'`` and use a ``__main__`` guard.
    """
    import multiprocessing as mp

    available = mp.get_all_start_methods()
    return "fork" if "fork" in available else "spawn"


def _in_worker_process():
    """Whether this process is a multiprocessing worker rather than the parent.

    The backstop for the recursion described in :func:`_default_start_method`: a
    spawned worker re-importing ``__main__`` reaches ``nardini_proteins`` again,
    and must run serially rather than build another pool. ``spawn`` sets the
    process name before it re-imports ``__main__``, so this is already true by the
    time the recursive call happens.
    """
    import multiprocessing as mp

    return mp.current_process().name != "MainProcess" or mp.parent_process() is not None


# .................................................................
# worker side
#
_WORKER = {}


def _init_worker(background, options):
    """Store the per-run configuration once per worker, not once per chunk."""
    _WORKER["background"] = background
    _WORKER["options"] = options


def _run_chunk(chunk):
    """Score one chunk of ``(name, sequence)`` pairs in a worker process.

    Returns
    -------
    tuple
        ``(results, failures)`` where ``results`` is a list of
        ``(name, NardiniResult)`` for a single sequence each, and ``failures`` is
        a list of ``(name, message)``.
    """
    from sparrow.nardini.vectors import nardini_vectors

    background = _WORKER["background"]
    options = _WORKER["options"]

    try:
        result = nardini_vectors(dict(chunk), background=background, **options)
    except Exception:
        # Isolate the offender: one bad sequence must not discard a whole chunk's
        # worth of work. Re-running the chunk one sequence at a time costs the
        # chunk again, but only for chunks that actually contain a failure.
        results, failures = [], []
        for name, sequence in chunk:
            try:
                single = nardini_vectors(
                    {name: sequence}, background=background, **options
                )
            except Exception as exc:
                failures.append((name, f"{type(exc).__name__}: {exc}"))
            else:
                results.append((name, single))
        return results, failures

    return [
        (name, _slice(result, index)) for index, name in enumerate(result.names)
    ], []


def _slice(result, index):
    """Extract one sequence from a multi-sequence result as its own result.

    The extracted result is named ``'sequence'`` so it is indistinguishable from
    what ``ProteinNardini`` builds for itself, which is what lets it be stored in
    that object's cache.
    """
    from sparrow.nardini.vectors import NardiniResult

    return NardiniResult(
        names=("sequence",),
        sequences=(result.sequences[index],),
        patterning_names=result.patterning_feature_names,
        composition_names=result.composition_feature_names,
        patterning_zscores=result.patterning_zscores[index : index + 1],
        composition_zscores=result.composition_zscores[index : index + 1],
        raw_patterning=result.raw_patterning[index : index + 1],
        raw_composition=result.raw_composition[index : index + 1],
        backend=result.backend,
    )


# .................................................................
# parent side
#
def _normalize_input(sequences):
    """Return ``(names, proteins, was_mapping)`` from any supported input.

    Accepts a mapping or an iterable, of sequences or Protein objects (or a mix).
    Protein objects are kept as-is so they can be populated in place.
    """
    from collections.abc import Mapping

    from sparrow.protein import Protein

    if isinstance(sequences, Mapping):
        items = list(sequences.items())
        was_mapping = True
    elif isinstance(sequences, (str, bytes)):
        raise SparrowException(
            "nardini_proteins() takes a collection of sequences or Protein "
            "objects, not a single sequence string. Wrap it: "
            "nardini_proteins({'IDR1': 'MEEK...'})."
        )
    else:
        items = list(enumerate(sequences))
        was_mapping = False

    if not items:
        raise SparrowException("No sequences were supplied to nardini_proteins().")

    names, proteins = [], []
    for key, value in items:
        if isinstance(value, Protein):
            protein = value
        elif isinstance(value, str):
            protein = Protein(value)
        else:
            raise SparrowException(
                f"Entry {key!r} is a {type(value).__name__}; nardini_proteins() "
                "takes amino acid strings or sparrow.Protein objects."
            )
        names.append(str(key))
        proteins.append(protein)

    return names, proteins, was_mapping


def _chunk(pairs, chunksize):
    for start in range(0, len(pairs), chunksize):
        yield pairs[start : start + chunksize]


def score_in_parallel(
    pairs,
    background,
    options,
    n_processes=None,
    chunksize=None,
    start_method=None,
    show_progress=False,
):
    """Score ``(name, sequence)`` pairs across processes, one result per sequence.

    The shared engine behind both :func:`nardini_proteins` and the
    ``n_processes`` argument of :func:`sparrow.nardini.vectors.nardini_vectors`,
    so there is one implementation of the chunking, the pool handling and the
    per-sequence error isolation.

    Parameters
    ----------
    pairs : list of tuple
        ``(name, sequence)`` pairs. Names must be unique - results are collected
        into a dict keyed by them.
    background : NardiniBackground or None
        Already resolved, so workers receive a small pickled object.
    options : dict
        Keyword arguments forwarded to
        :func:`~sparrow.nardini.vectors.nardini_vectors` in each worker. Must not
        contain ``n_processes``, or the workers would try to nest pools.
    n_processes : int or None, optional
        Worker processes. Default ``os.cpu_count()``. ``1`` runs in this process.
    chunksize : int or None, optional
        Sequences per task. Default aims for a few chunks per worker.
    start_method : str or None, optional
        Multiprocessing start method; see :func:`_default_start_method`.
    show_progress : bool, optional
        Progress bar over chunks. Default False.

    Returns
    -------
    tuple
        ``(computed, failures)`` where ``computed`` maps name to a
        single-sequence :class:`~sparrow.nardini.vectors.NardiniResult`, and
        ``failures`` is a list of ``(name, message)``.
    """
    if n_processes is None:
        n_processes = os.cpu_count() or 1
    n_processes = max(1, int(n_processes))

    if chunksize is None:
        chunksize = max(1, len(pairs) // (n_processes * _CHUNKS_PER_WORKER))
    chunksize = max(1, int(chunksize))

    chunks = list(_chunk(pairs, chunksize))

    # Run in-process when a pool would not pay for itself, and - importantly - when
    # we are already inside a worker. The latter is what stops a spawn-platform
    # script with no __main__ guard from recursively spawning: the re-imported
    # module still computes the right answer, just serially.
    serial = (
        n_processes == 1 or len(pairs) < _MIN_FOR_PARALLEL or _in_worker_process()
    )

    if serial:
        _init_worker(background, options)
        outcomes = map(_run_chunk, chunks)
        pool = None
    else:
        import multiprocessing as mp

        method = start_method or _default_start_method()
        pool = mp.get_context(method).Pool(
            processes=n_processes,
            initializer=_init_worker,
            initargs=(background, options),
        )
        outcomes = pool.imap_unordered(_run_chunk, chunks)

    if show_progress:
        from tqdm import tqdm

        outcomes = tqdm(outcomes, total=len(chunks), desc="NARDINI")

    computed, failures = {}, []
    try:
        for chunk_results, chunk_failures in outcomes:
            for name, result in chunk_results:
                computed[name] = result
            failures.extend(chunk_failures)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()

    return computed, failures


def nardini_proteins(
    sequences,
    background=DEFAULT_REFERENCE,
    num_scrambles=DEFAULT_NUM_SCRAMBLES,
    seed=None,
    typeset="8",
    blob_size=DEFAULT_BLOB_SIZE,
    min_fraction=DEFAULT_MIN_FRACTION,
    backend=DEFAULT_BACKEND,
    n_processes=None,
    chunksize=None,
    start_method=None,
    show_progress=False,
    on_error="raise",
):
    """Compute NARDINI+ vectors for many sequences in parallel.

    Returns Protein objects with the results already attached, so subsequent
    ``protein.nardini.vector(...)`` calls with the same options are free rather
    than recomputed. This is the way to build a proteome's worth of vectors.

    Parameters
    ----------
    sequences : dict, list, or iterable
        Amino acid strings, :class:`~sparrow.protein.Protein` objects, or a mix.
        A mapping returns a dict keyed the same way; anything else returns a list
        in input order. Protein objects are populated in place and returned;
        strings become new Protein objects.
    background : str, NardiniBackground, os.PathLike, or None, optional
        Reference set for the compositional half. Defaults to ``'human'``, the
        same default as :attr:`~sparrow.protein.Protein.nardini`, so that a later
        ``p.nardini.vector()`` with no arguments hits the cached result. Pass your
        own for a non-human proteome, or None for patterning features only.
    num_scrambles : int, optional
        Scrambles per sequence for the patterning null. Default 10,000. This is
        the main cost lever and the cost is linear in it.
    seed : int or None, optional
        Seed for reproducible patterning z-scores. Strongly recommended: without
        it a rerun gives different numbers, and you cannot cheaply recompute one
        sequence to match.
    typeset : {'8', '9'}, optional
        Residue grouping. Default ``'8'``.
    blob_size : int, optional
        Patterning window size. Default 5.
    min_fraction : float, optional
        Abundance threshold for scoring a patterning feature. Default 0.10.
    backend : str, optional
        Patterning statistic. Default ``'nardini'``.
    n_processes : int or None, optional
        Worker processes. Default is ``os.cpu_count()``. ``1`` runs in this
        process with no pool, which is also what happens for small inputs.
    chunksize : int or None, optional
        Sequences per task. Default aims for a few chunks per worker so that
        uneven sequence lengths do not leave workers idle.
    start_method : str or None, optional
        Multiprocessing start method. Default is ``'fork'`` where available
        (Linux, macOS) and ``'spawn'`` otherwise, because ``spawn`` re-imports the
        calling ``__main__`` module in every worker. On a ``spawn`` platform you
        must therefore call this from inside an ``if __name__ == "__main__":``
        block; without one, sparrow detects that it is running inside a worker and
        falls back to computing serially rather than spawning recursively.
    show_progress : bool, optional
        Progress bar over chunks. Default False.
    on_error : {'raise', 'skip'}, optional
        What to do about a sequence NARDINI cannot score - a non-standard residue,
        or one too short for the window. ``'raise'`` (default) propagates.
        ``'skip'`` omits it from the returned collection and warns with a summary,
        which is usually what you want for a real proteome where a handful of
        entries are malformed and the run takes hours.

    Returns
    -------
    dict or list
        Protein objects with NARDINI+ results attached, matching the input shape.
        With ``on_error='skip'`` the failures are absent, so the result may be
        smaller than the input.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the input is empty or holds something that is neither a sequence nor a
        Protein, if ``on_error`` is not recognized, or - with ``on_error='raise'``
        - if any sequence cannot be scored.

    Warns
    -----
    sparrow.sparrow_exceptions.SparrowWarning
        With ``on_error='skip'``, once, summarizing what was dropped.

    Notes
    -----
    The parallel result is identical to the serial one. Per-sequence seeds derive
    from the sequence rather than its position, so chunking and process count
    cannot change any vector.

    Examples
    --------
    >>> proteins = nardini_proteins(                        # doctest: +SKIP
    ...     {"IDR1": "MEEEKKK...", "IDR2": "QQQQNNN..."},
    ...     num_scrambles=10000, seed=42, show_progress=True,
    ... )
    >>> proteins["IDR1"].nardini.vector(seed=42, num_scrambles=10000)["Frac K+R"]
    ...                                                     # doctest: +SKIP

    See Also
    --------
    sparrow.nardini.nardini_vectors : The single-process equivalent, which returns
        a matrix rather than Protein objects.
    """
    if on_error not in ("raise", "skip"):
        raise SparrowException(f"on_error must be 'raise' or 'skip', got {on_error!r}")

    names, proteins, was_mapping = _normalize_input(sequences)

    options = {
        "num_scrambles": num_scrambles,
        "seed": seed,
        "typeset": typeset,
        "blob_size": blob_size,
        "min_fraction": min_fraction,
        "backend": backend,
    }

    # resolve the background once, here: workers get a small pickled object
    # rather than each re-reading a file or re-deriving a species key
    from sparrow.nardini.protein_interface import resolve_background

    resolved_background = resolve_background(background)

    pairs = [(name, protein.sequence) for name, protein in zip(names, proteins)]

    computed, failures = score_in_parallel(
        pairs,
        resolved_background,
        options,
        n_processes=n_processes,
        chunksize=chunksize,
        start_method=start_method,
        show_progress=show_progress,
    )

    if failures and on_error == "raise":
        name, message = failures[0]
        raise SparrowException(
            f"Could not compute a NARDINI+ vector for {name!r}: {message}. "
            f"{len(failures)} of {len(pairs)} sequences failed. Pass "
            "on_error='skip' to drop them and keep the rest."
        )

    if failures:
        listed = ", ".join(repr(name) for name, _ in failures[:5])
        if len(failures) > 5:
            listed += ", ..."
        warnings.warn(
            f"Skipped {len(failures)} of {len(pairs)} sequences that could not be "
            f"scored ({listed}). The first failure was: {failures[0][1]}",
            SparrowWarning,
            stacklevel=2,
        )

    # attach the results, then hand back the same shape we were given
    kept_names, kept_proteins = [], []
    for name, protein in zip(names, proteins):
        result = computed.get(name)
        if result is None:
            continue
        protein.nardini._store(result, background=background, **options)
        kept_names.append(name)
        kept_proteins.append(protein)

    if was_mapping:
        return dict(zip(kept_names, kept_proteins))
    return kept_proteins
