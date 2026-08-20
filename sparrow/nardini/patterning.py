"""NARDINI patterning asymmetries and scramble-based z-scores.

This module implements the patterning half of NARDINI: for each residue type and
each pair of types, quantify how unevenly that type (or pair) is distributed
along the sequence, then convert the value into a z-score against the
distribution obtained from randomly scrambled versions of the *same* sequence.

Because scrambling preserves composition exactly, the z-score isolates
*patterning* from composition: a large positive z-score means the residues of
that type are more segregated (blockier) than composition alone would predict,
and a large negative z-score means they are more evenly interspersed.

What is z-scored is **delta**: the mean squared deviation of the per-window
asymmetry from the whole-sequence asymmetry, unnormalized. The ``'kappa'`` backend
z-scores delta divided by ``delta_max`` instead, which is the same quantity once
z-scored - the divisor depends only on composition, and every scramble shares the
sequence's composition, so a constant cancels. See :ref:`nardini-delta-vs-kappa`.

The two underlying statistics, both evaluated over sliding windows ("blobs") of
``blob_size`` residues:

* **omega** (diagonal features) - asymmetry of one type against all other
  residues::

      sigma_all    = (f - (1 - f))^2
      sigma_window = (c/b - (1 - c/b))^2
      omega        = mean over windows of (sigma_window - sigma_all)^2

  where ``f`` is the type's whole-sequence fraction, ``c`` its count in a window
  and ``b`` the window size.

* **kappa** (off-diagonal features) - asymmetry between two types::

      sigma_all    = (f1 - f2)^2 / (f1 + f2)
      sigma_window = (c1/b - c2/b)^2 / (c1/b + c2/b)      [0 if c1 + c2 == 0]
      kappa        = mean over windows of (sigma_window - sigma_all)^2

A feature is only evaluated when the type is abundant enough to be meaningful:
the whole-sequence fraction must *exceed* ``min_fraction`` (both fractions must,
for a pair). Features that fail this test are reported as a raw value of 0.0 and
a z-score of 0.0.

The scramble distribution for each evaluated feature is summarized by fitting a
three-parameter gamma distribution (shape, location, scale) by maximum
likelihood, exactly as the reference implementation does, and the z-score uses
the *fitted* mean and variance rather than the sample moments.

Implementation note
-------------------
The residue groupings are exact partitions of the 20 amino acids, so a sequence
reduces to an array of integer type labels. Scrambling is then a permutation of
that label array, and all window counts for all types follow from one cumulative
sum. This makes the calculation a few hundred times faster than the loop-based
reference implementation while producing identical numbers, which matters because
the published protocol uses 100,000 scrambles per sequence.

The ``'kappa'`` and ``'iwd'`` backends get the same treatment through
:mod:`sparrow.patterning.vectorized`, which scores a whole block of scrambles per
call, so all three backends cost about the same.
"""

import numpy as np
from scipy import stats

from sparrow.nardini.groups import (
    DEFAULT_TYPESET,
    get_typeset,
    patterning_feature_names,
)
from sparrow.sparrow_exceptions import SparrowException

__all__ = [
    "DEFAULT_BLOB_SIZE",
    "DEFAULT_MIN_FRACTION",
    "DEFAULT_NUM_SCRAMBLES",
    "DEFAULT_BACKEND",
    "PATTERNING_BACKENDS",
    "resolve_backend",
    "encode_types",
    "type_fractions",
    "patterning_values",
    "patterning_zscores",
]

#: The patterning statistic used unless another is requested.
DEFAULT_BACKEND = "nardini"

#: Available patterning backends, mapped to a one-line summary.
#:
#: * ``'nardini'`` - the published NARDINI omega/kappa window asymmetry. This is
#:   the only backend whose z-scores live in the space the GIN centroids were
#:   defined in, so it is required for GIN cluster assignment.
#: * ``'kappa'`` - sparrow's own generalized kappa (``kappa_x``): the same
#:   asymmetry as above, divided by the delta of the most segregated arrangement
#:   of the same composition, and bounded 0-1. Because that divisor depends only
#:   on composition - which scrambling preserves - it cancels out of the z-score,
#:   so this measures the same thing as ``'nardini'`` and is offered for
#:   completeness and for continuity with the deprecated
#:   :mod:`sparrow.sequence_analysis.grammar`. See the NARDINI guide.
#: * ``'iwd'`` - sparrow's inverse-weighted distance, a clustering metric that
#:   ignores ``blob_size`` entirely.
PATTERNING_BACKENDS = {
    "nardini": "published NARDINI omega/kappa window asymmetry (required for GIN)",
    "kappa": "sparrow generalized kappa (kappa_x), normalized and bounded 0-1",
    "iwd": "sparrow inverse-weighted distance (window-size independent)",
}

# names used by the older sparrow.sequence_analysis.grammar module
_BACKEND_ALIASES = {
    "kappa_cython": "kappa",
    "iwd_combined": "iwd",
}


def resolve_backend(backend):
    """Normalize a patterning backend name.

    Accepts the canonical names in :data:`PATTERNING_BACKENDS` as well as the
    legacy names used by :mod:`sparrow.sequence_analysis.grammar`
    (``'kappa_cython'`` -> ``'kappa'``, ``'iwd_combined'`` -> ``'iwd'``).

    Parameters
    ----------
    backend : str
        Backend name.

    Returns
    -------
    str
        The canonical backend name.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the name is not recognized.
    """
    key = str(backend).strip().lower()
    key = _BACKEND_ALIASES.get(key, key)

    if key not in PATTERNING_BACKENDS:
        raise SparrowException(
            f"Unknown patterning backend '{backend}'. Must be one of {sorted(PATTERNING_BACKENDS)}."
        )
    return key


#: Sliding-window ("blob") size used for the asymmetry calculations.
DEFAULT_BLOB_SIZE = 5

#: A type must exceed this whole-sequence fraction for its features to be scored.
DEFAULT_MIN_FRACTION = 0.10

#: Number of scrambles used to build the null distribution. The NARDINI paper
#: uses 100,000; the NARDINI+ notebook offers 1,000 / 10,000 / 100,000 and
#: defaults to 10,000, which is a reasonable speed/precision compromise.
DEFAULT_NUM_SCRAMBLES = 10000

# scrambles are processed in blocks so that memory stays bounded regardless of
# num_scrambles while still getting the benefit of vectorization
_SCRAMBLE_BLOCK = 512


def encode_types(sequence, typeset=DEFAULT_TYPESET):
    """Encode a sequence as an array of integer residue-type labels.

    Parameters
    ----------
    sequence : str
        Amino acid sequence (upper case, standard residues only).
    typeset : {'8', '9'} or dict, optional
        Residue grouping, see :func:`sparrow.nardini.groups.get_typeset`.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(sequence),)`` and dtype ``int8`` where each element
        is the index of that residue's type within the grouping.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the sequence contains a residue not present in the grouping.
    """
    groups = get_typeset(typeset)

    lookup = np.full(128, -1, dtype=np.int8)
    for index, residues in enumerate(groups.values()):
        for residue in residues:
            lookup[ord(residue)] = index

    codes = np.frombuffer(sequence.encode("ascii", errors="replace"), dtype=np.uint8)
    labels = lookup[codes]

    if np.any(labels < 0):
        bad = sorted({sequence[i] for i in np.where(labels < 0)[0]})
        raise SparrowException(
            f"Sequence contains residue(s) with no NARDINI type: {bad}. "
            "NARDINI requires the 20 standard amino acids; strip or convert "
            "non-standard residues first (e.g. Protein(seq, validate=True))."
        )

    return labels


def type_fractions(labels, num_types):
    """Return the whole-sequence fraction of each residue type.

    Parameters
    ----------
    labels : numpy.ndarray
        Integer type labels, as returned by :func:`encode_types`.
    num_types : int
        Number of residue types in the grouping.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(num_types,)`` of fractions summing to 1.
    """
    counts = np.bincount(labels, minlength=num_types).astype(np.float64)
    return counts / labels.size


def _window_counts(labels_2d, num_types, blob_size, types=None):
    """Sliding-window counts of every type, for a batch of label arrays.

    Parameters
    ----------
    labels_2d : numpy.ndarray
        Integer array of shape ``(n_sequences, length)``.
    num_types : int
        Number of residue types.
    blob_size : int
        Window size.
    types : sequence of int, optional
        Only count these residue types, in this order. Default None (all of
        them). Restricting to the types some evaluated feature actually refers
        to is worth real time on the scramble path, where this runs thousands of
        times.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_sequences, n_types, n_windows)`` of counts, where
        ``n_types`` is ``num_types`` (or ``len(types)``) and ``n_windows`` is
        ``length - blob_size + 2``. The final window contains the trailing
        ``blob_size - 1`` residues, matching the published NARDINI protocol.
    """
    n_seqs, length = labels_2d.shape
    wanted = range(num_types) if types is None else list(types)

    # A window count is the difference of two prefix sums over an indicator of
    # "is this position type t". Done one type at a time rather than one-hotting
    # every type at once: it reuses a single (n_seqs, length + 1) buffer instead
    # of allocating an (n_seqs, num_types, length) cube, and it lets the caller
    # skip types no evaluated feature refers to. int16 is ample - a count cannot
    # exceed the sequence length - and halves the memory traffic of int32.
    out = np.empty((n_seqs, len(wanted), length - blob_size + 2), dtype=np.int16)
    prefix = np.zeros((n_seqs, length + 1), dtype=np.int16)

    for slot, residue_type in enumerate(wanted):
        np.cumsum(labels_2d == residue_type, axis=1, out=prefix[:, 1:])
        np.subtract(prefix[:, blob_size:], prefix[:, :-blob_size], out=out[:, slot, :-1])
        # NARDINI also scores the final partial window, but still normalizes
        # its counts by blob_size in _asymmetries_from_counts.
        np.subtract(prefix[:, -1], prefix[:, -blob_size], out=out[:, slot, -1])

    return out


def _asymmetries_from_counts(counts, fractions, pair_index, blob_size, mask=None, slot_of=None):
    """Compute all patterning features from precomputed window counts.

    Parameters
    ----------
    counts : numpy.ndarray
        Window counts of shape ``(n_sequences, num_types, n_windows)``.
    fractions : numpy.ndarray
        Whole-sequence type fractions, shape ``(num_types,)``. Scrambling
        preserves composition, so one set of fractions applies to every row.
    pair_index : list of tuple
        ``(i, j)`` type-index pairs in canonical feature order.
    blob_size : int
        Window size.
    mask : numpy.ndarray, optional
        Boolean mask over ``pair_index``. Features that are False are left at 0.0
        rather than computed, which is what the caller would do with them anyway.
    slot_of : numpy.ndarray, optional
        Maps a residue-type index onto its row in ``counts``, for when ``counts``
        covers only a subset of types. Default None (row == type index).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_sequences, len(pair_index))`` of raw asymmetries;
        features excluded by ``mask`` are 0.0.
    """
    n_seqs = counts.shape[0]
    out = np.zeros((n_seqs, len(pair_index)), dtype=np.float64)

    # window fractions
    frac_windows = counts / float(blob_size)

    # `counts` may cover only some types, in which case slot_of maps a type index
    # onto its row. Without it, the type index is the row.
    def row(type_index):
        return type_index if slot_of is None else slot_of[type_index]

    for feature, (i, j) in enumerate(pair_index):
        # Skipping unevaluated features is the single biggest saving on the
        # scramble path: their values are discarded by the caller anyway, and a
        # typical IDR evaluates only a third of the 36.
        if mask is not None and not mask[feature]:
            continue

        if i == j:
            # omega: one type against everything else
            f = fractions[i]
            sigma_all = (f - (1.0 - f)) ** 2
            window = frac_windows[:, row(i), :]
            sigma_window = (window - (1.0 - window)) ** 2
        else:
            # kappa: type i against type j
            f1, f2 = fractions[i], fractions[j]
            denominator = f1 + f2
            if denominator == 0:
                # cannot happen for gated features, but keep it defined
                continue
            sigma_all = (f1 - f2) ** 2 / denominator

            c1 = frac_windows[:, row(i), :]
            c2 = frac_windows[:, row(j), :]
            total = c1 + c2
            sigma_window = np.zeros_like(total)
            nonzero = total > 0
            np.divide(
                (c1 - c2) ** 2,
                total,
                out=sigma_window,
                where=nonzero,
            )

        out[:, feature] = np.mean((sigma_window - sigma_all) ** 2, axis=1)

    return out


def _pair_index(num_types):
    """Return canonical ``(i, j)`` type pairs (upper triangle, row-major)."""
    return [(i, j) for i in range(num_types) for j in range(i, num_types)]


def _alt_backend_argsets(groups, pairs, backend, mask):
    """Precompute the per-feature residue lists for a non-NARDINI backend.

    The underlying routines take Python lists of residues, and those lists are
    the same for every scramble. Building them once here rather than inside the
    scramble loop removes ~36 list/set constructions per scramble.

    Parameters
    ----------
    groups : collections.OrderedDict
        The residue grouping.
    pairs : list of tuple
        ``(i, j)`` type-index pairs in canonical order.
    backend : {'kappa', 'iwd'}
        Which statistic the arguments are for.
    mask : numpy.ndarray
        Boolean mask of features that clear the abundance threshold.

    Returns
    -------
    list of tuple
        ``(feature_index, args)`` for each evaluated feature, where ``args`` is
        the residue-list argument tuple that routine takes.
    """
    group_lists = [sorted(residues) for residues in groups.values()]

    argsets = []
    for feature, (i, j) in enumerate(pairs):
        if not mask[feature]:
            continue

        group1, group2 = group_lists[i], group_lists[j]

        if backend == "kappa":
            # an empty group2 makes kappa_x a binary (group1 vs rest) statistic
            argsets.append((feature, (group1, [] if i == j else group2)))
        else:  # iwd
            residues = group1 if i == j else sorted(set(group1) | set(group2))
            argsets.append((feature, (residues,)))

    return argsets


def _alt_backend_values(sequence, argsets, n_features, backend, blob_size):
    """Patterning values for the non-NARDINI backends.

    Uses the vectorized NumPy reimplementations in
    :mod:`sparrow.patterning.vectorized` rather than the Cython extensions, so
    that a sequence's observed value and its scramble null are produced by the
    same code path and are directly comparable. The NumPy versions carry these
    quantities in double precision where the Cython ones use single, so they
    agree with the compiled routines to ~1e-6 and are the more accurate of the
    two.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    argsets : list of tuple
        Precomputed ``(feature_index, args)`` pairs from
        :func:`_alt_backend_argsets`.
    n_features : int
        Total number of features (the returned array's length).
    backend : {'kappa', 'iwd'}
        Which statistic to compute.
    blob_size : int
        Window size (ignored by the ``'iwd'`` backend).

    Returns
    -------
    numpy.ndarray
        Raw values in canonical feature order; unevaluated features are 0.0.
    """
    from sparrow.patterning import vectorized as vec

    values = np.zeros(n_features, dtype=np.float64)

    if backend == "kappa":
        for feature, (group1, group2) in argsets:
            value = vec.kappa_x(sequence, group1, group2, blob_size, 1)
            # kappa_x reports -1 when undefined; that means "not scored"
            values[feature] = 0.0 if value < 0 else value
    else:
        for feature, (residues,) in argsets:
            values[feature] = vec.calculate_average_inverse_distance_from_sequence(sequence, residues)

    return values


def _evaluated_mask(fractions, pair_index, min_fraction):
    """Boolean mask of which features clear the abundance threshold."""
    mask = np.zeros(len(pair_index), dtype=bool)
    for feature, (i, j) in enumerate(pair_index):
        if i == j:
            mask[feature] = fractions[i] > min_fraction
        else:
            mask[feature] = fractions[i] > min_fraction and fractions[j] > min_fraction
    return mask


def patterning_values(
    sequence,
    typeset=DEFAULT_TYPESET,
    blob_size=DEFAULT_BLOB_SIZE,
    min_fraction=DEFAULT_MIN_FRACTION,
    backend=DEFAULT_BACKEND,
):
    """Compute the raw (un-z-scored) patterning values for a sequence.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    typeset : {'8', '9'} or dict, optional
        Residue grouping, see :func:`sparrow.nardini.groups.get_typeset`.
    blob_size : int, optional
        Sliding-window size. Default 5. Ignored by the ``'iwd'`` backend.
    min_fraction : float, optional
        A type's whole-sequence fraction must exceed this for its features to be
        evaluated; others are returned as 0.0. Default 0.10.
    backend : str, optional
        Which patterning statistic to use - see :data:`PATTERNING_BACKENDS`.
        Default ``'nardini'``.

    Returns
    -------
    numpy.ndarray
        Raw value per feature, in the order given by
        :func:`sparrow.nardini.groups.patterning_feature_names`. Length 36 for
        the default 8-type grouping.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the sequence is shorter than ``blob_size``, contains a non-standard
        residue, or ``backend`` is unknown.

    Examples
    --------
    >>> from sparrow.nardini.patterning import patterning_values
    >>> values = patterning_values("MEEEKKKKSSSTTTDDDQQQQNNNN")
    >>> len(values)
    36
    """
    backend = resolve_backend(backend)
    groups = get_typeset(typeset)
    num_types = len(groups)

    if blob_size < 2:
        raise SparrowException(f"blob_size must be >= 2 (got {blob_size})")
    if len(sequence) < blob_size:
        raise SparrowException(
            f"Sequence of length {len(sequence)} is shorter than blob_size {blob_size}; patterning cannot be computed."
        )

    labels = encode_types(sequence, typeset)
    fractions = type_fractions(labels, num_types)
    pairs = _pair_index(num_types)
    mask = _evaluated_mask(fractions, pairs, min_fraction)

    if backend != "nardini":
        argsets = _alt_backend_argsets(groups, pairs, backend, mask)
        return _alt_backend_values(sequence, argsets, len(pairs), backend, blob_size)

    counts = _window_counts(labels[None, :], num_types, blob_size)
    values = _asymmetries_from_counts(counts, fractions, pairs, blob_size)[0]

    # zero out features whose types are too rare to be meaningful
    values[~mask] = 0.0

    return values


#: A null distribution whose standard deviation is below this fraction of its own
#: mean is treated as a point mass. Some statistics genuinely cannot vary under
#: scrambling for some sequences - IWD on a sequence whose target group covers
#: every position, for instance - and the resulting spread is pure floating-point
#: noise, not signal.
_DEGENERATE_SCALE = 1e-12

#: A gamma fit is rejected if its mean differs from the sample mean by more than
#: this factor in either direction. Good fits agree with the sample mean to
#: several decimal places, so this only ever catches a failed optimization.
_GAMMA_MEAN_TOLERANCE = 2.0

#: As above, for the variance. Looser, because a legitimately skewed gamma fit
#: can differ from the sample variance by a fair margin.
_GAMMA_VAR_TOLERANCE = 10.0


def _fit_gamma_moments(column):
    """Fit a 3-parameter gamma by MLE and return its ``(mean, variance)``.

    The reference implementation summarizes each scramble distribution this way,
    and the fitted moments (not the sample moments) go into the z-score.

    Two safeguards are applied that the reference implementation lacks, because
    ``scipy.stats.gamma.fit`` is not reliable on these distributions:

    1. A fit that raises - notably ``FitError``, which the reference
       implementation propagates and which aborts the whole analysis - falls back
       to the sample moments.
    2. A fit that *converges* but to a clearly wrong answer is also rejected.
       The 3-parameter MLE with a free location occasionally lands in a bad local
       optimum and returns moments that do not describe the data at all (a mean
       6x and a variance 170x the sample values has been observed), which
       silently produces a z-score that is wrong by more than an order of
       magnitude. Fitted moments are therefore sanity-checked against the sample
       moments and discarded if they disagree beyond
       :data:`_GAMMA_MEAN_TOLERANCE` / :data:`_GAMMA_VAR_TOLERANCE`.

    Returns
    -------
    tuple of float
        ``(mean, variance)`` of the fitted gamma, or of the sample if the fit was
        rejected. ``(nan, nan)`` if the input is not finite.
    """
    if not np.all(np.isfinite(column)):
        return float("nan"), float("nan")

    sample_mean = float(np.mean(column))
    sample_var = float(np.var(column))

    # A gamma cannot be fit to a point mass, and there is no scale to z-score
    # against. Test the spread rather than `sample_var == 0`: np.var over N
    # identical values is not exactly zero, because subtracting the mean rounds.
    # A 2000-scramble null of a single repeated value came back with a variance
    # of 3e-30, which sailed past an equality test and left the caller dividing a
    # ~1e-15 numerator by a ~1e-15 denominator - reporting a confident-looking
    # z of 2.0 for a statistic that cannot vary at all.
    if np.ptp(column) == 0:
        return sample_mean, 0.0
    if np.sqrt(sample_var) <= _DEGENERATE_SCALE * max(1.0, abs(sample_mean)):
        return sample_mean, 0.0

    try:
        shape, loc, scale = stats.gamma.fit(column)
        fitted_mean = float(stats.gamma.mean(shape, loc, scale))
        fitted_var = float(stats.gamma.var(shape, loc, scale))
    except Exception:
        return sample_mean, sample_var

    if not np.isfinite(fitted_mean) or not np.isfinite(fitted_var) or fitted_var <= 0:
        return sample_mean, sample_var

    # reject a converged-but-nonsense fit
    if sample_mean > 0:
        mean_ratio = fitted_mean / sample_mean
        if not (1.0 / _GAMMA_MEAN_TOLERANCE <= mean_ratio <= _GAMMA_MEAN_TOLERANCE):
            return sample_mean, sample_var

    var_ratio = fitted_var / sample_var
    if not (1.0 / _GAMMA_VAR_TOLERANCE <= var_ratio <= _GAMMA_VAR_TOLERANCE):
        return sample_mean, sample_var

    return fitted_mean, fitted_var


def patterning_zscores(
    sequence,
    num_scrambles=DEFAULT_NUM_SCRAMBLES,
    typeset=DEFAULT_TYPESET,
    blob_size=DEFAULT_BLOB_SIZE,
    min_fraction=DEFAULT_MIN_FRACTION,
    seed=None,
    backend=DEFAULT_BACKEND,
    return_details=False,
):
    """Compute patterning z-scores for one sequence.

    The null distribution is built by scrambling the sequence ``num_scrambles``
    times, so it is specific to this sequence's composition and length. No
    external background is needed or used.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    num_scrambles : int, optional
        Number of scrambles used for the null distribution. Default 10,000. The
        published protocol uses 100,000; 1,000 is usually adequate for
        low-complexity sequences.
    typeset : {'8', '9'} or dict, optional
        Residue grouping, see :func:`sparrow.nardini.groups.get_typeset`.
    blob_size : int, optional
        Sliding-window size. Default 5. Ignored by the ``'iwd'`` backend.
    min_fraction : float, optional
        Abundance threshold below which a feature is not evaluated (its z-score
        is reported as 0.0). Default 0.10.
    seed : int or None, optional
        Seed for the scramble generator. Pass an integer for reproducible
        results. Default None.
    backend : str, optional
        Which patterning statistic to use - see :data:`PATTERNING_BACKENDS`.
        Default ``'nardini'``, which is the only one compatible with GIN cluster
        assignment.
    return_details : bool, optional
        If True, also return the intermediate quantities (see Returns).
        Default False.

    Returns
    -------
    numpy.ndarray or tuple
        By default, the z-score per feature in canonical feature order (length 36
        for the 8-type grouping).

        If ``return_details`` is True, returns
        ``(zscores, details)`` where ``details`` is a dict with keys:

        * ``'raw'`` : raw asymmetries for the real sequence
        * ``'scramble_values'`` : ``(num_scrambles, n_features)`` null values
        * ``'null_mean'`` / ``'null_var'`` : fitted gamma mean/variance per
          feature (NaN for features that were not evaluated)
        * ``'evaluated'`` : boolean mask of features that cleared
          ``min_fraction``
        * ``'type_fractions'`` : whole-sequence fraction of each residue type

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``num_scrambles`` is less than 1, or the sequence is too short or
        contains non-standard residues.

    Examples
    --------
    >>> from sparrow.nardini.patterning import patterning_zscores
    >>> z = patterning_zscores("MEEEKKKK" * 6, num_scrambles=200, seed=1)
    >>> len(z)
    36
    """
    if num_scrambles < 1:
        raise SparrowException(f"num_scrambles must be >= 1 (got {num_scrambles})")

    backend = resolve_backend(backend)
    groups = get_typeset(typeset)
    num_types = len(groups)
    pairs = _pair_index(num_types)
    n_features = len(pairs)

    raw = patterning_values(
        sequence,
        typeset=typeset,
        blob_size=blob_size,
        min_fraction=min_fraction,
        backend=backend,
    )

    labels = encode_types(sequence, typeset)
    fractions = type_fractions(labels, num_types)
    evaluated = _evaluated_mask(fractions, pairs, min_fraction)

    # build the null distribution from scrambles, in memory-bounded blocks
    rng = np.random.default_rng(seed)
    scramble_values = np.zeros((num_scrambles, n_features), dtype=np.float64)

    if backend == "nardini":
        # Only the evaluated features are ever looked at, so only the residue
        # types they refer to need counting. On a typical IDR that is around a
        # third of the features and half the types, and this is the inner loop of
        # the whole calculation - the published protocol runs it 100,000 times per
        # sequence.
        needed_types = sorted({t for feature, pair in enumerate(pairs) if evaluated[feature] for t in pair})
        slot_of = np.full(num_types, -1, dtype=np.intp)
        for slot, residue_type in enumerate(needed_types):
            slot_of[residue_type] = slot

        # Permuting into a preallocated C-ordered buffer, rather than letting
        # rng.permuted allocate: it returns an F-contiguous array when handed a
        # broadcast view, and downstream row-wise cumsums and FFTs are both
        # slower and - for the FFT - not bit-reproducible across row counts on a
        # non-C-contiguous input.
        scramble_buffer = np.empty((_SCRAMBLE_BLOCK, labels.size), dtype=labels.dtype)

        done = 0
        while done < num_scrambles:
            block = min(_SCRAMBLE_BLOCK, num_scrambles - done)

            # Permute the type labels; equivalent to shuffling the residues, since
            # the asymmetries depend only on which type sits at each position.
            # rng.permuted shuffles each row independently by Fisher-Yates, which
            # is O(n) per row and allocates once. The previous approach - argsort
            # of a block of uniform randoms - was O(n log n) and allocated two
            # large temporaries, and was the single largest cost here.
            permuted = rng.permuted(
                np.broadcast_to(labels, (block, labels.size)),
                axis=1,
                out=scramble_buffer[:block],
            )

            counts = _window_counts(permuted, num_types, blob_size, types=needed_types)
            scramble_values[done : done + block] = _asymmetries_from_counts(
                counts,
                fractions,
                pairs,
                blob_size,
                mask=evaluated,
                slot_of=slot_of,
            )
            done += block
    else:
        # The kappa and iwd backends use the vectorized NumPy reimplementations
        # in sparrow.patterning.vectorized, which score a whole block of
        # scrambles per call rather than one sequence at a time.
        from sparrow.patterning import vectorized as vec

        group_lists = [sorted(residues) for residues in groups.values()]
        residues = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        code_buffer = np.empty((_SCRAMBLE_BLOCK, residues.size), dtype=np.uint8)

        done = 0
        while done < num_scrambles:
            block = min(_SCRAMBLE_BLOCK, num_scrambles - done)

            # same Fisher-Yates shuffle, into a C-ordered buffer, as above
            codes = rng.permuted(
                np.broadcast_to(residues, (block, residues.size)),
                axis=1,
                out=code_buffer[:block],
            )

            for feature, (i, j) in enumerate(pairs):
                if not evaluated[feature]:
                    continue

                group1 = group_lists[i]
                group2 = [] if i == j else group_lists[j]

                if backend == "kappa":
                    ternary = vec.ternarize_winged_batch(codes, group1, group2, blob_size)
                    values = vec.kappa_from_ternary_batch(ternary, blob_size, 1, group2_present=len(group2) > 0)
                    # kappa_x reports -1 when undefined; that means "not scored"
                    values = np.where(values < 0, 0.0, values)
                else:
                    targets = group1 if i == j else sorted(set(group1) | set(group_lists[j]))
                    mask = vec._membership_lookup(targets)[codes]
                    values = vec.iwd_batch(mask)

                scramble_values[done : done + block, feature] = values

            done += block

    # features that were not evaluated stay at zero in the null too
    scramble_values[:, ~evaluated] = 0.0

    null_mean = np.full(n_features, np.nan)
    null_var = np.full(n_features, np.nan)
    for feature in range(n_features):
        if evaluated[feature]:
            null_mean[feature], null_var[feature] = _fit_gamma_moments(scramble_values[:, feature])

    zscores = np.zeros(n_features, dtype=np.float64)
    for feature in range(n_features):
        # a raw value of exactly zero means "not scored" (either gated out, or a
        # genuinely perfectly-uniform arrangement); report 0 rather than a
        # spurious z-score, matching the reference implementation
        if raw[feature] == 0:
            continue
        variance = null_var[feature]
        if not np.isfinite(variance) or variance <= 0:
            continue
        zscores[feature] = (raw[feature] - null_mean[feature]) / np.sqrt(variance)

    if not return_details:
        return zscores

    details = {
        "raw": raw,
        "scramble_values": scramble_values,
        "null_mean": null_mean,
        "null_var": null_var,
        "evaluated": evaluated,
        "type_fractions": fractions,
        "feature_names": patterning_feature_names(typeset),
        "backend": backend,
    }
    return zscores, details
