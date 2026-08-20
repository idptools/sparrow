"""Pure-NumPy implementations of sparrow's patterning metrics.

This module reimplements the kappa and inverse-weighted-distance (IWD) routines
that currently live in the compiled ``sparrow.patterning.kappa`` and
``sparrow.patterning.iwd`` Cython extensions, using vectorized NumPy instead.

Why
---
Two reasons:

* **Speed on repeated evaluation.** The Cython routines score one sequence at a
  time. Anything that needs a null distribution - NARDINI-style scramble
  z-scores, for instance - pays that per-call cost tens of thousands of times.
  The ``*_batch`` functions here evaluate a whole stack of sequences in one pass,
  which is one to two orders of magnitude faster.
* **No compiler needed.** These are a step towards dropping the compiled
  extensions entirely.

Accuracy
--------
The Cython code carries its intermediate kappa quantities in C ``float``
(single precision); this module works in ``float64`` throughout. The two
therefore agree to roughly single-precision tolerance (~1e-6 relative) rather
than bit-for-bit, and where they differ **this implementation is the more
accurate one**. The IWD routines are ``double`` on both sides and agree to
~1e-15.

Layout
------
Every public Cython entry point has a signature-compatible twin here, under the
same name:

* from ``sparrow.patterning.kappa`` -- :func:`calculate_sigma`,
  :func:`calculate_delta`, :func:`kappa_x`
* from ``sparrow.patterning.iwd`` --
  :func:`calculate_average_inverse_distance_from_sequence`,
  :func:`calculate_average_inverse_distance_charge`,
  :func:`calculate_average_bivariate_inverse_distance_charge`
* from ``sparrow.patterning.scd`` -- :func:`compute_scd_x`, :func:`compute_shd`

``sparrow.patterning.patterning.patterning_percentile`` is **not** reimplemented
here: it seeds C's ``srand()`` and draws its shuffles from ``rand()``, so a NumPy
version would return different (statistically equivalent) values for the same
``seed``.

On top of those, this module adds batch entry points with no Cython equivalent -
:func:`kappa_x_batch`, :func:`kappa_from_ternary_batch`, :func:`iwd_batch` and
:func:`ternarize_winged_batch` - which score a whole stack of sequences per call.

Several of these are also algorithmically better than the compiled versions
rather than merely vectorized: IWD's pairwise sum is computed as an
autocorrelation via FFT, SCD likewise (``O(n log n)`` instead of ``O(n^2)``), and
SHD collapses to prefix sums (``O(n)`` instead of ``O(n^2)``).
"""

import numpy as np

from sparrow import sparrow_exceptions

# shuffles are generated in blocks so memory stays bounded regardless of count
_PERMUTATION_BLOCK = 512

__all__ = [
    "calculate_sigma",
    "calculate_delta",
    "kappa_x",
    "kappa_x_batch",
    "kappa_from_ternary_batch",
    "ternarize_winged",
    "ternarize_winged_batch",
    "calculate_average_inverse_distance_from_sequence",
    "calculate_average_inverse_distance_charge",
    "calculate_average_bivariate_inverse_distance_charge",
    "iwd_from_binary_mask",
    "iwd_batch",
    "compute_scd_x",
    "compute_shd",
    "DEFAULT_HYDRO_DICT",
    "patterning_percentile",
    "binary_asymmetry",
]


# =====================================================================
# shared helpers
# =====================================================================


def _as_byte_array(sequence):
    """Return a sequence as a uint8 array of its ASCII codes."""
    return np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)


def _membership_lookup(residues):
    """Return a 256-entry boolean lookup table for a residue collection."""
    table = np.zeros(256, dtype=bool)
    for residue in residues:
        table[ord(residue)] = True
    return table


def ternarize_winged(sequence, group1, group2, window_size):
    """Encode a sequence as a ternary array padded with non-contributing wings.

    Reproduces the ``ternarize`` step inside the Cython ``kappa_x``: positions
    holding a ``group1`` residue become ``+1``, ``group2`` residues become
    ``-1``, everything else ``0``. The array is padded with ``window_size`` zeros
    at each end, which is what lets clusters at the termini contribute to the
    patterning as much as those in the middle.

    If ``group2`` is empty the encoding becomes binary: every residue not in
    ``group1`` is treated as ``-1``.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    group1 : list or str
        Residues forming the first group. Must be non-empty.
    group2 : list or str
        Residues forming the second group. May be empty, see above.
    window_size : int
        Window size, which also sets the wing width.

    Returns
    -------
    numpy.ndarray
        ``int8`` array of length ``2 * window_size + len(sequence)``.
    """
    codes = _as_byte_array(sequence)
    length = codes.size

    ternary = np.zeros(2 * window_size + length, dtype=np.int8)
    core = ternary[window_size : window_size + length]

    in_group1 = _membership_lookup(group1)[codes]

    if len(group2) == 0:
        # binary mode: everything that is not group1 is group2
        core[:] = np.where(in_group1, 1, -1)
    else:
        in_group2 = _membership_lookup(group2)[codes]
        core[:] = np.where(in_group1, 1, np.where(in_group2, -1, 0))

    return ternary


def _window_sums(indicator, window_size, dtype=None):
    """Sliding-window sums along the last axis.

    Parameters
    ----------
    indicator : numpy.ndarray
        Array of shape ``(..., n)``.
    window_size : int
        Window width.
    dtype : numpy.dtype, optional
        Accumulator type for the prefix sum. Default ``int32``, which cannot
        overflow for any realistic sequence and moves half the memory of int64.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(..., n - window_size + 1)``.
    """
    if dtype is None:
        # a running total cannot exceed the length, so int32 is ample and halves
        # the memory traffic of int64 - which is what this costs, not arithmetic
        dtype = np.int32
    prefix = np.zeros(indicator.shape[:-1] + (indicator.shape[-1] + 1,), dtype=dtype)
    np.cumsum(indicator, axis=-1, out=prefix[..., 1:])
    return prefix[..., window_size:] - prefix[..., :-window_size]


def _asymmetry_from_parts(difference, pair_total, total):
    """``(Fa - Fb)^2 / (Fa + Fb)`` given ``a - b`` and ``a + b`` directly.

    The ternary encoding gives both of those from one cumulative sum each - the
    running total of the ternary array is ``a - b``, and of its non-zero
    indicator is ``a + b``. Going through the difference and the sum therefore
    avoids materializing two separate boolean masks and two wide count arrays,
    which is the bulk of the cost on the batch path.
    """
    numerator = np.asarray(difference, dtype=np.float64) ** 2
    denominator = np.asarray(pair_total, dtype=np.float64) * total

    out = np.zeros(np.broadcast(numerator, denominator).shape, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def _asymmetry(count_a, count_b, total):
    """The Das & Pappu sigma: ``(Fa - Fb)^2 / (Fa + Fb)``, zero when both are 0.

    Written in terms of raw counts and a normalizing total, which is exactly
    equivalent to working with the fractions and avoids two divisions.
    """
    count_a = np.asarray(count_a, dtype=np.float64)
    count_b = np.asarray(count_b, dtype=np.float64)

    numerator = (count_a - count_b) ** 2
    denominator = (count_a + count_b) * total

    out = np.zeros(np.broadcast(numerator, denominator).shape, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def _delta_from_ternary(ternary, window_size, sigma):
    """Mean squared deviation of per-window asymmetry from the global asymmetry.

    This is the ``patterning_asymmetry`` (delta) step of the Cython ``kappa_x``,
    evaluated over the *winged* array.

    Parameters
    ----------
    ternary : numpy.ndarray
        Ternary array of shape ``(..., winged_length)``.
    window_size : int
        Window width.
    sigma : float or numpy.ndarray
        Global compositional asymmetry to compare each window against.

    Returns
    -------
    numpy.ndarray or float
        Delta, with the leading shape of ``ternary``.
    """
    # a - b and a + b straight from the ternary array, rather than counting the
    # +1s and -1s separately
    difference = _window_sums(ternary, window_size)
    pair_total = _window_sums(ternary != 0, window_size)

    window_asymmetry = _asymmetry_from_parts(difference, pair_total, float(window_size))

    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.ndim:
        sigma = sigma[..., None]

    return np.mean((window_asymmetry - sigma) ** 2, axis=-1)


def _deltamax_ternary(ternary, window_size):
    """Build the maximally-segregated ternary array with the same composition.

    The Cython implementation lays out all ``+1`` residues, then all ``0``
    residues, then all ``-1`` residues - i.e. ``(AAA)n-(XXX)m-(BBB)p`` - inside
    the wings. Because it depends only on composition, it is invariant under
    scrambling, which is what lets the batch path compute it once.

    Parameters
    ----------
    ternary : numpy.ndarray
        A winged ternary array (1D).
    window_size : int
        Window width, which is also the wing width.

    Returns
    -------
    numpy.ndarray
        A winged ternary array of the same length, maximally segregated.
    """
    core = ternary[window_size : ternary.size - window_size]

    n_a = int(np.count_nonzero(core == 1))
    n_b = int(np.count_nonzero(core == -1))
    n_c = core.size - n_a - n_b

    segregated = np.zeros_like(ternary)
    segregated[window_size : window_size + n_a] = 1
    segregated[window_size + n_a + n_c : window_size + n_a + n_c + n_b] = -1

    return segregated


# =====================================================================
# kappa
# =====================================================================


def calculate_sigma(sequence, group1, group2):
    """Compositional asymmetry between two residue groups.

    ``(f1 - f2)^2 / (f1 + f2)``, where ``f1`` and ``f2`` are the whole-sequence
    fractions of each group. Tends to 1 when the two groups are highly
    asymmetric in abundance and to 0 when they are balanced.

    Signature-compatible with ``sparrow.patterning.kappa.calculate_sigma``.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    group1, group2 : list or str
        Residue groups.

    Returns
    -------
    float
        Sigma, a non-negative float. Returns 0.0 when neither group is present.

    Examples
    --------
    >>> round(calculate_sigma("EEEEKKKK", ["E"], ["K"]), 6)
    0.0
    """
    codes = _as_byte_array(sequence)
    if codes.size == 0:
        return 0.0

    count_1 = int(np.count_nonzero(_membership_lookup(group1)[codes]))
    count_2 = int(np.count_nonzero(_membership_lookup(group2)[codes]))

    if count_1 + count_2 == 0:
        return 0.0

    return float(_asymmetry(count_1, count_2, float(codes.size)))


def calculate_delta(sequence, group1, group2, window_size):
    """Local deviation in patterning relative to the whole-sequence asymmetry.

    The classic (un-winged) delta of Das & Pappu: the mean squared difference
    between each sliding window's sigma and the whole-sequence sigma.

    Signature-compatible with ``sparrow.patterning.kappa.calculate_delta``. Note
    this is *not* the winged delta used internally by :func:`kappa_x` - they are
    different quantities, as in the Cython module.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    group1, group2 : list or str
        Residue groups.
    window_size : int
        Sliding-window width.

    Returns
    -------
    float
        Delta, a non-negative float.
    """
    codes = _as_byte_array(sequence)
    length = codes.size

    sigma = calculate_sigma(sequence, group1, group2)

    in_1 = _membership_lookup(group1)[codes].astype(np.int64)
    in_2 = _membership_lookup(group2)[codes].astype(np.int64)

    count_a = _window_sums(in_1, window_size)
    count_b = _window_sums(in_2, window_size)

    window_sigma = _asymmetry(count_a, count_b, float(window_size))

    n_blobs = length - window_size + 1
    return float(np.sum((sigma - window_sigma) ** 2) / n_blobs)


def _kappa_from_ternary(ternary, window_size, flatten, deltamax=None):
    """Shared kappa core, operating on an already-winged ternary array."""
    winged_length = ternary.shape[-1]

    total_a = np.count_nonzero(ternary == 1, axis=-1)
    total_b = np.count_nonzero(ternary == -1, axis=-1)
    sigma = _asymmetry(total_a, total_b, float(winged_length))

    delta = _delta_from_ternary(ternary, window_size, sigma)

    if deltamax is None:
        segregated = _deltamax_ternary(np.atleast_2d(ternary)[0], window_size)
        deltamax = _delta_from_ternary(segregated, window_size, sigma)

    deltamax = np.asarray(deltamax, dtype=np.float64)

    kappa = np.divide(
        delta,
        deltamax,
        out=np.full(np.shape(delta), np.inf, dtype=np.float64),
        where=deltamax != 0,
    )

    if flatten:
        kappa = np.minimum(kappa, 1.0)

    return kappa


def kappa_x(sequence, group1, group2, window_size, flatten=0):
    """Generalized kappa: patterning of one residue group against another.

    Kappa is ``delta / delta_max``, where ``delta`` measures how much the local
    (windowed) compositional asymmetry deviates from the whole-sequence
    asymmetry, and ``delta_max`` is that same quantity for the maximally
    segregated sequence of identical composition. Values near 0 mean well-mixed,
    values near 1 mean segregated.

    Signature-compatible with ``sparrow.patterning.kappa.kappa_x``, including its
    sentinel returns and exceptions.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    group1 : list or str
        First residue group. Must be non-empty.
    group2 : list or str
        Second residue group. If empty, kappa becomes a binary statistic in which
        every residue not in ``group1`` counts as the second group.
    window_size : int
        Sliding-window width. Must be >= 2.
    flatten : int, optional
        If non-zero, cap values above 1 at exactly 1. Default 0.

    Returns
    -------
    float
        Kappa. Returns ``-1`` if the window is larger than the sequence, or if a
        required group is absent from the sequence.

    Raises
    ------
    sparrow.sparrow_exceptions.PatterningException
        If ``window_size`` is below 2, or ``group1`` is empty.

    Examples
    --------
    >>> round(kappa_x("EEEEEEEEEEKKKKKKKKKK", ["E"], ["K"], 5, 1), 4)
    1.0
    """
    length = len(sequence)

    if window_size > length:
        return -1

    if window_size < 2:
        raise sparrow_exceptions.PatterningException("window_size must be 2 or larger")

    if len(group1) == 0:
        raise sparrow_exceptions.PatterningException(
            "Passed group1 is empty, group1 must contain one or more possible residues"
        )

    ternary = ternarize_winged(sequence, group1, group2, window_size)

    count_1 = int(np.count_nonzero(ternary == 1))
    count_2 = int(np.count_nonzero(ternary == -1))

    if count_1 == 0 and len(group1) > 0:
        return -1.0
    if count_2 == 0 and len(group2) > 0:
        return -1.0

    return float(_kappa_from_ternary(ternary, window_size, flatten))


def ternarize_winged_batch(codes, group1, group2, window_size):
    """Ternarize a stack of equal-length sequences at once.

    The batch analogue of :func:`ternarize_winged`, taking pre-encoded ASCII
    codes so callers scoring many features over the same scramble set can encode
    once and reuse.

    Parameters
    ----------
    codes : numpy.ndarray
        ``uint8`` array of shape ``(n_sequences, length)`` of ASCII codes.
    group1, group2 : list or str
        Residue groups; ``group2`` may be empty for binary mode.
    window_size : int
        Window size, which also sets the wing width.

    Returns
    -------
    numpy.ndarray
        ``int8`` array of shape ``(n_sequences, 2 * window_size + length)``.
    """
    codes = np.atleast_2d(codes)
    n_sequences, length = codes.shape

    ternary = np.zeros((n_sequences, 2 * window_size + length), dtype=np.int8)
    core = ternary[:, window_size : window_size + length]

    in_group1 = _membership_lookup(group1)[codes]

    if len(group2) == 0:
        core[:] = np.where(in_group1, 1, -1)
    else:
        in_group2 = _membership_lookup(group2)[codes]
        core[:] = np.where(in_group1, 1, np.where(in_group2, -1, 0))

    return ternary


def kappa_from_ternary_batch(ternary, window_size, flatten=0, group2_present=True):
    """Kappa for a stack of already-ternarized sequences.

    The lower-level batch entry point. When every row has the same composition -
    as for a set of scrambles - ``delta_max`` is identical across rows and is
    computed once, which is where most of the speed-up comes from.

    Parameters
    ----------
    ternary : numpy.ndarray
        Ternary array of shape ``(n_sequences, winged_length)``, from
        :func:`ternarize_winged_batch`.
    window_size : int
        Sliding-window width.
    flatten : int, optional
        If non-zero, cap values above 1 at exactly 1. Default 0.
    group2_present : bool, optional
        Whether the caller supplied a non-empty ``group2``. Controls the
        "group absent" sentinel, matching :func:`kappa_x`. Default True.

    Returns
    -------
    numpy.ndarray
        Kappa per row, with ``-1`` where a required group is absent.
    """
    if window_size < 2:
        raise sparrow_exceptions.PatterningException("window_size must be 2 or larger")

    ternary = np.atleast_2d(ternary)
    winged_length = ternary.shape[-1]

    count_1 = np.count_nonzero(ternary == 1, axis=-1)
    count_2 = np.count_nonzero(ternary == -1, axis=-1)

    uniform = bool(np.all(count_1 == count_1[0]) and np.all(count_2 == count_2[0]))

    if uniform:
        # one delta_max for the whole stack
        sigma = _asymmetry(count_1[0], count_2[0], float(winged_length))
        deltamax = _delta_from_ternary(_deltamax_ternary(ternary[0], window_size), window_size, sigma)
    else:
        deltamax = np.array(
            [
                _delta_from_ternary(
                    _deltamax_ternary(row, window_size),
                    window_size,
                    _asymmetry(c1, c2, float(winged_length)),
                )
                for row, c1, c2 in zip(ternary, count_1, count_2)
            ]
        )

    kappa = _kappa_from_ternary(ternary, window_size, flatten, deltamax=deltamax)

    # reproduce kappa_x's sentinel for an absent group
    undefined = count_1 == 0
    if group2_present:
        undefined = undefined | (count_2 == 0)

    return np.where(undefined, -1.0, kappa)


def kappa_x_batch(sequences, group1, group2, window_size, flatten=0):
    """Compute :func:`kappa_x` for many equal-length sequences at once.

    This is the batch entry point that has no Cython equivalent, and a large part
    of why this module exists.

    Parameters
    ----------
    sequences : iterable of str
        Equal-length amino acid sequences.
    group1 : list or str
        First residue group. Must be non-empty.
    group2 : list or str
        Second residue group. If empty, kappa becomes binary (see
        :func:`kappa_x`).
    window_size : int
        Sliding-window width.
    flatten : int, optional
        If non-zero, cap values above 1 at exactly 1. Default 0.

    Returns
    -------
    numpy.ndarray
        Kappa per input sequence, shape ``(n_sequences,)``. Entries where a
        required group is absent are ``-1``, matching :func:`kappa_x`.

    Raises
    ------
    sparrow.sparrow_exceptions.PatterningException
        If the sequences are not all the same length, or ``window_size`` is
        below 2.

    Examples
    --------
    >>> vals = kappa_x_batch(["EEEEEKKKKK", "EKEKEKEKEK"], ["E"], ["K"], 5, 1)
    >>> bool(vals[0] > vals[1])
    True
    """
    sequences = list(sequences)
    if not sequences:
        return np.zeros(0, dtype=np.float64)

    lengths = {len(s) for s in sequences}
    if len(lengths) != 1:
        raise sparrow_exceptions.PatterningException("kappa_x_batch requires all sequences to be the same length")

    if window_size > lengths.pop():
        return np.full(len(sequences), -1.0, dtype=np.float64)

    codes = np.stack([_as_byte_array(s) for s in sequences])
    ternary = ternarize_winged_batch(codes, group1, group2, window_size)

    return kappa_from_ternary_batch(ternary, window_size, flatten, group2_present=len(group2) > 0)


def binary_asymmetry(sequence, group):
    """Whole-sequence binary compositional asymmetry, ``(f - (1 - f))^2``.

    The statistic underlying :func:`patterning_percentile`, where ``f`` is the
    fraction of the sequence made up of ``group`` residues. Unlike
    :func:`calculate_sigma` there is no second group and no normalizing
    denominator - everything not in ``group`` is the other class.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    group : list or str
        Residues forming the positive class.

    Returns
    -------
    float
        The asymmetry, between 0 and 1.
    """
    codes = _as_byte_array(sequence)
    if codes.size == 0:
        return 0.0

    n_positive = int(np.count_nonzero(_membership_lookup(group)[codes]))
    fraction = n_positive / codes.size
    return float((fraction - (1.0 - fraction)) ** 2)


def patterning_percentile(sequence, group, window_size, count, seed, return_distribution=False):
    """Percentile of a sequence's binary patterning against random shuffles.

    Binarizes the sequence into ``group`` vs. everything else, measures how much
    the local (windowed) composition deviates from the whole-sequence
    composition, and asks where that sits in the distribution obtained from
    ``count`` random shuffles of the same sequence.

    Returns a value between 0 and 1: **0** means the real sequence is far more
    evenly patterned than chance, **1** means far blockier than chance.

    Signature-compatible with
    ``sparrow.patterning.patterning.patterning_percentile``.

    .. warning::
       The compiled version draws its shuffles from C's ``rand()``. The C
       standard leaves that generator's algorithm implementation-defined, so its
       ``seed`` -> result mapping is not portable across platforms or libc
       versions in the first place. This implementation uses NumPy's generator,
       so **the same** ``seed`` **gives a different (statistically equivalent)
       set of shuffles** than the compiled version. Percentiles agree to within
       sampling error, not exactly. Do not compare a stored value from one
       implementation against the other; recompute instead.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    group : list or str
        Residues forming the positive class. Must be non-empty.
    window_size : int
        Sliding-window width. Must be >= 2 and no larger than the sequence.
    count : int
        Number of random shuffles making up the null distribution.
    seed : int
        Seed for the shuffle generator.
    return_distribution : bool, optional
        If True, also return the observed statistic and the sorted null
        distribution. Default False.

    Returns
    -------
    float or tuple
        The percentile. If ``return_distribution`` is True, returns
        ``(percentile, observed, sorted_null)``.

    Raises
    ------
    sparrow.sparrow_exceptions.PatterningException
        If the window is larger than the sequence or smaller than 2, if
        ``group`` is empty, or if ``seed`` is not integer-convertible.

    Examples
    --------
    >>> blocky = patterning_percentile("Q" * 20 + "A" * 20, ["Q"], 5, 500, 1)
    >>> mixed = patterning_percentile("QA" * 20, ["Q"], 5, 500, 1)
    >>> bool(blocky > mixed)
    True
    """
    length = len(sequence)

    if window_size > length:
        raise sparrow_exceptions.PatterningException("window_size is larger than sequence length")
    if window_size < 2:
        raise sparrow_exceptions.PatterningException("window_size must be 2 or larger")

    try:
        seed = int(seed)
    except (TypeError, ValueError):
        raise sparrow_exceptions.PatterningException("Cannot convert seed into an integer")

    if len(group) == 0:
        raise sparrow_exceptions.PatterningException(
            "Passed group is empty, group must contain one or more possible residues"
        )

    # +1 / -1 encoding with no wings. For a two-class split this makes the
    # generic _asymmetry helper reduce exactly to (f - (1 - f))^2, so the same
    # windowing machinery used elsewhere in this module applies unchanged.
    codes = _as_byte_array(sequence)
    binary = np.where(_membership_lookup(group)[codes], 1, -1).astype(np.int8)

    n_positive = int(np.count_nonzero(binary == 1))
    sigma = _asymmetry(n_positive, length - n_positive, float(length))

    observed = float(_delta_from_ternary(binary, window_size, sigma))

    rng = np.random.default_rng(seed)
    null = np.empty(count, dtype=np.float64)

    done = 0
    while done < count:
        block = min(_PERMUTATION_BLOCK, count - done)
        order = rng.random((block, length)).argsort(axis=1)
        null[done : done + block] = _delta_from_ternary(binary[order], window_size, sigma)
        done += block

    null.sort()

    # the fraction of shuffles no blockier than the real sequence
    percentile = float(np.count_nonzero(null <= observed) / count)

    if return_distribution:
        return percentile, observed, null

    return percentile


# =====================================================================
# inverse-weighted distance
# =====================================================================


def _inverse_distance_kernel(length):
    """Return ``[0, 1, 1/2, 1/3, ...]`` up to ``length - 1``."""
    kernel = np.zeros(length, dtype=np.float64)
    if length > 1:
        kernel[1:] = 1.0 / np.arange(1, length, dtype=np.float64)
    return kernel


def iwd_from_binary_mask(binary_mask, weights=None):
    """Inverse-weighted distance for a binary occupancy mask.

    For every marked position, sum ``1 / distance`` to every other marked
    position; average those sums, weighted by ``weights``.

    Reproduces the Cython ``__compute_IWD_from_binary_mask``.

    Parameters
    ----------
    binary_mask : array_like
        1D array of 0/1 (or boolean) values.
    weights : array_like, optional
        Per-position weights, same length as the mask. Defaults to uniform.

    Returns
    -------
    float
        The IWD, a non-negative float. Returns 0.0 for an empty mask.
    """
    mask = np.asarray(binary_mask)
    positions = np.flatnonzero(mask == 1)

    if positions.size == 0:
        return 0.0

    # pairwise inverse distances; the diagonal (self-distance) is excluded
    separation = np.abs(positions[:, None] - positions[None, :]).astype(np.float64)
    with np.errstate(divide="ignore"):
        inverse = np.where(separation > 0, 1.0 / separation, 0.0)
    per_hit = inverse.sum(axis=1)

    if weights is None:
        return float(per_hit.sum() / positions.size)

    hit_weights = np.asarray(weights, dtype=np.float64)[positions]
    weight_sum = hit_weights.sum()
    if weight_sum == 0:
        return 0.0

    return float(np.dot(per_hit, hit_weights) / weight_sum)


def calculate_average_inverse_distance_from_sequence(sequence, target_residues):
    """Inverse-weighted distance between the target residues of a sequence.

    Signature-compatible with
    ``sparrow.patterning.iwd.calculate_average_inverse_distance_from_sequence``.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    target_residues : list or str
        Residues that count as occupied positions.

    Returns
    -------
    float
        The IWD, a non-negative float.

    Examples
    --------
    >>> clustered = calculate_average_inverse_distance_from_sequence("AAAAAQQQQQ", ["Q"])
    >>> spread = calculate_average_inverse_distance_from_sequence("AQAQAQAQAQ", ["Q"])
    >>> bool(clustered > spread)
    True
    """
    codes = _as_byte_array(sequence)
    if codes.size == 0:
        return 0.0

    mask = _membership_lookup(target_residues)[codes]
    return iwd_from_binary_mask(mask)


def iwd_batch(masks):
    """Inverse-weighted distance for a stack of binary masks, in one pass.

    Uses the identity that the sum of inverse pairwise distances equals
    ``sum_d occupancy_autocorrelation(d) / d``, so the whole stack is evaluated
    with a single batched FFT rather than an explicit pairwise loop per row. This
    is the batch entry point with no Cython equivalent.

    Parameters
    ----------
    masks : array_like
        Array of shape ``(n_masks, length)`` of 0/1 (or boolean) values.

    Returns
    -------
    numpy.ndarray
        IWD per mask, shape ``(n_masks,)``. Rows with no marked position give
        0.0.

    Examples
    --------
    >>> import numpy as np
    >>> vals = iwd_batch(np.array([[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0]]))
    >>> bool(vals[0] > vals[1])
    True
    """
    # ascontiguousarray, not just astype: pocketfft takes a different path for a
    # non-C-contiguous input, and the results then differ in the last bit between
    # rows of the same call. That is invisible almost everywhere, but it turns a
    # null distribution that should be an exact point mass into one with ~1e-16
    # of spread, which the degenerate-null guard in sparrow.nardini.patterning
    # then has to cope with.
    masks = np.ascontiguousarray(np.atleast_2d(np.asarray(masks)), dtype=np.float64)
    n_masks, length = masks.shape

    counts = masks.sum(axis=1)

    # autocorrelation of each row via FFT: A[d] = sum_i mask[i] * mask[i + d]
    size = 1
    while size < 2 * length:
        size *= 2
    spectrum = np.fft.rfft(masks, n=size, axis=1)
    autocorr = np.fft.irfft(spectrum * np.conjugate(spectrum), n=size, axis=1)
    # A[d] is a count of residue pairs, so it is an integer; rounding away the
    # FFT's ~1e-17 noise keeps sparse masks giving exactly 0.0
    autocorr = np.rint(autocorr[:, :length])

    # sum over separations of pairs-at-distance-d / d. Each unordered pair is
    # counted once by the autocorrelation, and each contributes to two hits'
    # per-hit sums, hence the factor of 2.
    kernel = _inverse_distance_kernel(length)
    pair_sum = 2.0 * (autocorr @ kernel)

    out = np.zeros(n_masks, dtype=np.float64)
    np.divide(pair_sum, counts, out=out, where=counts > 0)
    return out


def calculate_average_inverse_distance_charge(linear_NCPR, sequence, charge):
    """Charge-weighted IWD for the positive or negative residues of a sequence.

    Each residue's contribution is weighted by the magnitude of the local net
    charge per residue at its position, and residues whose local NCPR has the
    wrong sign contribute nothing.

    Signature-compatible with
    ``sparrow.patterning.iwd.calculate_average_inverse_distance_charge``.

    Parameters
    ----------
    linear_NCPR : array_like
        Per-residue local net charge per residue across the sequence.
    sequence : str
        Amino acid sequence.
    charge : {'-', '+'}
        ``'-'`` to score clustering of D/E, ``'+'`` for R/K.

    Returns
    -------
    float
        The charge-weighted IWD, a non-negative float.
    """
    codes = _as_byte_array(sequence)
    ncpr = np.asarray(linear_NCPR, dtype=np.float64)

    if charge == "-":
        selected = _membership_lookup("DE")[codes]
        wanted_sign = -1.0
    else:
        selected = _membership_lookup("RK")[codes]
        wanted_sign = 1.0

    positions = np.flatnonzero(selected)
    if positions.size == 0:
        return 0.0

    separation = np.abs(positions[:, None] - positions[None, :]).astype(np.float64)
    with np.errstate(divide="ignore"):
        inverse = np.where(separation > 0, 1.0 / separation, 0.0)
    per_hit = inverse.sum(axis=1)

    local_charge = ncpr[positions]
    # only residues sitting in a correctly-signed local environment contribute
    contributing = np.sign(local_charge) == wanted_sign
    weights = np.where(contributing, np.abs(local_charge), 0.0)

    # NB: the denominator is the residue count, not the weight sum
    return float(np.dot(per_hit, weights) / positions.size)


def calculate_average_bivariate_inverse_distance_charge(linear_NCPR, sequence):
    """Charge-weighted IWD between the positive and negative residues.

    Quantifies how intermixed oppositely-charged residues are, weighting each
    positive/negative pair by the difference in their local net charge per
    residue and dividing by their separation.

    Signature-compatible with
    ``sparrow.patterning.iwd.calculate_average_bivariate_inverse_distance_charge``.

    Parameters
    ----------
    linear_NCPR : array_like
        Per-residue local net charge per residue across the sequence.
    sequence : str
        Amino acid sequence.

    Returns
    -------
    float
        The bivariate charge-weighted IWD, a non-negative float. Returns 0.0 if
        the sequence lacks either charge type.
    """
    codes = _as_byte_array(sequence)
    ncpr = np.asarray(linear_NCPR, dtype=np.float64)

    negative_positions = np.flatnonzero(_membership_lookup("DE")[codes])
    positive_positions = np.flatnonzero(_membership_lookup("RK")[codes])

    if negative_positions.size == 0 or positive_positions.size == 0:
        return 0.0

    # a residue only carries its charge if the local environment agrees in sign
    masked = np.zeros_like(ncpr)
    is_negative = _membership_lookup("DE")[codes] & (ncpr < 0)
    is_positive = _membership_lookup("RK")[codes] & (ncpr > 0)
    masked[is_negative] = ncpr[is_negative]
    masked[is_positive] = ncpr[is_positive]

    charge_difference = np.abs(masked[positive_positions][None, :] - masked[negative_positions][:, None])
    separation = np.abs(positive_positions[None, :] - negative_positions[:, None]).astype(np.float64)

    return float(np.sum(charge_difference / separation) / negative_positions.size)


# =====================================================================
# sequence charge / hydropathy decoration
# =====================================================================

#: Normalized Kyte-Doolittle hydrophobicity used by :func:`compute_shd` when no
#: scale is supplied. Matches ``sparrow.patterning.scd``.
DEFAULT_HYDRO_DICT = {
    "A": 0.730,
    "R": 0.000,
    "N": 0.432,
    "D": 0.378,
    "C": 0.595,
    "Q": 0.514,
    "E": 0.459,
    "G": 0.649,
    "H": 0.514,
    "I": 0.973,
    "L": 0.973,
    "K": 0.514,
    "M": 0.838,
    "F": 1.000,
    "P": 1.000,
    "S": 0.595,
    "T": 0.676,
    "W": 0.946,
    "Y": 0.865,
    "V": 0.892,
}


def compute_scd_x(sequence, group1=("E", "D"), group2=("R", "K")):
    """Sequence charge decoration (SCD) generalized to arbitrary residue groups.

    The Sawle & Ghosh charge-patterning parameter: for every residue pair, the
    product of their group charges weighted by the square root of their
    separation, averaged over the sequence.

    Signature-compatible with ``sparrow.patterning.scd.compute_scd_x``.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    group1 : iterable of str, optional
        Residues assigned charge ``-1``. Default ``('E', 'D')``.
    group2 : iterable of str, optional
        Residues assigned charge ``+1``. Default ``('R', 'K')``.

    Returns
    -------
    float
        The sequence charge decoration.

    Examples
    --------
    >>> blocky = compute_scd_x("EEEEEEEEEEKKKKKKKKKK")
    >>> mixed = compute_scd_x("EKEKEKEKEKEKEKEKEKEK")
    >>> bool(blocky < mixed)
    True
    """
    codes = _as_byte_array(sequence)
    length = codes.size
    if length < 2:
        return 0.0

    charge_table = np.zeros(256, dtype=np.float64)
    for residue in group1:
        charge_table[ord(residue)] = -1.0
    for residue in group2:
        charge_table[ord(residue)] = 1.0

    charges = charge_table[codes]

    # SCD sums q_m * q_n * sqrt(m - n) over pairs m > n. Grouping by separation
    # d = m - n turns the inner sum into the autocorrelation of the charge
    # sequence at lag d, so the whole O(n^2) double loop collapses to one FFT.
    size = 1
    while size < 2 * length:
        size *= 2
    spectrum = np.fft.rfft(charges, n=size)
    autocorr = np.fft.irfft(spectrum * np.conjugate(spectrum), n=size)[:length]
    # charges are -1/0/+1, so each lag sum is an integer; rounding removes the
    # FFT's floating-point noise
    autocorr = np.rint(autocorr)

    separations = np.sqrt(np.arange(length, dtype=np.float64))
    total = float(np.dot(autocorr[1:], separations[1:]))

    return total / length


def compute_shd(seq, hydro_dict=None):
    """Sequence hydropathy decoration (SHD).

    The Zheng et al. hydrophobicity-patterning parameter: for every residue pair,
    the sum of their hydropathies divided by their separation, averaged over the
    sequence.

    Signature-compatible with ``sparrow.patterning.scd.compute_shd``.

    Parameters
    ----------
    seq : str
        Amino acid sequence.
    hydro_dict : dict, optional
        Mapping from residue to hydrophobicity. Defaults to
        :data:`DEFAULT_HYDRO_DICT` (normalized Kyte-Doolittle).

    Returns
    -------
    float
        The sequence hydropathy decoration.

    Raises
    ------
    ValueError
        If a residue in the sequence is absent from ``hydro_dict``.
    """
    if hydro_dict is None:
        hydro_dict = DEFAULT_HYDRO_DICT

    for residue in set(seq):
        if residue not in hydro_dict:
            raise ValueError(f"When calculating SHD the hydrophobicity dictionary lacked the residue {residue}")

    length = len(seq)
    if length < 2:
        return 0.0

    hydropathy = np.array([hydro_dict[residue] for residue in seq], dtype=np.float64)

    # SHD sums (h_m + h_n) / (m - n) over pairs m > n. Because the numerator is
    # a sum rather than a product, grouping by separation d gives
    #     sum_d (1/d) * [ sum_{m>=d} h_m  +  sum_{n<=N-1-d} h_n ]
    # and both bracketed terms are just prefix sums, so this is O(n) rather than
    # the O(n^2) double loop.
    prefix = np.concatenate([[0.0], np.cumsum(hydropathy)])
    total_sum = prefix[-1]

    separations = np.arange(1, length, dtype=np.float64)
    # sum of h over indices >= d, and over indices <= N-1-d
    upper = total_sum - prefix[1:length]
    lower = prefix[length - np.arange(1, length)]

    total = float(np.sum((upper + lower) / separations))
    return total / length
