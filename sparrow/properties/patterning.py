"""Patterning: how a sequence's residues are arranged, not just what they are.

One statistic underlies most of this module, and NARDINI's patterning features
too: the windowed compositional asymmetry of Das & Pappu (2013).

* The **asymmetry** between two groups of residues, ``a`` and ``b``, is
  ``sigma = (f_a - f_b)^2 / (f_a + f_b)``, where ``f`` is each group's fraction.
  In counts it is ``(a - b)^2 / ((a + b) * total)`` - :func:`asymmetry`.
* **delta** is the mean, over every window of ``window_size`` consecutive
  residues, of the squared difference between the window's asymmetry and the
  whole sequence's - :func:`window_delta`. It is large when the two groups are
  segregated into blocks, small when they are mixed.

Everything else is delta in a particular setting:

* :func:`delta` - the original Das & Pappu delta, for two residue groups;
* :func:`kappa_x` - delta of a sequence padded with neutral wings, divided by
  the delta of the most segregated arrangement of the same composition, so it
  runs from 0 (well mixed) to about 1 (fully segregated); :func:`kappa` is the
  charge-patterning kappa built from it;
* :func:`patterning_percentile` - where one group's delta (against everything
  else) sits among random shuffles of the sequence;
* NARDINI's omega and pair features (:mod:`sparrow.nardini.patterning`), which
  call :func:`asymmetry` and :func:`window_delta` directly on window counts.

The module also holds the other patterning metrics: the inverse-weighted
distance (:func:`iwd` and its charge-weighted forms), and the sequence charge
and hydropathy decorations (:func:`scd`, :func:`shd`). IWD and SCD sum over every
pair of residues, which grouping the pairs by separation turns into an
autocorrelation - one FFT rather than ``O(N^2)`` work - and SHD collapses further,
to prefix sums.

Batches
-------
Every sequence-level function here takes a batch and returns one value per
sequence. The batch is split into groups of equal length
(:meth:`~sparrow.properties.encoding.EncodedBatch.by_length`), and each group is
scored as one ``(k, length)`` array. Grouping by exact length, rather than
padding to a common one, keeps every sequence's value independent of what else is
in the batch: a padded row would be summed in a different order and differ in
the last bit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from sparrow.data import amino_acids
from sparrow.properties import encoding, profiles
from sparrow.properties.composition import Batch, as_counts
from sparrow.properties.windows import window_sums
from sparrow.sparrow_exceptions import PatterningException, SparrowException

__all__ = [
    "KAPPA_WINDOW_SIZES",
    "asymmetry",
    "window_delta",
    "ternary_delta",
    "segregated_arrangements",
    "kappa_from_ternary",
    "sigma",
    "delta",
    "kappa_x",
    "kappa",
    "binary_asymmetry",
    "patterning_percentile",
    "CHARGE_IWD_WINDOW",
    "iwd_from_masks",
    "iwd",
    "iwd_charged_weighted",
    "bivariate_iwd_charged_weighted",
    "scd",
    "shd",
]

#: Window sizes whose kappa values are averaged for :func:`kappa`, the charge
#: patterning parameter, following Das & Pappu (2013).
KAPPA_WINDOW_SIZES = (5, 6)

#: Shuffles are generated in blocks of this many, so memory stays bounded however
#: many are asked for.
_SHUFFLE_BLOCK = 512

#: Sentinel :func:`kappa_x` returns when kappa is undefined for a sequence.
KAPPA_UNDEFINED = -1.0


# =====================================================================
# the windowed asymmetry
# =====================================================================


def asymmetry(
    difference: NDArray[np.generic] | float,
    pair_total: NDArray[np.generic] | float,
    total: NDArray[np.generic] | float,
) -> NDArray[np.float64]:
    """Das & Pappu's compositional asymmetry between two groups, from counts.

    ``sigma = (f_a - f_b)^2 / (f_a + f_b)`` with ``f = count / total``, which in
    counts is ``(a - b)^2 / ((a + b) * total)``. Working in counts means one
    rounding rather than several, and lets the caller pass ``a - b`` and
    ``a + b`` directly when it has them - a ternary encoding gives both from one
    window sum each.

    Parameters
    ----------
    difference : array_like
        ``a - b``.
    pair_total : array_like
        ``a + b``.
    total : array_like
        The length the fractions are taken over: the window size for a window,
        the sequence length for a whole sequence.

    Returns
    -------
    numpy.ndarray
        The asymmetry, between 0 and 1; 0.0 where ``a + b`` is zero.

    References
    ----------
    Das RK, Pappu RV. Conformations of intrinsically disordered proteins are
    influenced by linear sequence distributions of oppositely charged residues.
    PNAS. 2013;110(33):13392-13397.
    """
    numerator = np.asarray(difference, dtype=np.float64) ** 2
    denominator = np.asarray(pair_total, dtype=np.float64) * np.asarray(total, dtype=np.float64)

    if denominator.ndim == 0 and denominator > 0:
        # every window holds the same number of the two groups (as for NARDINI's
        # omega, one type against all the rest), so nothing needs masking
        quotient: NDArray[np.float64] = numerator / denominator
        return quotient

    out = np.zeros(np.broadcast(numerator, denominator).shape, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator > 0)
    return out


def window_delta(
    difference: NDArray[np.generic],
    pair_total: NDArray[np.generic] | float,
    window_size: int,
    sigma: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """delta: the mean squared deviation of each window's asymmetry from ``sigma``.

    Parameters
    ----------
    difference : numpy.ndarray
        ``a - b`` in every window, shape ``(..., n_windows)``.
    pair_total : numpy.ndarray or float
        ``a + b`` in every window, the same shape (or a scalar, when every window
        holds the same number of the two groups).
    window_size : int
        Window width.
    sigma : float or numpy.ndarray
        The whole-sequence asymmetry to compare each window against: a scalar,
        or one per row (shape ``(...)``).

    Returns
    -------
    numpy.ndarray
        delta, with the leading shape of ``difference``.
    """
    window_asymmetry = asymmetry(difference, pair_total, float(window_size))

    reference = np.asarray(sigma, dtype=np.float64)
    if reference.ndim:
        reference = reference[..., None]

    result: NDArray[np.float64] = np.mean((window_asymmetry - reference) ** 2, axis=-1)
    return result


def ternary_delta(
    ternary: NDArray[np.int8], window_size: int, sigma: NDArray[np.float64] | float
) -> NDArray[np.float64]:
    """delta of a ternary (+1 / -1 / 0) arrangement.

    The window sums of the ternary array are ``a - b``, and those of its non-zero
    indicator are ``a + b``, so neither group needs counting on its own.

    Parameters
    ----------
    ternary : numpy.ndarray
        Ternary array of shape ``(..., length)``.
    window_size : int
        Window width.
    sigma : float or numpy.ndarray
        Whole-sequence asymmetry, a scalar or one per row.

    Returns
    -------
    numpy.ndarray
        delta, with the leading shape of ``ternary``.
    """
    difference = window_sums(ternary, window_size)
    pair_total = window_sums(ternary != 0, window_size)
    return window_delta(difference, pair_total, window_size, sigma)


# =====================================================================
# kappa
# =====================================================================


def segregated_arrangements(ternary: NDArray[np.int8], window_size: int) -> NDArray[np.int8]:
    """The maximally segregated arrangement of each row's own composition.

    All the ``+1`` residues, then all the ``0`` residues, then all the ``-1``
    residues - ``(AAA)n-(XXX)m-(BBB)p`` - inside the same neutral wings. Its delta
    is ``delta_max``, the normalizer of kappa. It depends only on composition,
    which scrambling preserves.

    Parameters
    ----------
    ternary : numpy.ndarray
        Winged ternary arrays, shape ``(n, winged_length)``.
    window_size : int
        Window width, which is also the wing width.

    Returns
    -------
    numpy.ndarray
        ``int8`` array of the same shape.
    """
    ternary = np.atleast_2d(ternary)
    length = ternary.shape[1] - 2 * window_size
    core = ternary[:, window_size : window_size + length]
    end = window_size + length

    n_a = np.count_nonzero(core == 1, axis=1)
    n_b = np.count_nonzero(core == -1, axis=1)

    segregated = np.zeros_like(ternary)
    if ternary.shape[0] == 1:
        # the common case - one arrangement, as for the scrambles of one
        # sequence - is two slice assignments
        segregated[0, window_size : window_size + n_a[0]] = 1
        segregated[0, end - n_b[0] : end] = -1
        return segregated

    position = np.arange(ternary.shape[1])[None, :]
    segregated[(position >= window_size) & (position < window_size + n_a[:, None])] = 1
    segregated[(position >= end - n_b[:, None]) & (position < end)] = -1
    return segregated


def kappa_from_ternary(
    ternary: NDArray[np.int8],
    window_size: int,
    flatten: bool = False,
    group2_present: bool = True,
) -> NDArray[np.float64]:
    """kappa for a stack of winged ternary arrays of the same length.

    ``delta / delta_max``. When every row has the same composition - as for a set
    of scrambles of one sequence - ``delta_max`` is the same for every row and is
    computed once, which is most of the saving on a scramble path.

    Parameters
    ----------
    ternary : numpy.ndarray
        Winged ternary arrays of shape ``(n, winged_length)``, from
        :func:`sparrow.properties.encoding.ternarize_winged_batch`.
    window_size : int
        Window width. Must be at least 2.
    flatten : bool, optional
        If True, cap values above 1 at exactly 1. Default False.
    group2_present : bool, optional
        Whether a non-empty second group was used. With no second group the
        encoding is binary, and a missing second group is not undefined.
        Default True.

    Returns
    -------
    numpy.ndarray
        kappa per row, or ``KAPPA_UNDEFINED`` (-1) where a group is absent.

    Raises
    ------
    sparrow.sparrow_exceptions.PatterningException
        If ``window_size`` is below 2.
    """
    if window_size < 2:
        raise PatterningException("window_size must be 2 or larger")

    ternary = np.atleast_2d(ternary)
    winged_length = ternary.shape[-1]

    count_1 = np.count_nonzero(ternary == 1, axis=-1)
    count_2 = np.count_nonzero(ternary == -1, axis=-1)
    sigma_all = asymmetry(count_1 - count_2, count_1 + count_2, float(winged_length))

    delta_seq = ternary_delta(ternary, window_size, sigma_all)

    if np.all(count_1 == count_1[0]) and np.all(count_2 == count_2[0]):
        # one composition, so one delta_max for the whole stack
        delta_max = ternary_delta(segregated_arrangements(ternary[:1], window_size)[0], window_size, sigma_all[0])
    else:
        delta_max = ternary_delta(segregated_arrangements(ternary, window_size), window_size, sigma_all)

    result = np.divide(
        delta_seq,
        delta_max,
        out=np.full(np.shape(delta_seq), np.inf, dtype=np.float64),
        where=np.asarray(delta_max) != 0,
    )
    if flatten:
        result = np.minimum(result, 1.0)

    undefined = count_1 == 0
    if group2_present:
        undefined = undefined | (count_2 == 0)
    kappas: NDArray[np.float64] = np.where(undefined, KAPPA_UNDEFINED, result)
    return kappas


def _check_kappa_arguments(group1: Sequence[str], window_size: int) -> None:
    if window_size < 2:
        raise PatterningException("window_size must be 2 or larger")
    if len(group1) == 0:
        raise PatterningException("Passed group1 is empty, group1 must contain one or more possible residues")


def kappa_x(
    sequences: str | Sequence[str] | encoding.EncodedBatch,
    group1: Sequence[str],
    group2: Sequence[str],
    window_size: int,
    flatten: bool = False,
) -> NDArray[np.float64]:
    """Generalized kappa: how segregated one group of residues is from another.

    kappa is ``delta / delta_max``, where delta is computed over the sequence
    padded with ``window_size`` neutral residues at each end - so clusters at the
    termini count as much as those in the middle - and ``delta_max`` is the same
    for the most segregated arrangement of the sequence's composition. Values
    near 0 mean well mixed, values near 1 segregated.

    Parameters
    ----------
    sequences : str, sequence of str, or EncodedBatch
        The batch.
    group1 : sequence of str
        First residue group. Must be non-empty.
    group2 : sequence of str
        Second residue group. If empty, kappa is binary: every residue not in
        ``group1`` counts as the second group.
    window_size : int
        Window width. Must be at least 2.
    flatten : bool, optional
        If True, cap values above 1 at exactly 1. Very skewed compositions can
        exceed 1, because the most segregated arrangement then does not quite
        maximize delta. Default False.

    Returns
    -------
    numpy.ndarray
        ``(n_sequences,)`` kappa values; -1 for a sequence shorter than the
        window, or missing a group it needs.

    Raises
    ------
    sparrow.sparrow_exceptions.PatterningException
        If ``window_size`` is below 2 or ``group1`` is empty.

    Examples
    --------
    >>> kappa_x(["EEEEEEEEEEKKKKKKKKKK", "EKEKEKEKEKEKEKEKEKEK"], ["E"], ["K"], 5, flatten=True).round(4).tolist()
    [1.0, 0.0078]
    """
    _check_kappa_arguments(group1, window_size)
    batch = encoding.encode_batch(sequences)

    result = np.full(len(batch), KAPPA_UNDEFINED, dtype=np.float64)
    for indices, codes in batch.by_length():
        if codes.shape[1] < window_size:
            continue
        ternary = encoding.ternarize_winged_batch(codes, group1, group2, window_size)
        result[indices] = kappa_from_ternary(ternary, window_size, flatten, group2_present=len(group2) > 0)
    return result


def kappa(sequences: str | Sequence[str] | encoding.EncodedBatch) -> NDArray[np.float64]:
    """The charge patterning parameter kappa of Das & Pappu (2013).

    The mean of :func:`kappa_x` for the positive (R, K) against the negative
    (E, D) residues at window sizes 5 and 6, with values above 1 capped at 1.

    Parameters
    ----------
    sequences : str, sequence of str, or EncodedBatch
        The batch.

    Returns
    -------
    numpy.ndarray
        ``(n_sequences,)`` kappa values between 0 and 1; -1 for a sequence
        shorter than 6 residues or lacking either charge.
    """
    batch = encoding.encode_batch(sequences)
    per_window = np.stack(
        [kappa_x(batch, amino_acids.POS, amino_acids.NEG, size, flatten=True) for size in KAPPA_WINDOW_SIZES]
    )
    mean = per_window.mean(axis=0)
    result: NDArray[np.float64] = np.where(batch.lengths < max(KAPPA_WINDOW_SIZES), KAPPA_UNDEFINED, mean)
    return result


# =====================================================================
# sigma and delta
# =====================================================================


def sigma(sequences: Batch, group1: Sequence[str], group2: Sequence[str]) -> NDArray[np.float64]:
    """Whole-sequence compositional asymmetry between two groups of residues.

    ``(f_1 - f_2)^2 / (f_1 + f_2)``: near 1 when one group dominates, 0 when they
    are balanced. A residue in both groups counts towards both.

    Parameters
    ----------
    sequences : str, sequence of str, EncodedBatch or ResidueCounts
        The batch.
    group1, group2 : sequence of str
        The two residue groups.

    Returns
    -------
    numpy.ndarray
        ``(n_sequences,)`` asymmetries; 0.0 when neither group is present.

    Examples
    --------
    >>> sigma(["EEEEKKKK", "EEEEEEEK"], ["E"], ["K"]).round(4).tolist()
    [0.0, 0.5625]
    """
    counts = as_counts(sequences)
    count_1 = counts.counts[:, encoding.residue_membership(group1)].sum(axis=1)
    count_2 = counts.counts[:, encoding.residue_membership(group2)].sum(axis=1)
    return asymmetry(count_1 - count_2, count_1 + count_2, counts.lengths.astype(np.float64))


def delta(
    sequences: str | Sequence[str] | encoding.EncodedBatch,
    group1: Sequence[str],
    group2: Sequence[str],
    window_size: int,
) -> NDArray[np.float64]:
    """Das & Pappu's delta between two groups of residues.

    The mean squared difference between each window's asymmetry and the whole
    sequence's (:func:`sigma`), over the windows of the sequence itself - no
    wings. This is the quantity NARDINI z-scores for a pair of residue types.

    Parameters
    ----------
    sequences : str, sequence of str, or EncodedBatch
        The batch.
    group1, group2 : sequence of str
        The two residue groups.
    window_size : int
        Window width.

    Returns
    -------
    numpy.ndarray
        ``(n_sequences,)`` delta values; NaN for a sequence shorter than the
        window, which has no windows to average.
    """
    batch = encoding.encode_batch(sequences)
    member_1 = encoding.residue_membership(group1)
    member_2 = encoding.residue_membership(group2)

    result = np.full(len(batch), np.nan, dtype=np.float64)
    for indices, codes in batch.by_length():
        length = codes.shape[1]
        if length < window_size:
            continue
        in_1 = member_1[codes]
        in_2 = member_2[codes]
        count_1 = in_1.sum(axis=1)
        count_2 = in_2.sum(axis=1)
        whole = asymmetry(count_1 - count_2, count_1 + count_2, float(length))

        window_1 = window_sums(in_1, window_size)
        window_2 = window_sums(in_2, window_size)
        result[indices] = window_delta(window_1 - window_2, window_1 + window_2, window_size, whole)
    return result


# =====================================================================
# patterning percentile
# =====================================================================


def binary_asymmetry(sequences: Batch, group: Sequence[str]) -> NDArray[np.float64]:
    """Whole-sequence asymmetry of one group against everything else.

    ``(f - (1 - f))^2``, where ``f`` is the fraction of the sequence in
    ``group``: 0 when exactly half the sequence is in the group, 1 when all or
    none of it is.

    Parameters
    ----------
    sequences : str, sequence of str, EncodedBatch or ResidueCounts
        The batch.
    group : sequence of str
        The residue group.

    Returns
    -------
    numpy.ndarray
        ``(n_sequences,)`` asymmetries between 0 and 1; 0.0 for an empty sequence.
    """
    counts = as_counts(sequences)
    inside = counts.counts[:, encoding.residue_membership(group)].sum(axis=1)
    lengths = counts.lengths
    return asymmetry(2 * inside - lengths, lengths, lengths.astype(np.float64))


def _percentile_of(
    codes: NDArray[np.int8], member: NDArray[np.bool_], window_size: int, count: int, seed: int
) -> tuple[float, float, NDArray[np.float64]]:
    """Patterning percentile of one sequence, with its observed delta and null."""
    # +1 / -1 with no wings: every window then holds the same number of the two
    # classes, and the asymmetry reduces exactly to (f - (1 - f))^2
    binary = np.where(member[codes], 1, -1).astype(np.int8)
    length = binary.size
    n_positive = int(np.count_nonzero(binary == 1))
    whole = asymmetry(2 * n_positive - length, length, float(length))

    observed = float(ternary_delta(binary, window_size, whole))

    rng = np.random.default_rng(seed)
    null = np.empty(count, dtype=np.float64)
    done = 0
    while done < count:
        block = min(_SHUFFLE_BLOCK, count - done)
        order = rng.random((block, length)).argsort(axis=1)
        null[done : done + block] = ternary_delta(binary[order], window_size, whole)
        done += block
    null.sort()

    # the fraction of shuffles no blockier than the real sequence
    return float(np.count_nonzero(null <= observed) / count), observed, null


def patterning_percentile(
    sequences: str | Sequence[str] | encoding.EncodedBatch,
    group: Sequence[str],
    window_size: int,
    count: int,
    seed: int,
    return_distribution: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Where each sequence's patterning of one group sits among random shuffles.

    The sequence is reduced to ``group`` against everything else, its delta over
    windows of ``window_size`` is computed, and the percentile is the fraction of
    ``count`` random shuffles of the same sequence whose delta is no larger.
    **0** means far more evenly patterned than chance, **1** far blockier.

    Every sequence is shuffled with a generator seeded from ``seed`` alone, so a
    sequence gets the same shuffles - and the same percentile - however it is
    batched.

    Parameters
    ----------
    sequences : str, sequence of str, or EncodedBatch
        The batch.
    group : sequence of str
        Residues forming the positive class. Must be non-empty.
    window_size : int
        Window width. Must be at least 2 and no larger than any sequence.
    count : int
        Number of shuffles in each null distribution.
    seed : int
        Seed for the shuffle generator.
    return_distribution : bool, optional
        If True, also return each sequence's observed delta and its sorted null
        distribution. Default False.

    Returns
    -------
    numpy.ndarray or tuple
        ``(n_sequences,)`` percentiles; with ``return_distribution``,
        ``(percentiles, observed, null)`` where ``null`` has shape
        ``(n_sequences, count)``.

    Raises
    ------
    sparrow.sparrow_exceptions.PatterningException
        If the window is smaller than 2 or larger than a sequence, ``group`` is
        empty, or ``seed`` is not an integer.

    Examples
    --------
    >>> blocky, mixed = patterning_percentile(["Q" * 20 + "A" * 20, "QA" * 20], ["Q"], 5, 500, 1)
    >>> bool(blocky > mixed)
    True
    """
    if window_size < 2:
        raise PatterningException("window_size must be 2 or larger")
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        raise PatterningException("Cannot convert seed into an integer")
    if len(group) == 0:
        raise PatterningException("Passed group is empty, group must contain one or more possible residues")

    batch = encoding.encode_batch(sequences)
    if np.any(batch.lengths < window_size):
        raise PatterningException("window_size is larger than sequence length")

    member = encoding.residue_membership(group)
    percentiles = np.empty(len(batch), dtype=np.float64)
    observed = np.empty(len(batch), dtype=np.float64)
    null = np.empty((len(batch), count), dtype=np.float64)
    for index in range(len(batch)):
        codes = batch.codes[batch.offsets[index] : batch.offsets[index + 1]]
        percentiles[index], observed[index], null[index] = _percentile_of(codes, member, window_size, count, seed)

    if return_distribution:
        return percentiles, observed, null
    return percentiles


# =====================================================================
# inverse-weighted distance
# =====================================================================


def _inverse_distance_kernel(length: int) -> NDArray[np.float64]:
    """``[0, 1, 1/2, 1/3, ...]``, up to ``1 / (length - 1)``."""
    kernel = np.zeros(length, dtype=np.float64)
    if length > 1:
        kernel[1:] = 1.0 / np.arange(1, length, dtype=np.float64)
    return kernel


def _autocorrelation(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Autocorrelation of every row at every lag, ``A[d] = sum_i v[i] v[i + d]``, by FFT.

    Rounded to the nearest integer, which is exact for the integer-valued inputs
    used here (0/1 masks and -1/0/+1 charges) and removes the FFT's ~1e-17 noise.
    """
    length = values.shape[1]
    size = 1
    while size < 2 * length:
        size *= 2
    spectrum = np.fft.rfft(values, n=size, axis=1)
    autocorrelation: NDArray[np.float64] = np.rint(
        np.fft.irfft(spectrum * np.conjugate(spectrum), n=size, axis=1)[:, :length]
    )
    return autocorrelation


def iwd_from_masks(masks: NDArray[np.generic]) -> NDArray[np.float64]:
    """Inverse-weighted distance for a stack of 0/1 occupancy masks of one length.

    For every marked position, sum ``1 / distance`` to every other marked
    position, and average over the marked positions. Grouping the pairs by their
    separation ``d`` turns the double sum into ``sum_d A(d) / d``, where ``A`` is
    the mask's autocorrelation, so a whole stack is one batched FFT rather than
    a pairwise loop per row.

    Parameters
    ----------
    masks : array_like
        Array of shape ``(n_masks, length)`` of 0/1 (or boolean) values.

    Returns
    -------
    numpy.ndarray
        IWD per mask; 0.0 for a mask with fewer than two marked positions.

    Examples
    --------
    >>> import numpy as np
    >>> values = iwd_from_masks(np.array([[1, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0]]))
    >>> bool(values[0] > values[1])
    True
    """
    # ascontiguousarray, not just astype: pocketfft takes a different path for a
    # non-C-contiguous input, and the results then differ in the last bit between
    # rows of the same call - which turns a null distribution that should be an
    # exact point mass into one with ~1e-16 of spread
    stack = np.ascontiguousarray(np.atleast_2d(np.asarray(masks)), dtype=np.float64)
    counts = stack.sum(axis=1)

    # Each unordered pair is counted once by the autocorrelation and contributes
    # to two positions' sums, hence the factor of 2. Each row is summed on its
    # own rather than as a matrix product: BLAS orders a matrix product's
    # additions by the number of rows, which would give a sequence different last
    # bits alone than in a stack (or than its own scrambles).
    pair_sum = 2.0 * (_autocorrelation(stack) * _inverse_distance_kernel(stack.shape[1])).sum(axis=1)

    out = np.zeros(stack.shape[0], dtype=np.float64)
    np.divide(pair_sum, counts, out=out, where=counts > 0)
    return out


def iwd(
    sequences: str | Sequence[str] | encoding.EncodedBatch,
    target_residues: Sequence[str],
) -> NDArray[np.float64]:
    """Inverse-weighted distance (IWD) between the target residues of each sequence.

    A clustering metric: high when the target residues sit close together, low
    when they are spread out.

    Parameters
    ----------
    sequences : str, sequence of str, or EncodedBatch
        The batch.
    target_residues : sequence of str
        Residues that count as occupied positions.

    Returns
    -------
    numpy.ndarray
        ``(n_sequences,)`` IWD values; 0.0 with fewer than two target residues.

    Examples
    --------
    >>> clustered, spread = iwd(["AAAAAQQQQQ", "AQAQAQAQAQ"], ["Q"])
    >>> bool(clustered > spread)
    True
    """
    batch = encoding.encode_batch(sequences)
    member = encoding.residue_membership(target_residues)

    result = np.zeros(len(batch), dtype=np.float64)
    for indices, codes in batch.by_length():
        result[indices] = iwd_from_masks(member[codes])
    return result


#: Window over which the local net charge per residue is taken for the
#: charge-weighted IWD.
CHARGE_IWD_WINDOW = 8

# which residue codes are positive and which negative
_IS_POSITIVE = encoding.residue_membership(amino_acids.POS)
_IS_NEGATIVE = encoding.residue_membership(amino_acids.NEG)


def _pairwise_inverse_distance_sums(positions: NDArray[np.intp]) -> NDArray[np.float64]:
    """For each position, the sum of ``1 / distance`` to every other position in the set."""
    separation = np.abs(positions[:, None] - positions[None, :]).astype(np.float64)
    with np.errstate(divide="ignore"):
        inverse = np.where(separation > 0, 1.0 / separation, 0.0)
    sums: NDArray[np.float64] = inverse.sum(axis=1)
    return sums


def _charged_iwd_of(codes: NDArray[np.int8], ncpr: NDArray[np.float64], charge: str) -> float:
    """Charge-weighted IWD of one sequence's positive or negative residues.

    ``ncpr`` is the sequence's local net charge per residue at every position.
    """
    if charge == "-":
        selected = _IS_NEGATIVE[codes]
        wanted_sign = -1.0
    else:
        selected = _IS_POSITIVE[codes]
        wanted_sign = 1.0

    positions = np.flatnonzero(selected)
    if positions.size == 0:
        return 0.0

    per_hit = _pairwise_inverse_distance_sums(positions)
    local_charge = ncpr[positions]
    # only residues sitting in a correctly signed local environment contribute
    weights = np.where(np.sign(local_charge) == wanted_sign, np.abs(local_charge), 0.0)

    # the denominator is the residue count, not the weight sum
    return float(np.dot(per_hit, weights) / positions.size)


def iwd_charged_weighted(
    sequences: str | Sequence[str] | encoding.EncodedBatch,
    charge: str,
    window_size: int = CHARGE_IWD_WINDOW,
) -> NDArray[np.float64]:
    """Charge-weighted IWD of the positive or the negative residues of each sequence.

    Each residue's contribution is weighted by the magnitude of the local net
    charge per residue (over ``window_size`` residues) at its position, and a
    residue whose local NCPR has the opposite sign contributes nothing.

    Parameters
    ----------
    sequences : str, sequence of str, or EncodedBatch
        The batch.
    charge : {'-', '+'}
        ``'-'`` for the clustering of D and E, ``'+'`` for R and K.
    window_size : int, optional
        Window for the local NCPR, which is
        :func:`~sparrow.properties.profiles.net_charge_profile` with its ends
        extended. Default 8.

    Returns
    -------
    numpy.ndarray
        ``(n_sequences,)`` non-negative values.

    Raises
    ------
    sparrow.sparrow_exceptions.PatterningException
        If ``charge`` is not ``'-'`` or ``'+'``.
    sparrow.sparrow_exceptions.SparrowException
        If a sequence is shorter than ``window_size``.
    """
    if charge not in ("-", "+"):
        raise PatterningException(f"charge must be '-' or '+', not {charge!r}")
    batch = encoding.encode_batch(sequences)
    local_ncpr = profiles.net_charge_profile(batch, window_size, end_mode=profiles.EXTEND_ENDS)
    return np.array(
        [
            _charged_iwd_of(batch.codes[batch.offsets[i] : batch.offsets[i + 1]], local_ncpr[i], charge)
            for i in range(len(batch))
        ],
        dtype=np.float64,
    )


def _bivariate_charged_iwd_of(codes: NDArray[np.int8], ncpr: NDArray[np.float64]) -> float:
    """Charge-weighted IWD between one sequence's positive and negative residues.

    ``ncpr`` is the sequence's local net charge per residue at every position.
    """
    is_negative = _IS_NEGATIVE[codes]
    is_positive = _IS_POSITIVE[codes]
    negative_positions = np.flatnonzero(is_negative)
    positive_positions = np.flatnonzero(is_positive)
    if negative_positions.size == 0 or positive_positions.size == 0:
        return 0.0

    # a residue only carries its charge if its local environment agrees in sign
    masked = np.zeros_like(ncpr)
    negative_here = is_negative & (ncpr < 0)
    positive_here = is_positive & (ncpr > 0)
    masked[negative_here] = ncpr[negative_here]
    masked[positive_here] = ncpr[positive_here]

    charge_difference = np.abs(masked[positive_positions][None, :] - masked[negative_positions][:, None])
    separation = np.abs(positive_positions[None, :] - negative_positions[:, None]).astype(np.float64)
    return float(np.sum(charge_difference / separation) / negative_positions.size)


def bivariate_iwd_charged_weighted(
    sequences: str | Sequence[str] | encoding.EncodedBatch,
    window_size: int = CHARGE_IWD_WINDOW,
) -> NDArray[np.float64]:
    """Charge-weighted IWD between the positive and negative residues of each sequence.

    How intermixed the oppositely charged residues are: each positive/negative
    pair is weighted by the difference in their local net charge per residue
    and divided by their separation.

    Parameters
    ----------
    sequences : str, sequence of str, or EncodedBatch
        The batch.
    window_size : int, optional
        Window for the local NCPR, which is
        :func:`~sparrow.properties.profiles.net_charge_profile` with its ends
        extended. Default 8.

    Returns
    -------
    numpy.ndarray
        ``(n_sequences,)`` non-negative values; 0.0 for a sequence lacking either
        charge.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If a sequence is shorter than ``window_size``.
    """
    batch = encoding.encode_batch(sequences)
    local_ncpr = profiles.net_charge_profile(batch, window_size, end_mode=profiles.EXTEND_ENDS)
    return np.array(
        [
            _bivariate_charged_iwd_of(batch.codes[batch.offsets[i] : batch.offsets[i + 1]], local_ncpr[i])
            for i in range(len(batch))
        ],
        dtype=np.float64,
    )


# =====================================================================
# sequence charge and hydropathy decoration
# =====================================================================


def scd(
    sequences: str | Sequence[str] | encoding.EncodedBatch,
    group1: Sequence[str] = tuple(amino_acids.NEG),
    group2: Sequence[str] = tuple(amino_acids.POS),
) -> NDArray[np.float64]:
    """Sequence charge decoration (SCD) of Sawle & Ghosh, for any two groups.

    ``SCD = (1/N) sum_{m > n} q_m q_n sqrt(m - n)``, with ``group1`` residues
    charged -1 and ``group2`` residues +1 (a residue in both counts as +1).
    Grouping the pairs by separation turns the double sum into the charge
    sequence's autocorrelation, so it costs one FFT - ``O(N log N)`` - rather than
    ``O(N^2)``.

    Parameters
    ----------
    sequences : str, sequence of str, or EncodedBatch
        The batch.
    group1 : sequence of str, optional
        Residues charged -1. Default E and D.
    group2 : sequence of str, optional
        Residues charged +1. Default R and K.

    Returns
    -------
    numpy.ndarray
        ``(n_sequences,)`` SCD values; 0.0 for a sequence shorter than 2.

    References
    ----------
    Sawle L, Ghosh K. A theoretical method to compute sequence dependent
    configurational properties in charged polymers and proteins. J Chem Phys.
    2015;143(8):085101.

    Examples
    --------
    >>> blocky, mixed = scd(["EEEEEEEEEEKKKKKKKKKK", "EKEKEKEKEKEKEKEKEKEK"])
    >>> bool(blocky < mixed)
    True
    """
    batch = encoding.encode_batch(sequences)
    charge = np.zeros(encoding.NUM_RESIDUES, dtype=np.float64)
    charge[encoding.residue_membership(group1)] = -1.0
    charge[encoding.residue_membership(group2)] = 1.0

    result = np.zeros(len(batch), dtype=np.float64)
    for indices, codes in batch.by_length():
        length = codes.shape[1]
        if length < 2:
            continue
        separations = np.sqrt(np.arange(length, dtype=np.float64))
        autocorrelation = _autocorrelation(charge[codes])
        # summed row by row, not as a matrix product - see iwd_from_masks
        result[indices] = (autocorrelation[:, 1:] * separations[1:]).sum(axis=1) / length
    return result


def _hydropathy_table(scale: Mapping[str, float], batch: Sequence[str]) -> NDArray[np.float64]:
    """A 256-entry table of ``scale`` by ASCII code, checked against the batch.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If a character in the batch has no value in the scale.
    """
    missing = sorted(set("".join(batch)) - set(scale))
    if missing:
        raise SparrowException(f"When calculating SHD the hydropathy scale has no value for {missing}")
    table = np.zeros(encoding.ASCII_TABLE_SIZE, dtype=np.float64)
    for character, value in scale.items():
        table[ord(character)] = float(value)
    return table


def shd(
    sequences: str | Sequence[str],
    hydropathy: Mapping[str, float] | None = None,
) -> NDArray[np.float64]:
    """Sequence hydropathy decoration (SHD) of Zheng et al.

    ``SHD = (1/N) sum_{m > n} (h_m + h_n) / (m - n)``. Because the numerator is a
    sum rather than a product, grouping by separation ``d`` leaves only prefix
    sums: ``O(N)`` rather than ``O(N^2)``.

    The scale is looked up character by character, so any alphabet it covers
    works - the SHD paper's own checks use binary ``'0'``/``'1'`` sequences.

    Parameters
    ----------
    sequences : str or sequence of str
        The batch.
    hydropathy : mapping of str to float, optional
        Value of each character. Default the HPS model's hydropathy
        (:data:`sparrow.data.amino_acids.AA_hydro_HPS`), which SHD was defined
        with.

    Returns
    -------
    numpy.ndarray
        ``(n_sequences,)`` SHD values; 0.0 for a sequence shorter than 2.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If a sequence holds a character the scale has no value for. The message
        lists every such character.

    References
    ----------
    Zheng W, Dignon GL, Brown M, Kim YC, Mittal J. Hydropathy patterning
    complements charge patterning to describe conformational preferences of
    disordered proteins. J Phys Chem Lett. 2020;11(9):3408-3415.
    """
    batch = encoding.as_batch(sequences)
    table = _hydropathy_table(amino_acids.AA_hydro_HPS if hydropathy is None else hydropathy, batch)

    result = np.zeros(len(batch), dtype=np.float64)
    by_length: dict[int, list[int]] = {}
    for index, sequence in enumerate(batch):
        by_length.setdefault(len(sequence), []).append(index)

    for length, indices in by_length.items():
        if length < 2:
            continue
        values = table[np.stack([encoding.ascii_codes(batch[i]) for i in indices])]

        # sum_d (1/d) [ sum_{m >= d} h_m + sum_{n <= N-1-d} h_n ], both prefix sums
        prefix = np.concatenate([np.zeros((len(indices), 1)), np.cumsum(values, axis=1)], axis=1)
        total = prefix[:, -1:]
        separations = np.arange(1, length, dtype=np.float64)
        upper = total - prefix[:, 1:length]
        lower = prefix[:, length - np.arange(1, length)]
        result[indices] = np.sum((upper + lower) / separations, axis=1) / length
    return result
