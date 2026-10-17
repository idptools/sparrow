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

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from sparrow.data import amino_acids
from sparrow.properties import encoding
from sparrow.properties.composition import Batch, as_counts
from sparrow.properties.windows import window_sums
from sparrow.sparrow_exceptions import PatterningException

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
        kappa per row, or :data:`KAPPA_UNDEFINED` (-1) where a group is absent.

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
