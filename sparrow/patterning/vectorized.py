"""Inverse-weighted distance, SCD and SHD, in NumPy.

These are the last patterning metrics still to move into
:mod:`sparrow.properties.patterning`, where kappa, delta and the patterning
percentile now live. They were compiled Cython extensions until they were
reimplemented here in vectorized NumPy.

Several of them are algorithmically better than the compiled versions rather
than merely vectorized: IWD's pairwise sum is computed as an autocorrelation by
FFT, SCD likewise (``O(n log n)`` instead of ``O(n^2)``), and SHD collapses to
prefix sums (``O(n)`` instead of ``O(n^2)``). The IWD routines are double
precision on both sides and agree with the compiled ones to ~1e-15.
"""

import numpy as np

from sparrow.data import amino_acids
from sparrow.properties.encoding import ascii_codes, membership_lookup

__all__ = [
    "calculate_average_inverse_distance_from_sequence",
    "calculate_average_inverse_distance_charge",
    "calculate_average_bivariate_inverse_distance_charge",
    "iwd_from_binary_mask",
    "iwd_batch",
    "compute_scd_x",
    "compute_shd",
    "DEFAULT_HYDRO_DICT",
]


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
    codes = ascii_codes(sequence)
    if codes.size == 0:
        return 0.0

    mask = membership_lookup(target_residues)[codes]
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
    codes = ascii_codes(sequence)
    ncpr = np.asarray(linear_NCPR, dtype=np.float64)

    if charge == "-":
        selected = membership_lookup("DE")[codes]
        wanted_sign = -1.0
    else:
        selected = membership_lookup("RK")[codes]
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
    codes = ascii_codes(sequence)
    ncpr = np.asarray(linear_NCPR, dtype=np.float64)

    negative_positions = np.flatnonzero(membership_lookup("DE")[codes])
    positive_positions = np.flatnonzero(membership_lookup("RK")[codes])

    if negative_positions.size == 0 or positive_positions.size == 0:
        return 0.0

    # a residue only carries its charge if the local environment agrees in sign
    masked = np.zeros_like(ncpr)
    is_negative = membership_lookup("DE")[codes] & (ncpr < 0)
    is_positive = membership_lookup("RK")[codes] & (ncpr > 0)
    masked[is_negative] = ncpr[is_negative]
    masked[is_positive] = ncpr[is_positive]

    charge_difference = np.abs(masked[positive_positions][None, :] - masked[negative_positions][:, None])
    separation = np.abs(positive_positions[None, :] - negative_positions[:, None]).astype(np.float64)

    return float(np.sum(charge_difference / separation) / negative_positions.size)


# =====================================================================
# sequence charge / hydropathy decoration
# =====================================================================

#: The HPS model's residue hydropathy, used by :func:`compute_shd` when no scale
#: is supplied. See :data:`sparrow.data.amino_acids.AA_hydro_HPS`.
DEFAULT_HYDRO_DICT = amino_acids.AA_hydro_HPS


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
    codes = ascii_codes(sequence)
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
        :data:`DEFAULT_HYDRO_DICT`, the HPS model's hydropathy scale.

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
