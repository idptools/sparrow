"""Counting the k-mers of a background, for the null models that learn from one.

A k-mer is a run of ``k`` consecutive residues, and its index is the run read
as a number in base 20, first residue most significant, with each residue its
code in :data:`sparrow.properties.encoding.RESIDUES`: in ``"AC"``, A is 0 and C
is 1, so its index is ``0 * 20 + 1 = 1``. The counts of every k-mer of one
length are then a single array, one entry for each of the ``20**k`` k-mers.

Every overlapping run in every background sequence is counted, apart from any
that spans a residue other than the 20 standard amino acids, or the join
between two sequences. Lengths of up to :data:`MAX_KMER_SIZE` are counted.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from sparrow.properties import encoding
from sparrow.sparrow_exceptions import NullModelWarning, SparrowException

__all__ = ["MAX_KMER_SIZE", "MIN_MEAN_KMER_COUNT", "count_background_kmers", "kmer_residues", "weighted_choice"]

#: The longest k-mer a background is counted in. There are 20**4 = 160,000
#: 4-mers, so even a background of a million residues sees each about 6 times
#: on average, and a whole human proteome (about 11 million residues) about 70
#: times. There are 3.2 million 5-mers, which the human proteome would see
#: about 3 times each - too few for their frequencies to mean much.
MAX_KMER_SIZE = 4

#: The average count of each possible k-mer below which fitting warns that the
#: background is too small. Counts are roughly Poisson, so at an average of 10
#: a typical k-mer's frequency is already uncertain by about a third
#: (``1 / sqrt(10)``).
MIN_MEAN_KMER_COUNT = 10

# what goes between two sequences when they are joined into one array, so that
# no k-mer is counted across the join: a byte no sequence has
_SEPARATOR = "\x00"

# ASCII code -> residue code, NOT_A_RESIDUE for everything else
_RESIDUE_CODE_TABLE = np.full(encoding.ASCII_TABLE_SIZE, encoding.NOT_A_RESIDUE, dtype=np.int8)
for _residue, _code in encoding.RESIDUE_INDEX.items():
    _RESIDUE_CODE_TABLE[ord(_residue)] = _code


def count_background_kmers(sequences: list[str], largest: int, model_name: str) -> dict[int, NDArray[np.int64]]:
    """The count of every k-mer in a background, for each length up to ``largest``.

    Warns how many residues were not counted, if any were not one of the 20
    standard amino acids, and if the background has too few of the longest
    k-mers for their frequencies to mean much (see
    :data:`MIN_MEAN_KMER_COUNT`).

    Parameters
    ----------
    sequences : list of str
        The background, upper case.
    largest : int
        The longest k-mer to count, at most :data:`MAX_KMER_SIZE`.
    model_name : str
        The null model being fit, for the messages.

    Returns
    -------
    dict
        For each length from 1 to ``largest``, an array of ``20**length``
        counts indexed by k-mer index.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the background has no run of ``largest`` standard residues.
    """
    codes = _joined_codes(sequences)
    counts = {size: _count_kmers(codes, size) for size in range(1, largest + 1)}

    total = int(counts[largest].sum())
    if total == 0:
        raise SparrowException(
            f"The background has no run of {largest} standard residues, so {model_name} has nothing to learn from."
        )

    possible = encoding.NUM_RESIDUES**largest
    mean_count = total / possible
    if mean_count < MIN_MEAN_KMER_COUNT:
        warnings.warn(
            f"The background has {total} {largest}-mers, an average of {mean_count:.2g} for each of the "
            f"{possible} possible ones (at least {MIN_MEAN_KMER_COUNT} is recommended), so most of their "
            f"frequencies rest on a handful of counts. Fit {model_name} to a larger background, or use shorter k-mers.",
            NullModelWarning,
            stacklevel=4,
        )
    return counts


def kmer_residues(indices: NDArray[np.integer], size: int) -> NDArray[np.int64]:
    """The residue codes of each k-mer, from its index.

    Parameters
    ----------
    indices : numpy.ndarray
        ``(n,)`` k-mer indices.
    size : int
        The k-mer length.

    Returns
    -------
    numpy.ndarray
        ``(n, size)`` residue codes, in order along each k-mer.
    """
    residues = np.empty((len(indices), size), dtype=np.int64)
    remainder = np.asarray(indices, dtype=np.int64)
    for position in range(size - 1, -1, -1):
        remainder, residues[:, position] = np.divmod(remainder, encoding.NUM_RESIDUES)
    return residues


def weighted_choice(weights: NDArray[np.floating] | NDArray[np.integer], uniform: float) -> int:
    """An index, picked with a probability proportional to its weight.

    Parameters
    ----------
    weights : numpy.ndarray
        Non-negative weights, at least one of them positive.
    uniform : float
        A random number in [0, 1), which decides the pick.

    Returns
    -------
    int
        The picked index, always one with a positive weight.
    """
    cumulative = np.cumsum(weights, dtype=np.float64)
    index = int(np.searchsorted(cumulative, uniform * cumulative[-1], side="right"))
    # rounding can carry the target to the very end; the last index with any
    # weight is then the one picked
    return min(index, int(np.flatnonzero(weights)[-1]))


def _joined_codes(sequences: list[str]) -> NDArray[np.int8]:
    """Every sequence as residue codes, one after another, with a gap between each.

    The gaps, and any residue other than the 20 standard amino acids, are
    :data:`~sparrow.properties.encoding.NOT_A_RESIDUE`, so no k-mer is counted
    across them. Warns how many residues were not standard, if any.

    Parameters
    ----------
    sequences : list of str
        The background, upper case.

    Returns
    -------
    numpy.ndarray
        ``int8`` residue codes.
    """
    joined = _SEPARATOR.join(sequences)
    # errors="replace" turns a non-ASCII character into "?", which is not a residue
    ascii_codes = np.frombuffer(joined.encode("ascii", errors="replace"), dtype=np.uint8)
    codes = _RESIDUE_CODE_TABLE[ascii_codes]

    not_residues = ascii_codes[(codes == encoding.NOT_A_RESIDUE) & (ascii_codes != ord(_SEPARATOR))]
    if not_residues.size > 0:
        characters, character_counts = np.unique(not_residues, return_counts=True)
        tally = ", ".join(f"{chr(code)}: {number}" for code, number in zip(characters, character_counts))
        warnings.warn(
            f"{not_residues.size} residues in the background are not one of the 20 standard amino acids "
            f"({tally}); they, and every k-mer that spans one, were not counted.",
            NullModelWarning,
            stacklevel=5,
        )
    return codes


def _count_kmers(codes: NDArray[np.int8], size: int) -> NDArray[np.int64]:
    """The count of every k-mer of one length.

    Parameters
    ----------
    codes : numpy.ndarray
        Residue codes, from :func:`_joined_codes`.
    size : int
        The k-mer length.

    Returns
    -------
    numpy.ndarray
        ``20**size`` counts, indexed by k-mer index.
    """
    possible = encoding.NUM_RESIDUES**size
    windows = len(codes) - size + 1
    if windows < 1:
        return np.zeros(possible, dtype=np.int64)

    is_residue = codes != encoding.NOT_A_RESIDUE
    residue_codes = np.where(is_residue, codes, 0).astype(np.int64)

    # each window's index, built up one residue at a time, and whether every
    # residue in it is a standard one
    indices = np.zeros(windows, dtype=np.int64)
    counted = np.ones(windows, dtype=bool)
    for position in range(size):
        indices = indices * encoding.NUM_RESIDUES + residue_codes[position : position + windows]
        counted &= is_residue[position : position + windows]

    return np.bincount(indices[counted], minlength=possible).astype(np.int64)
