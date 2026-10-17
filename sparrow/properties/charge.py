"""pH-dependent charge and the isoelectric point.

Both use Henderson-Hasselbalch with the side-chain pKa values in
:data:`sparrow.data.amino_acids.AA_PKA` - the EMBOSS values, except lysine at
10.0 rather than 10.8, as in localCIDER - treating K, R and H as bases and E, D,
Y and C as acids. The termini are not included. This reproduces localCIDER's
``get_isoelectric_point()``, including its convention of normalizing the charge
by the number of titratable residues.

Like :mod:`sparrow.properties.composition`, both are functions of composition
only, so they take sequences or precomputed
:class:`~sparrow.properties.composition.ResidueCounts`, and a whole batch is
computed at once - the isoelectric point by a bisection that runs on every
sequence in step, each one stopping as soon as it has converged.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sparrow.data import amino_acids
from sparrow.properties import encoding
from sparrow.properties.composition import Batch, ResidueCounts, as_counts

__all__ = [
    "DEFAULT_PH",
    "PI_THRESHOLD",
    "charge_at_pH",
    "isoelectric_point",
]

#: pH used by :func:`charge_at_pH` unless another is given.
DEFAULT_PH = 7.4

#: The isoelectric point is the pH at which the mean titratable charge is within
#: this of zero. Matches localCIDER.
PI_THRESHOLD = 0.02

#: The pH range the bisection starts from.
PI_SEARCH_RANGE = (0.0, 14.0)

#: Bisection steps before the range is widened by one pH unit. A very highly
#: charged sequence can have its pI outside the starting range.
PI_STEPS_PER_WIDENING = 20

#: How many times the range may be widened before the bisection stops and returns
#: the midpoint of the range it has reached.
PI_MAX_WIDENINGS = 10


# The titratable residues in a fixed order - bases, then acids - with their
# residue codes, pKa values and the sign of the charge each carries when ionized.
_TITRATABLE = tuple(amino_acids.TITRATABLE_BASES) + tuple(amino_acids.TITRATABLE_ACIDS)
_TITRATABLE_COLUMNS = np.array([encoding.RESIDUE_INDEX[residue] for residue in _TITRATABLE])
_TITRATABLE_PKA = np.array([amino_acids.AA_PKA[residue] for residue in _TITRATABLE])
_TITRATABLE_SIGN = np.array([1.0] * len(amino_acids.TITRATABLE_BASES) + [-1.0] * len(amino_acids.TITRATABLE_ACIDS))


def _titratable_counts(counts: ResidueCounts) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Counts of each titratable residue, ``(n, 7)``, and their total, ``(n,)``."""
    # row-major, so each row's charges are summed in the same order alone or in
    # a batch (indexing columns can return them laid out column by column)
    titratable = np.ascontiguousarray(counts.counts[:, _TITRATABLE_COLUMNS], dtype=np.float64)
    return titratable, titratable.sum(axis=1)


def _summed_charge(titratable: NDArray[np.float64], pH: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """Summed side-chain charge of each sequence, by Henderson-Hasselbalch.

    A base contributes ``1 / (1 + 10^(pH - pKa))`` per residue and an acid
    ``-1 / (1 + 10^(pKa - pH))``; with ``sign`` +1 for a base and -1 for an acid,
    both are ``sign / (1 + 10^(sign * (pH - pKa)))``. ``pH`` may be one value for
    the whole batch or one per sequence.
    """
    shift = np.asarray(pH, dtype=np.float64)[..., None] - _TITRATABLE_PKA
    per_residue = _TITRATABLE_SIGN / (1.0 + 10.0 ** (_TITRATABLE_SIGN * shift))
    total: NDArray[np.float64] = (titratable * per_residue).sum(axis=-1)
    return total


def _mean_charge(
    titratable: NDArray[np.float64], n_titratable: NDArray[np.float64], pH: float | NDArray[np.float64]
) -> NDArray[np.float64]:
    """Charge per titratable residue; 0.0 for a sequence with none."""
    result: NDArray[np.float64] = _summed_charge(titratable, pH) / np.maximum(n_titratable, 1.0)
    return result


def charge_at_pH(sequences: Batch, pH: float = DEFAULT_PH, normalize: bool = True) -> NDArray[np.float64]:
    """Charge of the titratable side chains of each sequence at a given pH.

    Parameters
    ----------
    sequences : str, sequence of str, or ResidueCounts
        The batch.
    pH : float, optional
        pH at which to evaluate the charge. Default 7.4.
    normalize : bool, optional
        If True (default), divide by the number of titratable residues, giving
        the mean charge per titratable group. If False, return the summed charge.

    Returns
    -------
    numpy.ndarray
        ``(n_sequences,)`` charges. A sequence with no titratable residues has
        charge 0.0.

    Examples
    --------
    >>> charge_at_pH(["KKKKEEEE", "AAAA"], pH=0.0).round(3).tolist()
    [0.5, 0.0]
    """
    titratable, n_titratable = _titratable_counts(as_counts(sequences))
    if normalize:
        return _mean_charge(titratable, n_titratable, pH)
    return _summed_charge(titratable, pH)


def isoelectric_point(sequences: Batch, threshold: float = PI_THRESHOLD) -> NDArray[np.float64]:
    """Estimated isoelectric point (pI) of each sequence.

    Found by bisecting on pH until the mean titratable charge
    (:func:`charge_at_pH` with ``normalize=True``) is within ``threshold`` of
    zero. Every sequence in the batch is bisected in step, and each stops as soon
    as it has converged.

    Parameters
    ----------
    sequences : str, sequence of str, or ResidueCounts
        The batch.
    threshold : float, optional
        Convergence tolerance on the mean charge. Default 0.02.

    Returns
    -------
    numpy.ndarray
        ``(n_sequences,)`` isoelectric points. A sequence with no titratable
        residues has no defined pI; it returns 7.0, the first midpoint tried,
        because its charge is zero at every pH.

    Examples
    --------
    >>> isoelectric_point(["GSGS", "DDDDDDKK"]).round(2).tolist()
    [7.0, 3.61]
    """
    titratable, n_titratable = _titratable_counts(as_counts(sequences))
    n_sequences = n_titratable.size

    low = np.full(n_sequences, PI_SEARCH_RANGE[0])
    high = np.full(n_sequences, PI_SEARCH_RANGE[1])
    charge = np.zeros(n_sequences)
    steps = np.zeros(n_sequences, dtype=np.int64)
    widenings = np.zeros(n_sequences, dtype=np.int64)
    pI = np.full(n_sequences, np.nan)

    # the sequences still being bisected
    rows = np.arange(n_sequences)
    while rows.size:
        steps[rows] += 1

        # every PI_STEPS_PER_WIDENING steps without converging, widen the range
        # by a pH unit on the side the charge says the pI lies - or give up and
        # take the midpoint, once that has happened PI_MAX_WIDENINGS times
        widen = rows[steps[rows] == PI_STEPS_PER_WIDENING]
        if widen.size:
            give_up = widen[widenings[widen] == PI_MAX_WIDENINGS]
            pI[give_up] = 0.5 * (high[give_up] + low[give_up])
            rows = np.setdiff1d(rows, give_up, assume_unique=True)
            widen = np.setdiff1d(widen, give_up, assume_unique=True)
            widenings[widen] += 1
            steps[widen] = 0
            basic = charge[widen] > 0
            high[widen[basic]] += 1
            low[widen[~basic]] -= 1
            if not rows.size:
                break

        middle = 0.5 * (high[rows] + low[rows])
        current = _mean_charge(titratable[rows], n_titratable[rows], middle)
        charge[rows] = current

        too_basic = current > threshold
        too_acidic = current < -threshold
        converged = ~(too_basic | too_acidic)

        low[rows[too_basic]] = middle[too_basic]
        high[rows[too_acidic]] = middle[too_acidic]
        pI[rows[converged]] = middle[converged]
        rows = rows[~converged]

    return pI
