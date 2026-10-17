"""Properties that depend only on a sequence's composition.

Everything here is a function of how many of each residue a sequence has, not of
where they sit. That is what lets a whole batch be computed in one pass: the
sequences are counted once into an ``(n_sequences, 20)`` matrix
(:func:`count_residues`), and every property is then a few array operations on
that matrix, however long or numerous the sequences are.

Each function accepts either sequences or a :class:`ResidueCounts` that has
already been computed. Pass the counts when you need several properties of the
same batch, so the sequences are only counted once::

    counts = count_residues(sequences)
    fcr = fraction_charged(counts)
    ncpr = net_charge_per_residue(counts)

How the arithmetic is done
--------------------------
A fraction is the summed integer count of the residues in the set, divided by the
length - one rounding, whatever the set. That makes it independent of the order
the residues are listed in, which the previous implementation (summing one
per-residue fraction at a time) was not, and it is the more accurate of the two.
Weighted sums over residues (mean scales, molecular weight) accumulate one
residue column at a time in a fixed order, so a sequence gets the same value
whether it is computed alone or as part of a batch of any size.

Empty sequences
---------------
An empty sequence has every fraction, its SEG complexity and its molecular weight
equal to 0.0. Its mean over a scale is NaN, because there is nothing to average.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sparrow.data import amino_acids
from sparrow.properties import encoding
from sparrow.sparrow_exceptions import SparrowException

__all__ = [
    "WATER_MASS_DA",
    "ResidueCounts",
    "count_residues",
    "as_counts",
    "amino_acid_fractions",
    "fraction_in_set",
    "fraction_positive",
    "fraction_negative",
    "fraction_charged",
    "net_charge_per_residue",
    "fraction_aromatic",
    "fraction_aliphatic",
    "fraction_polar",
    "fraction_proline",
    "fraction_chain_expanding",
    "fraction_disorder_promoting",
    "log_residue_ratio",
    "mean_scale",
    "hydrophobicity",
    "polyproline_II_propensity",
    "molecular_weight",
    "seg_complexity",
]

#: Mass of the water lost when a peptide bond forms, in Daltons (Da). sparrow
#: has always used 18, the nominal mass; the monoisotopic mass is 18.011.
WATER_MASS_DA = 18.0


@dataclass(frozen=True)
class ResidueCounts:
    """How many of each standard residue every sequence in a batch has.

    Attributes
    ----------
    counts : numpy.ndarray
        ``int64`` array of shape ``(n_sequences, 20)``; column ``j`` counts
        residue :data:`sparrow.properties.encoding.RESIDUES` ``[j]``.
    lengths : numpy.ndarray
        ``int64`` array of shape ``(n_sequences,)``, each sequence's length.
    """

    counts: NDArray[np.int64]
    lengths: NDArray[np.int64]

    def __len__(self) -> int:
        """Number of sequences in the batch."""
        return int(self.lengths.size)


#: What every property in this module accepts.
Batch = str | Sequence[str] | ResidueCounts


def count_residues(sequences: str | Sequence[str]) -> ResidueCounts:
    """Count every standard residue in every sequence of a batch.

    Parameters
    ----------
    sequences : str or sequence of str
        One sequence, or many. Lower-case residues are upper-cased first.

    Returns
    -------
    ResidueCounts
        The counts and lengths of the batch.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If any sequence contains a non-standard residue.

    Examples
    --------
    >>> counts = count_residues(["MEEK", "GS"])
    >>> counts.lengths.tolist()
    [4, 2]
    """
    batch = encoding.as_batch(sequences)
    codes, lengths = encoding.concatenated_residue_codes(batch)

    if len(batch) == 1:
        # the same counts as below, without the bookkeeping that tells sequences
        # apart - worth having, because Protein counts one sequence at a time
        counts = np.bincount(codes, minlength=encoding.NUM_RESIDUES).astype(np.int64)[None, :]
    else:
        # one bincount over (sequence index, residue code) pairs counts the batch
        owner = np.repeat(np.arange(len(batch), dtype=np.int64), lengths)
        flat = np.bincount(owner * encoding.NUM_RESIDUES + codes, minlength=len(batch) * encoding.NUM_RESIDUES)
        counts = flat.astype(np.int64).reshape(len(batch), encoding.NUM_RESIDUES)
    return ResidueCounts(counts=counts, lengths=lengths)


def as_counts(sequences: Batch) -> ResidueCounts:
    """Counts for a batch, counting the sequences only if that has not been done.

    Parameters
    ----------
    sequences : str, sequence of str, or ResidueCounts
        The batch. Counts are returned unchanged.

    Returns
    -------
    ResidueCounts
        The batch's counts.
    """
    if isinstance(sequences, ResidueCounts):
        return sequences
    return count_residues(sequences)


def _columns(residues: Iterable[str]) -> list[int]:
    """Residue codes of a set of residues, each once.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If a residue is not one of the 20 standard amino acids.
    """
    columns = []
    for residue in dict.fromkeys(residues):
        if residue not in encoding.RESIDUE_INDEX:
            raise SparrowException(f"{residue!r} is not one of the 20 standard amino acids")
        columns.append(encoding.RESIDUE_INDEX[residue])
    return columns


def _per_residue(numerator: NDArray[np.generic], lengths: NDArray[np.int64]) -> NDArray[np.float64]:
    """``numerator / lengths``, giving 0.0 for an empty sequence rather than NaN.

    An empty sequence has a numerator of zero, so dividing it by 1 instead of 0
    gives 0.0 without a separate mask.
    """
    result: NDArray[np.float64] = np.divide(numerator, np.maximum(lengths, 1), dtype=np.float64)
    return result


def amino_acid_fractions(sequences: Batch) -> NDArray[np.float64]:
    """Fraction of each sequence made up by each of the 20 standard amino acids.

    Parameters
    ----------
    sequences : str, sequence of str, or ResidueCounts
        The batch.

    Returns
    -------
    numpy.ndarray
        ``(n_sequences, 20)`` array; column ``j`` is residue
        :data:`sparrow.properties.encoding.RESIDUES` ``[j]``. Each row sums to 1,
        or is all zeros for an empty sequence.

    Examples
    --------
    >>> amino_acid_fractions("ACAA")[0, :2].tolist()
    [0.75, 0.25]
    """
    counts = as_counts(sequences)
    return _per_residue(counts.counts, counts.lengths[:, None])


def fraction_in_set(sequences: Batch, residues: Iterable[str]) -> NDArray[np.float64]:
    """Fraction of each sequence made up by a set of residues.

    Parameters
    ----------
    sequences : str, sequence of str, or ResidueCounts
        The batch.
    residues : iterable of str
        The residues to count. Repeats are counted once, and order does not
        matter.

    Returns
    -------
    numpy.ndarray
        ``(n_sequences,)`` fractions between 0 and 1.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If a residue in ``residues`` is not a standard amino acid.

    Examples
    --------
    >>> fraction_in_set(["QQNA", "AAAA"], "QN").tolist()
    [0.75, 0.0]
    """
    counts = as_counts(sequences)
    selected = counts.counts[:, _columns(residues)].sum(axis=1)
    return _per_residue(selected, counts.lengths)


def fraction_positive(sequences: Batch) -> NDArray[np.float64]:
    """Fraction of residues carrying a formal positive charge (R, K)."""
    return fraction_in_set(sequences, amino_acids.POS)


def fraction_negative(sequences: Batch) -> NDArray[np.float64]:
    """Fraction of residues carrying a formal negative charge (E, D)."""
    return fraction_in_set(sequences, amino_acids.NEG)


def fraction_charged(sequences: Batch) -> NDArray[np.float64]:
    """Fraction of charged residues, FCR (E, D, R, K)."""
    return fraction_in_set(sequences, amino_acids.CHARGE)


def net_charge_per_residue(sequences: Batch) -> NDArray[np.float64]:
    """Net charge per residue, NCPR: (positive count - negative count) / length.

    Parameters
    ----------
    sequences : str, sequence of str, or ResidueCounts
        The batch.

    Returns
    -------
    numpy.ndarray
        ``(n_sequences,)`` values between -1 and +1.
    """
    counts = as_counts(sequences)
    positive = counts.counts[:, _columns(amino_acids.POS)].sum(axis=1)
    negative = counts.counts[:, _columns(amino_acids.NEG)].sum(axis=1)
    return _per_residue(positive - negative, counts.lengths)


def fraction_aromatic(sequences: Batch) -> NDArray[np.float64]:
    """Fraction of aromatic residues (F, W, Y)."""
    return fraction_in_set(sequences, amino_acids.ARO)


def fraction_aliphatic(sequences: Batch) -> NDArray[np.float64]:
    """Fraction of aliphatic residues (A, L, M, I, V)."""
    return fraction_in_set(sequences, amino_acids.ALI)


def fraction_polar(sequences: Batch) -> NDArray[np.float64]:
    """Fraction of polar residues (G, H, N, Q, S, T).

    Notes
    -----
    Cysteine is **not** counted. NARDINI+ uses a polar set that includes it
    (:data:`sparrow.data.amino_acids.NARDINI_POLAR`), so its ``'Frac Polar'``
    feature differs from this by the cysteine content. Both conventions are
    in use, so both sets are kept.
    """
    return fraction_in_set(sequences, amino_acids.POLAR)


def fraction_proline(sequences: Batch) -> NDArray[np.float64]:
    """Fraction of proline residues."""
    return fraction_in_set(sequences, "P")


def fraction_chain_expanding(sequences: Batch) -> NDArray[np.float64]:
    """Fraction of chain-expanding residues (E, D, R, K, P).

    Chain-expanding residues drive a disordered chain towards more expanded
    conformations, through a mix of charge-charge repulsion and backbone
    stiffening.
    """
    return fraction_in_set(sequences, amino_acids.EXPANDING)


def fraction_disorder_promoting(sequences: Batch) -> NDArray[np.float64]:
    """Fraction of disorder-promoting residues (T, A, G, R, D, H, Q, K, S, E, P).

    The set is :data:`sparrow.data.amino_acids.DISORDER_PROMOTING`.
    """
    return fraction_in_set(sequences, amino_acids.DISORDER_PROMOTING)


def log_residue_ratio(sequences: Batch, numerator: str, denominator: str) -> NDArray[np.float64]:
    """``log10`` of the ratio of two residue counts, with a +1 pseudo-count.

    The pseudo-count keeps the ratio finite when either residue is absent, which
    is common in short IDRs. This is the convention NARDINI+ uses for its
    ``'R/K Ratio'`` and ``'E/D Ratio'`` features.

    Parameters
    ----------
    sequences : str, sequence of str, or ResidueCounts
        The batch.
    numerator : str
        Residue on top of the ratio.
    denominator : str
        Residue underneath.

    Returns
    -------
    numpy.ndarray
        ``log10((count(numerator) + 1) / (count(denominator) + 1))`` for each
        sequence: zero when the two are equally abundant, positive when the
        numerator dominates.
    """
    if len(numerator) != 1 or len(denominator) != 1:
        raise SparrowException(f"log_residue_ratio takes two single residues, not {numerator!r} and {denominator!r}")
    counts = as_counts(sequences)
    (top,) = _columns(numerator)
    (bottom,) = _columns(denominator)
    ratio = (counts.counts[:, top] + 1) / (counts.counts[:, bottom] + 1)
    result: NDArray[np.float64] = np.log10(ratio)
    return result


def _weighted_residue_sum(counts: ResidueCounts, values: NDArray[np.float64]) -> NDArray[np.float64]:
    """``sum_j counts[:, j] * values[j]`` for every sequence.

    Each row is reduced on its own, not as a matrix product, so the order of the
    additions - and therefore the result, to the last bit - does not depend on
    how many sequences are in the batch. A residue whose value is NaN (missing
    from the scale) only matters for the sequences that contain it.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If a sequence contains a residue with no value.
    """
    missing = np.isnan(values)
    if np.any(missing):
        needed = np.flatnonzero(missing)[counts.counts[:, missing].any(axis=0)]
        if needed.size:
            raise SparrowException(f"The scale has no value for residue(s) {[encoding.RESIDUES[j] for j in needed]}")
        values = np.where(missing, 0.0, values)
    total: NDArray[np.float64] = (counts.counts * values).sum(axis=1)
    return total


def _mean_of_values(counts: ResidueCounts, values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mean of a per-residue value array over each sequence; NaN when empty."""
    total = _weighted_residue_sum(counts, values)
    out = np.full(len(counts), np.nan, dtype=np.float64)
    np.divide(total, counts.lengths, out=out, where=counts.lengths > 0)
    return out


# the built-in scales as arrays indexed by residue code, built once
_KYTE_DOOLITTLE = encoding.scale_vector(amino_acids.AA_hydro_KD)
_PPII_HILSER = encoding.scale_vector(amino_acids.AA_PPII_HILSER)
_MOLECULAR_WEIGHT = encoding.scale_vector(amino_acids.AA_MOLECULAR_WEIGHT)


def mean_scale(sequences: Batch, scale: Mapping[str, float | None]) -> NDArray[np.float64]:
    """Mean of a per-residue scale over each sequence.

    Parameters
    ----------
    sequences : str, sequence of str, or ResidueCounts
        The batch.
    scale : mapping of str to float
        Value of each residue, e.g. :data:`sparrow.data.amino_acids.AA_hydro_KD`.

    Returns
    -------
    numpy.ndarray
        ``(n_sequences,)`` means; NaN for an empty sequence.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If a sequence contains a residue the scale has no value for.
    """
    return _mean_of_values(as_counts(sequences), encoding.scale_vector(scale))


def hydrophobicity(sequences: Batch) -> NDArray[np.float64]:
    """Mean Kyte-Doolittle hydrophobicity (:data:`sparrow.data.amino_acids.AA_hydro_KD`)."""
    return _mean_of_values(as_counts(sequences), _KYTE_DOOLITTLE)


def polyproline_II_propensity(sequences: Batch) -> NDArray[np.float64]:
    """Mean polyproline II (PPII) propensity on the Hilser scale.

    PPII is an extended left-handed helical conformation that disordered regions
    sample readily, and a high mean propensity is associated with more expanded
    ensembles. The scale is :data:`sparrow.data.amino_acids.AA_PPII_HILSER`.
    """
    return _mean_of_values(as_counts(sequences), _PPII_HILSER)


def molecular_weight(sequences: Batch) -> NDArray[np.float64]:
    """Molecular weight of each sequence, in Daltons (Da).

    The sum of the free-residue molecular weights minus one water
    (:data:`WATER_MASS_DA`) per peptide bond - one fewer than the number of
    residues, so a single residue is its free amino acid weight.

    Parameters
    ----------
    sequences : str, sequence of str, or ResidueCounts
        The batch.

    Returns
    -------
    numpy.ndarray
        ``(n_sequences,)`` weights in Da; 0.0 for an empty sequence.

    Examples
    --------
    >>> molecular_weight(["G", "GG"]).round(1).tolist()
    [75.1, 132.2]
    """
    counts = as_counts(sequences)
    residue_mass = _weighted_residue_sum(counts, _MOLECULAR_WEIGHT)
    peptide_bonds = np.maximum(counts.lengths - 1, 0)
    return np.where(counts.lengths > 0, residue_mass - WATER_MASS_DA * peptide_bonds, 0.0)


def seg_complexity(
    sequences: Batch,
    alphabet: Sequence[str] = encoding.RESIDUES,
) -> NDArray[np.float64]:
    """Wootton-Federhen (SEG) compositional complexity of each sequence.

    The Shannon entropy of the residue composition, in units of the alphabet
    size, as used by the classic SEG algorithm: ``-sum_a p_a log_K(p_a)`` with
    ``K = len(alphabet)``. Higher means more compositionally diverse.

    Parameters
    ----------
    sequences : str, sequence of str, or ResidueCounts
        The batch.
    alphabet : sequence of str, optional
        Residues to include in the sum. Default the 20 standard amino acids.
        Residues outside it still count towards the length, so they lower every
        ``p_a``.

    Returns
    -------
    numpy.ndarray
        ``(n_sequences,)`` complexities, each >= 0; 0.0 for an empty sequence.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``alphabet`` has fewer than two residues (the entropy has no base) or
        contains a non-standard one.

    References
    ----------
    Wootton JC, Federhen S. Statistics of local complexity in amino acid
    sequences and sequence databases. Comput Chem. 1993;17(2):149-163.
    """
    counts = as_counts(sequences)
    columns = _columns(alphabet)
    if len(columns) < 2:
        raise SparrowException(f"SEG complexity needs an alphabet of at least two residues, not {list(alphabet)}")
    log_base = np.log(len(columns))

    fractions = _per_residue(counts.counts[:, columns], counts.lengths[:, None])
    # 0 log 0 is taken as 0, its limit: an absent residue's log stays 0
    logs = np.zeros_like(fractions)
    np.log(fractions, out=logs, where=fractions > 0)
    # subtracting from 0.0 rather than negating keeps a homopolymer at +0.0
    complexity: NDArray[np.float64] = 0.0 - (fractions * (logs / log_base)).sum(axis=1)
    return complexity
