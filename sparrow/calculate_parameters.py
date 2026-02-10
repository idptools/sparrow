"""Utility functions for computing sequence-derived parameters.

This module contains lightweight, dependency-minimal helpers for amino acid
composition, sequence complexity, and hydrophobicity calculations.
"""

import math

import numpy as np

from sparrow.data import amino_acids

from . import sparrow_exceptions


# .................................................................
#
def calculate_aa_fractions(s):
    """Compute per-amino-acid fractional composition.

    Parameters
    ----------
    s : str
        Amino acid sequence (uppercase one-letter codes expected).

    Returns
    -------
    dict[str, float]
        Mapping from each standard amino acid to its fractional occurrence
        (counts divided by total sequence length).

    Examples
    --------
    >>> calculate_aa_fractions("ACAA")['A']
    0.75
    """

    aa_dict = {}
    for i in amino_acids.VALID_AMINO_ACIDS:
        aa_dict[i] = 0

    for i in s:
        aa_dict[i] = aa_dict[i] + 1

    len_s = len(s)
    for i in amino_acids.VALID_AMINO_ACIDS:
        aa_dict[i] = aa_dict[i] / len_s

    return aa_dict


def calculate_seg_complexity(s, alphabet=amino_acids.VALID_AMINO_ACIDS):
    """Calculate Wootton-Federhen (SEG) compositional complexity.

    This is the Shannon-like compositional complexity used by the classic
    SEG algorithm. Larger negative summed probabilities (before the sign
    inversion) indicate more diverse composition; the returned value is
    positive.

    Parameters
    ----------
    s : str
        Amino acid sequence.
    alphabet : iterable[str], optional
        Alphabet to consider (default: the 20 standard amino acids). Residues
        not present in ``alphabet`` are ignored in probability estimates.

    Returns
    -------
    float
        Compositional complexity of the sequence (>= 0).
    """

    alphabet_size = len(alphabet)
    seq_len = len(s)

    complexity = 0
    for a in alphabet:
        p = s.count(a) / seq_len

        if p > 0:
            complexity = p * math.log(p, alphabet_size) + complexity

    return -complexity


# .................................................................
#
def calculate_hydrophobicity(s, mode="KD", normalize=False):
    """Compute mean hydrophobicity for a sequence.

    Parameters
    ----------
    s : str
        Amino acid sequence.
    mode : {'KD'}, optional
        Hydrophobicity scale selector. Only ``'KD'`` (Kyte-Doolittle) implemented.
    normalize : bool, optional
        If True, use normalized (0-1) scale values.

    Returns
    -------
    float
        Mean per-residue hydrophobicity under the selected scale.

    Raises
    ------
    sparrow_exceptions.CalculationException
        If an invalid residue or unknown mode is encountered.
    """
    return np.mean(calculate_linear_hydrophobicity(s, mode, normalize))


# .................................................................
#
def calculate_linear_hydrophobicity(s, mode="KD", normalize=False):
    """Return per-residue hydrophobicity values.

    Parameters
    ----------
    s : str
        Amino acid sequence.
    mode : {'KD'}, optional
        Hydrophobicity scale selector. Only ``'KD'`` implemented.
    normalize : bool, optional
        If True, return normalized (0-1) hydrophobicity values.

    Returns
    -------
    list[float]
        Hydrophobicity value for each residue in ``s``.

    Raises
    ------
    sparrow_exceptions.CalculationException
        If an invalid residue or unknown mode is encountered.

    Examples
    --------
    >>> calculate_linear_hydrophobicity('AA', mode='KD')  # doctest: +NORMALIZE_WHITESPACE
    [1.8, 1.8]
    """

    if mode == "KD":
        try:
            if normalize:
                return [amino_acids.AA_hydro_KD_normalized[r] for r in s]
            else:
                return [amino_acids.AA_hydro_KD[r] for r in s]
        except KeyError:
            raise sparrow_exceptions.CalculationException(
                "Invalid residue found in %s" % (s)
            )
    else:
        raise sparrow_exceptions.CalculationException(
            "Invalid mode passed: %s" % (mode)
        )
