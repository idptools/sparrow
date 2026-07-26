"""Stateless functions computing sequence-dependent physical properties.

Every function here takes an amino acid sequence and returns a number. They hold no
state and know nothing about :class:`~sparrow.protein.Protein`, which is what lets
them be the single implementation behind two things at once:

* the corresponding ``Protein`` properties -- :attr:`~sparrow.protein.Protein.FCR`,
  :attr:`~sparrow.protein.Protein.isoelectric_point`,
  :attr:`~sparrow.protein.Protein.PPII_propensity` and the rest; and
* the matching NARDINI+ compositional features in :mod:`sparrow.nardini`.

Those two used to compute the same quantities separately, which is exactly the kind
of duplication that drifts. Because both now call these functions, a ``Protein``
property and its NARDINI+ feature are bit-identical by construction, and a test in
``test_regressions.py`` pins that for every one of them.

.. warning::
   One definition is deliberately *not* shared. NARDINI+ counts cysteine as polar,
   sparrow does not, so :func:`fraction_polar` and the NARDINI+ ``'Frac Polar'``
   feature differ by the cysteine content. The published reference data is defined
   against the NARDINI+ convention, so neither can be changed to match the other.
   They still share this implementation -- only the residue set differs, passed
   through :func:`fraction_in_set`.

.. note::
   These values feed the bundled per-species backgrounds and GIN cluster centroids,
   so changes here are not free: a shift in the last decimal place moves every
   compositional z-score and every centroid distance. :func:`fraction_in_set`
   documents one place where the obvious simplification would do exactly that.

Adding a new property? Follow :func:`calculate_molecular_weight` as a template:
take the sequence, return a float, no state, numpy-style docstring. Expose it on
``Protein`` as a cached property rather than duplicating the arithmetic there.
"""

import numpy as np

from sparrow.data import amino_acids



def calculate_molecular_weight(sequence):
    """
    Function that returns the molecular weight of a protein sequence assuming standard
    amino acid molecular weights.

    The weight is the sum of the free-residue molecular weights minus one water
    molecule (18 Da) per peptide bond, i.e. one fewer than the number of
    residues.

    Parameters
    -------------
    sequence : str
        String containing the amino acid sequence (upper case one-letter residue codes)

    Returns
    -----------
    float
        Returns the residue or polypeptide molecular weight in Daltons (Da). An
        empty sequence returns 0.0.

    """

    # an empty sequence has no residues and no peptide bonds; handled explicitly
    # because the peptide-bond correction below would otherwise ADD one water
    if len(sequence) == 0:
        return 0.0

    # compute niave MW
    MW = 0
    for i in sequence:
        MW = MW + amino_acids.AA_MOLECULAR_WEIGHT[i]

    # subtract one water per peptide bond (there are len(sequence)-1 of them,
    # so a single residue is returned as-is)
    return MW - 18 * (len(sequence) - 1)


# .................................................................
#
def mean_residue_scale(sequence, scale):
    """Mean of a per-residue scale over a sequence.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    scale : dict
        Mapping from residue to value, e.g.
        :data:`sparrow.data.amino_acids.AA_PPII_HILSER`.

    Returns
    -------
    float
        Arithmetic mean of the scale over the residues.
    """
    return float(np.mean([scale[residue] for residue in sequence]))


def fraction_in_set(sequence, residues):
    """Fraction of a sequence made up of a set of residues.

    Notes
    -----
    Sums the per-residue fractions rather than dividing a summed count, because
    ``a/n + b/n`` and ``(a + b)/n`` are not always the same float. The NARDINI+
    compositional features have always been computed the former way, and the
    bundled per-species backgrounds and GIN centroids are defined against those
    exact values, so this preserves them bit for bit.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    residues : iterable of str
        Residues to count.

    Returns
    -------
    float
        Fraction of the sequence in ``residues``, between 0 and 1.
    """
    length = len(sequence)
    if length == 0:
        return 0.0
    return float(sum(sequence.count(residue) / length for residue in residues))


def fraction_chain_expanding(sequence):
    """Fraction of residues that are chain-expanding (E, D, R, K, P).

    Chain-expanding residues drive a disordered chain towards more expanded
    conformations, through a mix of charge-charge repulsion and backbone
    stiffening.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.

    Returns
    -------
    float
        Fraction of the sequence in
        :data:`sparrow.data.amino_acids.EXPANDING`, between 0 and 1.
    """
    return fraction_in_set(sequence, amino_acids.EXPANDING)


def fraction_disorder_promoting(sequence):
    """Fraction of residues that are disorder-promoting.

    Uses the standard disorder-promoting set
    (:data:`sparrow.data.amino_acids.DISORDER_PROMOTING`): T, A, G, R, D, H, Q, K,
    S, E, P.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.

    Returns
    -------
    float
        Fraction of the sequence that is disorder-promoting, between 0 and 1.
    """
    return fraction_in_set(sequence, amino_acids.DISORDER_PROMOTING)


def polyproline_II_propensity(sequence):
    """Mean polyproline II (PPII) propensity on the Hilser scale.

    PPII is an extended left-handed helical conformation that disordered regions
    sample readily, and a high mean propensity is associated with more expanded
    ensembles.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.

    Returns
    -------
    float
        Mean per-residue PPII propensity
        (:data:`sparrow.data.amino_acids.AA_PPII_HILSER`).
    """
    return mean_residue_scale(sequence, amino_acids.AA_PPII_HILSER)


def charge_at_pH(sequence, pH=7.4, normalize=True):
    """Mean charge of the titratable side chains at a given pH.

    Uses Henderson-Hasselbalch with the EMBOSS side-chain pKa values
    (:data:`sparrow.data.amino_acids.AA_PKA`), treating K/R/H as bases and
    E/D/Y/C as acids. Termini are not included.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    pH : float, optional
        pH at which to evaluate the charge. Default 7.4.
    normalize : bool, optional
        If True (default) divide by the number of titratable residues, giving a
        mean charge per titratable group. If False return the summed charge.

    Returns
    -------
    float
        The (mean) net charge. Returns 0.0 for a sequence with no titratable
        residues.
    """
    pKa = amino_acids.AA_PKA

    total = 0.0
    countable = 0
    for residue in sequence:
        if residue in ("K", "R", "H"):
            total += 1.0 / (1.0 + 10.0 ** (pH - pKa[residue]))
            countable += 1
        elif residue in ("E", "D", "Y", "C"):
            total += -1.0 / (1.0 + 10.0 ** (pKa[residue] - pH))
            countable += 1

    if normalize:
        if countable == 0:
            return 0.0
        return total / countable

    return total


def isoelectric_point(sequence, threshold=0.02):
    """Estimate the isoelectric point (pI) of a sequence.

    Found by bisecting on pH until the mean charge of the titratable side chains
    falls within ``threshold`` of zero. This reproduces localCIDER's
    ``get_isoelectric_point()``, including its convention of normalizing the
    charge by the number of titratable residues.

    A sequence with no titratable residues has no defined pI and returns 7.0.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    threshold : float, optional
        Convergence tolerance on the normalized charge. Default 0.02.

    Returns
    -------
    float
        The estimated isoelectric point, between 0 and 14 (the bracket is widened
        if the sequence is charged enough to push the pI outside it).
    """
    min_pH, max_pH = 0.0, 14.0

    charge = 0.0
    iterations = 0
    expansions = 0

    while True:
        iterations += 1

        # a very highly charged sequence can have its pI outside [0, 14]; widen
        # the bracket rather than bisecting forever
        if iterations == 20:
            if expansions == 10:
                return 0.5 * (max_pH + min_pH)
            expansions += 1
            iterations = 0
            if charge > 0:
                max_pH += 1
            else:
                min_pH -= 1

        mid_pH = 0.5 * (max_pH + min_pH)
        charge = charge_at_pH(sequence, mid_pH, normalize=True)

        if charge > threshold:
            min_pH = mid_pH
        elif charge < -threshold:
            max_pH = mid_pH
        else:
            return mid_pH


# .................................................................
#
def fraction_positive(sequence):
    """Fraction of residues carrying a formal positive charge (K, R)."""
    return fraction_in_set(sequence, amino_acids.POS)


def fraction_negative(sequence):
    """Fraction of residues carrying a formal negative charge (D, E)."""
    return fraction_in_set(sequence, amino_acids.NEG)


def fraction_charged(sequence):
    """Fraction of charged residues, FCR (D, E, K, R)."""
    return fraction_in_set(sequence, amino_acids.CHARGE)


def net_charge_per_residue(sequence):
    """Net charge per residue, NCPR: positive fraction minus negative fraction."""
    return fraction_positive(sequence) - fraction_negative(sequence)


def fraction_aromatic(sequence):
    """Fraction of aromatic residues (F, W, Y)."""
    return fraction_in_set(sequence, amino_acids.ARO)


def fraction_aliphatic(sequence):
    """Fraction of aliphatic residues (A, L, M, I, V)."""
    return fraction_in_set(sequence, amino_acids.ALI)


def fraction_polar(sequence):
    """Fraction of polar residues (G, H, N, Q, S, T).

    Notes
    -----
    Cysteine is **not** counted here. NARDINI+ uses a polar set that does include
    it (``'Frac Polar'`` = QNSTGCH), so ``Protein.fraction_polar`` and the NARDINI+
    feature of the same name legitimately differ by the cysteine content. The two
    definitions are both in use in the literature; sparrow keeps each where it
    belongs rather than silently changing either. Both go through
    :func:`fraction_in_set`, so there is one implementation and two residue sets.
    """
    return fraction_in_set(sequence, amino_acids.POLAR)


def log_residue_ratio(sequence, numerator, denominator):
    """``log10`` of the ratio of two residue counts, with a +1 pseudo-count.

    The pseudo-count keeps the ratio finite when either residue is absent, which
    is common in short IDRs. This is the convention NARDINI+ uses for its
    ``'R/K Ratio'`` and ``'E/D Ratio'`` features.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    numerator : str
        Residue on top of the ratio.
    denominator : str
        Residue underneath.

    Returns
    -------
    float
        ``log10((count(numerator) + 1) / (count(denominator) + 1))``. Zero when the
        two residues are equally abundant, positive when the numerator dominates.
    """
    return float(
        np.log10((sequence.count(numerator) + 1) / (sequence.count(denominator) + 1))
    )
