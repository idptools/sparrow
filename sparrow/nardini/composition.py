"""NARDINI+ compositional features.

These are the composition half of a NARDINI+ vector: 54 numbers describing what a
sequence is *made of*, as opposed to how those residues are *arranged* (which is
what :mod:`sparrow.nardini.patterning` measures).

The feature set, in canonical order, is:

* the 20 individual amino acid fractions (``'Frac A'`` ... ``'Frac Y'``);
* five grouped fractions (``'Frac K+R'``, ``'Frac D+E'``, ``'Frac Polar'``,
  ``'Frac Aliphatic'``, ``'Frac Aromatic'``);
* two log-ratios (``'R/K Ratio'``, ``'E/D Ratio'``);
* seven whole-sequence properties (``'Frac Chain Expanding'``, ``'FCR'``,
  ``'NCPR'``, ``'Hydrophobicity'``, ``'Disorder Promoting'``, ``'Iso point'``,
  ``'PPII'``);
* 19 single-residue patch fractions (``'A Patch'`` ... ``'Y Patch'``); and
* the RG-motif patch fraction (``'RG Frac'``).

Tryptophan has no patch feature. This matches the NARDINI+ reference
implementation, which drops it because W patches essentially do not occur in the
proteome.

Unlike the patterning features, these are *raw* values. Turning them into
z-scores requires a background distribution estimated over a reference set of
IDRs - see :mod:`sparrow.nardini.background`.

Where the numbers come from
---------------------------
This module *assembles* the feature vector; it implements none of the quantities.
Each one comes from the same function behind the corresponding
:class:`~sparrow.protein.Protein` property:

* the amino acid fractions, grouped fractions, ratios and whole-sequence
  properties - :mod:`sparrow.properties.composition` and
  :mod:`sparrow.properties.charge`;
* the patch features - :mod:`sparrow.properties.patches`.

A batch is encoded once; every compositional feature is read off its residue
counts, and every patch feature found in one pass along the whole batch, so
``composition_matrix`` over a whole IDRome takes about a second.
``composition_values(seq)['FCR']`` and ``Protein(seq).FCR`` are the same number by
construction, and a test asserts that for every feature with a counterpart.

.. warning::
   One definition deliberately differs. ``'Frac Polar'`` counts cysteine
   (:data:`sparrow.data.amino_acids.NARDINI_POLAR`, QNSTGCH);
   :attr:`sparrow.protein.Protein.fraction_polar` does not (GHNQST), so the two
   differ by the cysteine content. Both conventions are used, and the bundled
   backgrounds and GIN centroids are defined against this one, so they are kept
   distinct. Only the residue set differs - the implementation is shared.

.. note::
   These values feed the bundled per-species backgrounds and the GIN centroids.
   The current arithmetic reproduces the values those were built from to within a
   few units in the last place - the exact last bit of a floating-point sum
   depends on the order it is added up in, and even Python's own ``sum`` changed
   that in 3.12 - and leaves every GIN cluster call over the 24,508 human IDRs
   unchanged. Changing what a feature *means* would move it far more than that,
   and would mean rebuilding the bundled data.
"""

from collections import OrderedDict

import numpy as np

from sparrow.data import amino_acids
from sparrow.properties import charge, composition, encoding, patches
from sparrow.sparrow_exceptions import SparrowException

__all__ = [
    "PATCH_INTERRUPTION",
    "PATCH_MIN_TARGET_COUNT",
    "RG_MIN_ADJACENT_PAIRS",
    "PATCH_RESIDUES",
    "COMPOSITION_FEATURE_NAMES",
    "composition_feature_names",
    "composition_values",
    "composition_matrix",
]

#: Maximum number of non-target residues bridged inside a patch.
PATCH_INTERRUPTION = patches.DEFAULT_INTERRUPTION

#: Minimum number of target residues required for a patch to count.
PATCH_MIN_TARGET_COUNT = patches.DEFAULT_MIN_TARGET_COUNT

#: Minimum number of adjacent ``RG`` pairs required for an RG patch to count.
RG_MIN_ADJACENT_PAIRS = patches.DEFAULT_MIN_ADJACENT_RG_PAIRS

#: Residues that get a patch feature. Tryptophan is deliberately excluded.
PATCH_RESIDUES = tuple(aa for aa in amino_acids.VALID_AMINO_ACIDS if aa != "W")

_GROUPED_FRACTIONS = OrderedDict(
    [
        ("Frac K+R", amino_acids.POS),
        ("Frac D+E", amino_acids.NEG),
        ("Frac Polar", amino_acids.NARDINI_POLAR),
        ("Frac Aliphatic", amino_acids.ALI),
        ("Frac Aromatic", amino_acids.ARO),
    ]
)

_SCALAR_FEATURES = (
    "Frac Chain Expanding",
    "FCR",
    "NCPR",
    "Hydrophobicity",
    "Disorder Promoting",
    "Iso point",
    "PPII",
)

#: All 54 compositional feature names, in canonical order.
COMPOSITION_FEATURE_NAMES = tuple(
    [f"Frac {aa}" for aa in amino_acids.VALID_AMINO_ACIDS]
    + list(_GROUPED_FRACTIONS)
    + ["R/K Ratio", "E/D Ratio"]
    + list(_SCALAR_FEATURES)
    + [f"{aa} Patch" for aa in PATCH_RESIDUES]
    + ["RG Frac"]
)

# 'Frac A' ... 'Frac Y' are read straight off the amino acid fraction matrix,
# whose columns are in encoding.RESIDUES order
assert tuple(amino_acids.VALID_AMINO_ACIDS) == encoding.RESIDUES


def composition_feature_names():
    """Return the ordered compositional feature names.

    Returns
    -------
    tuple of str
        The 54 feature names in canonical order.

    Examples
    --------
    >>> names = composition_feature_names()
    >>> len(names), names[0], names[-1]
    (54, 'Frac A', 'RG Frac')
    """
    return COMPOSITION_FEATURE_NAMES


def _patch_columns(encoded):
    """The 19 single-residue patch fractions and the RG fraction, one column each."""
    columns = [
        patches.patch_fraction(
            encoded,
            residue_selector=aa,
            interruption=PATCH_INTERRUPTION,
            min_target_count=PATCH_MIN_TARGET_COUNT,
        )
        for aa in PATCH_RESIDUES
    ]
    columns.append(
        patches.rg_patch_fraction(
            encoded,
            interruption=PATCH_INTERRUPTION,
            min_adjacent_rg_pairs=RG_MIN_ADJACENT_PAIRS,
        )
    )
    return columns


def composition_matrix(sequences, validate=True):
    """Compute the 54 compositional features for many sequences.

    Parameters
    ----------
    sequences : iterable of str
        Amino acid sequences.
    validate : bool, optional
        If True (default), also reject empty sequences. Non-standard residues are
        rejected either way, and lower-case residues are upper-cased.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_sequences, 54)`` in canonical feature order.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``validate`` is True and a sequence is empty, or if any sequence
        contains a non-standard residue.

    Examples
    --------
    >>> composition_matrix(["MEEEKKKK", "QQQQNNNN"]).shape
    (2, 54)
    """
    batch = encoding.as_batch(list(sequences))
    if validate:
        batch = [
            encoding.validate_protein_sequence(
                sequence,
                allow_empty=False,
                uppercase=True,
                exception_cls=SparrowException,
                sequence_name="sequence",
            )
            for sequence in batch
        ]

    if not batch:
        return np.zeros((0, len(COMPOSITION_FEATURE_NAMES)), dtype=np.float64)

    # every feature below comes from one encoding of the batch, through the same
    # functions behind the corresponding Protein properties. The one deliberate
    # difference is 'Frac Polar' - see amino_acids.NARDINI_POLAR.
    encoded = encoding.encode_batch(batch)
    counts = composition.count_residues(encoded)

    columns = list(composition.amino_acid_fractions(counts).T)
    columns += [composition.fraction_in_set(counts, residues) for residues in _GROUPED_FRACTIONS.values()]
    columns += [
        composition.log_residue_ratio(counts, "R", "K"),
        composition.log_residue_ratio(counts, "E", "D"),
        composition.fraction_chain_expanding(counts),
        composition.fraction_charged(counts),
        composition.net_charge_per_residue(counts),
        composition.hydrophobicity(counts),
        composition.fraction_disorder_promoting(counts),
        charge.isoelectric_point(counts),
        composition.polyproline_II_propensity(counts),
    ]

    matrix = np.column_stack(columns + _patch_columns(encoded))

    assert matrix.shape == (len(batch), len(COMPOSITION_FEATURE_NAMES)), matrix.shape
    return matrix


def composition_values(sequence, validate=True):
    """Compute the 54 NARDINI+ compositional features for one sequence.

    Parameters
    ----------
    sequence : str
        Amino acid sequence (standard residues only).
    validate : bool, optional
        If True (default) check that the sequence is non-empty and made only of
        the 20 standard amino acids.

    Returns
    -------
    collections.OrderedDict
        Mapping from feature name to value, in the order given by
        :func:`composition_feature_names`.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the sequence is empty or contains a non-standard residue.

    Examples
    --------
    >>> values = composition_values("MEEEKKKKSSSTTTDDDQQQQNNNN")
    >>> len(values)
    54
    >>> round(values["FCR"], 4)
    0.4
    """
    if not isinstance(sequence, str):
        raise SparrowException("sequence must be a string")
    row = composition_matrix([sequence], validate=validate)[0]
    return OrderedDict(zip(COMPOSITION_FEATURE_NAMES, row.tolist()))
