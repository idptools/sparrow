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
This module *assembles* the feature vector; it does not implement the individual
quantities. Each one is computed by the same code behind the corresponding
:class:`~sparrow.protein.Protein` property:

* the 20 amino acid fractions - :func:`sparrow.calculate_parameters.calculate_aa_fractions`;
* the grouped fractions, ratios and whole-sequence properties -
  :mod:`sparrow.sequence_analysis.physical_properties`;
* the patch features - :mod:`sparrow.sequence_analysis.patching`.

So ``composition_values(seq)['FCR']`` and ``Protein(seq).FCR`` are the same number
by construction rather than by coincidence, and a test asserts that for every
feature that has a counterpart.

.. warning::
   One definition deliberately differs. ``'Frac Polar'`` counts cysteine
   (QNSTGCH); :attr:`sparrow.protein.Protein.fraction_polar` does not (GHNQST), so
   the two differ by the cysteine content. Both conventions are used, and the
   bundled backgrounds and GIN centroids are defined against this one, so they are
   kept distinct. Only the residue set differs - the implementation is shared.

.. note::
   These values feed the bundled per-species backgrounds and the GIN centroids, so
   they must stay bit-identical. Changing how a feature is summed - even in a way
   that looks equivalent - shifts every compositional z-score and every centroid
   distance downstream.
"""

from collections import OrderedDict

import numpy as np

from sparrow import calculate_parameters
from sparrow.data import amino_acids
from sparrow.sequence_analysis import physical_properties
from sparrow.sequence_analysis.patching import patch_fraction, rg_patch_fraction
from sparrow.sparrow_exceptions import SparrowException
from sparrow.tools import general_tools

__all__ = [
    "PATCH_INTERRUPTION",
    "PATCH_MIN_TARGET_COUNT",
    "RG_MIN_ADJACENT_PAIRS",
    "PATCH_RESIDUES",
    "COMPOSITION_FEATURE_NAMES",
    "composition_feature_names",
    "charge_at_pH",
    "isoelectric_point",
    "composition_values",
    "composition_matrix",
]

#: Maximum number of non-target residues bridged inside a patch.
PATCH_INTERRUPTION = 2

#: Minimum number of target residues required for a patch to count.
PATCH_MIN_TARGET_COUNT = 4

#: Minimum number of adjacent ``RG`` pairs required for an RG patch to count.
RG_MIN_ADJACENT_PAIRS = 2

#: Residues that get a patch feature. Tryptophan is deliberately excluded.
PATCH_RESIDUES = tuple(aa for aa in amino_acids.VALID_AMINO_ACIDS if aa != "W")

#: NARDINI+ counts cysteine as polar; sparrow's own ``amino_acids.POLAR`` (behind
#: :attr:`sparrow.protein.Protein.fraction_polar`) does not. Both conventions are in
#: use, and the published per-species backgrounds and GIN centroids are defined
#: against this one, so the two are kept distinct rather than silently reconciled.
#: The *implementation* is shared - only the residue set differs.
_NARDINI_POLAR = tuple("QNSTGCH")

_GROUPED_FRACTIONS = OrderedDict(
    [
        ("Frac K+R", tuple("KR")),
        ("Frac D+E", tuple("DE")),
        ("Frac Polar", _NARDINI_POLAR),
        ("Frac Aliphatic", tuple("ALMIV")),
        ("Frac Aromatic", tuple("FWY")),
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


# These are general sequence properties, not NARDINI-specific ones, so they live in
# sparrow.sequence_analysis.physical_properties and are exposed on Protein as
# `p.charge_at_pH(...)` and `p.isoelectric_point`. Re-exported here because they are
# part of this module's documented API and are what the 'Iso point' feature uses.
charge_at_pH = physical_properties.charge_at_pH
isoelectric_point = physical_properties.isoelectric_point


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
    0.44
    """
    if validate:
        sequence = general_tools.validate_protein_sequence(
            sequence,
            allow_empty=False,
            uppercase=True,
            exception_cls=SparrowException,
            sequence_name="sequence",
        )

    # Everything below comes from sparrow.sequence_analysis.physical_properties, the
    # same functions behind the corresponding Protein properties, so a NARDINI+
    # feature and its Protein counterpart cannot drift apart. The one deliberate
    # exception is 'Frac Polar' - see _NARDINI_POLAR.
    features = OrderedDict()

    # individual amino acid fractions
    aa_fractions = calculate_parameters.calculate_aa_fractions(sequence)
    for aa in amino_acids.VALID_AMINO_ACIDS:
        features[f"Frac {aa}"] = aa_fractions[aa]

    # grouped fractions
    for name, residues in _GROUPED_FRACTIONS.items():
        features[name] = physical_properties.fraction_in_set(sequence, residues)

    features["R/K Ratio"] = physical_properties.log_residue_ratio(sequence, "R", "K")
    features["E/D Ratio"] = physical_properties.log_residue_ratio(sequence, "E", "D")

    # whole-sequence properties
    features["Frac Chain Expanding"] = physical_properties.fraction_chain_expanding(sequence)
    features["FCR"] = physical_properties.fraction_charged(sequence)
    features["NCPR"] = physical_properties.net_charge_per_residue(sequence)
    features["Hydrophobicity"] = physical_properties.mean_residue_scale(sequence, amino_acids.AA_hydro_KD)
    features["Disorder Promoting"] = physical_properties.fraction_disorder_promoting(sequence)
    features["Iso point"] = physical_properties.isoelectric_point(sequence)
    features["PPII"] = physical_properties.polyproline_II_propensity(sequence)

    # single-residue patch fractions
    for aa in PATCH_RESIDUES:
        features[f"{aa} Patch"] = float(
            patch_fraction(
                sequence,
                residue_selector=aa,
                interruption=PATCH_INTERRUPTION,
                min_target_count=PATCH_MIN_TARGET_COUNT,
            )
        )

    # RG-motif patch fraction
    features["RG Frac"] = float(
        rg_patch_fraction(
            sequence,
            interruption=PATCH_INTERRUPTION,
            min_adjacent_rg_pairs=RG_MIN_ADJACENT_PAIRS,
        )
    )

    return features


def composition_matrix(sequences, validate=True):
    """Compute compositional features for many sequences.

    Parameters
    ----------
    sequences : iterable of str
        Amino acid sequences.
    validate : bool, optional
        If True (default) validate each sequence.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_sequences, 54)`` in canonical feature order.

    Examples
    --------
    >>> composition_matrix(["MEEEKKKK", "QQQQNNNN"]).shape
    (2, 54)
    """
    rows = []
    for sequence in sequences:
        values = composition_values(sequence, validate=validate)
        rows.append([values[name] for name in COMPOSITION_FEATURE_NAMES])

    if not rows:
        return np.zeros((0, len(COMPOSITION_FEATURE_NAMES)), dtype=np.float64)

    return np.asarray(rows, dtype=np.float64)
