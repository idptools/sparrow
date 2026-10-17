"""Sequence properties, computed in batches.

This package is the one place in sparrow where sequence-derived properties are
actually calculated. :class:`~sparrow.protein.Protein` and :mod:`sparrow.nardini`
both call into it rather than carrying arithmetic of their own, so a property
and its NARDINI+ counterpart cannot drift apart.

Every function here works on a batch: it takes many sequences (or many
arrangements of one sequence) and returns an array. A single sequence is simply
a batch of one, which is how ``Protein`` uses it. Each sequence gets the same
value - to the last bit - alone or in a batch of any size.

The batch functions can be imported from the package itself, and
:func:`compute` gives any of the ``Protein`` properties for a whole batch by
name::

    from sparrow.properties import compute, fraction_charged, kappa

    sequences = ["MEEEKKKKSSSTTTDDD", "GSGSGSGSRGRGRG"]
    fcr = fraction_charged(sequences)              # one value per sequence
    table = compute(sequences, ["FCR", "kappa", "SCD"])
    table["kappa"]                                 # the same as Protein(s).kappa

Modules
-------
:mod:`~sparrow.properties.encoding`
    Validation, and turning sequences into integer arrays.
:mod:`~sparrow.properties.composition`
    Everything that depends only on composition: fractions, residue sets,
    ratios, mean scales, molecular weight, SEG complexity.
:mod:`~sparrow.properties.charge`
    pH-dependent charge and the isoelectric point.
:mod:`~sparrow.properties.patches`
    Patch fractions.
:mod:`~sparrow.properties.windows`
    Sliding-window sums and counts.
:mod:`~sparrow.properties.patterning`
    How residues are arranged: kappa, delta, the patterning percentile, IWD,
    SCD and SHD.
:mod:`~sparrow.properties.profiles`
    Linear profiles: any of these properties along a sequence, one window at
    a time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sparrow.properties import charge, composition, encoding, patches, patterning, profiles, windows
from sparrow.properties.charge import charge_at_pH, isoelectric_point
from sparrow.properties.composition import (
    ResidueCounts,
    amino_acid_fractions,
    count_residues,
    fraction_aliphatic,
    fraction_aromatic,
    fraction_chain_expanding,
    fraction_charged,
    fraction_disorder_promoting,
    fraction_in_set,
    fraction_negative,
    fraction_polar,
    fraction_positive,
    fraction_proline,
    hydrophobicity,
    log_residue_ratio,
    mean_scale,
    molecular_weight,
    net_charge_per_residue,
    polyproline_II_propensity,
    seg_complexity,
)
from sparrow.properties.encoding import EncodedBatch, encode_batch
from sparrow.properties.patches import patch_fraction, rg_patch_fraction
from sparrow.properties.patterning import (
    binary_asymmetry,
    bivariate_iwd_charged_weighted,
    delta,
    iwd,
    iwd_charged_weighted,
    kappa,
    kappa_x,
    patterning_percentile,
    scd,
    shd,
    sigma,
)
from sparrow.properties.profiles import fraction_profile, net_charge_profile, scale_profile, window_profile
from sparrow.sparrow_exceptions import SparrowException

__all__ = [
    # the modules
    "charge",
    "composition",
    "encoding",
    "patches",
    "patterning",
    "profiles",
    "windows",
    # a batch, encoded or counted once and passed to many functions
    "EncodedBatch",
    "encode_batch",
    "ResidueCounts",
    "count_residues",
    # every property by the name of its Protein property
    "PROPERTY_NAMES",
    "compute",
    # composition
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
    # charge
    "charge_at_pH",
    "isoelectric_point",
    # patches
    "patch_fraction",
    "rg_patch_fraction",
    # patterning
    "sigma",
    "delta",
    "kappa_x",
    "kappa",
    "binary_asymmetry",
    "patterning_percentile",
    "iwd",
    "iwd_charged_weighted",
    "bivariate_iwd_charged_weighted",
    "scd",
    "shd",
    # profiles
    "window_profile",
    "fraction_profile",
    "net_charge_profile",
    "scale_profile",
]


@dataclass(frozen=True)
class _PreparedBatch:
    """A batch in each of the forms the properties read: characters, codes and counts."""

    sequences: list[str]
    encoded: EncodedBatch
    counts: ResidueCounts


# Each Protein scalar property, by its name, and how to compute it for a whole
# batch - the same function, with the same arguments, that the property calls.
# Listed in the order Protein defines them.
_PROPERTIES: dict[str, Callable[[_PreparedBatch], NDArray[np.float64]]] = {
    "molecular_weight": lambda batch: composition.molecular_weight(batch.counts),
    "FCR": lambda batch: composition.fraction_charged(batch.counts),
    "fraction_positive": lambda batch: composition.fraction_positive(batch.counts),
    "fraction_negative": lambda batch: composition.fraction_negative(batch.counts),
    "NCPR": lambda batch: composition.net_charge_per_residue(batch.counts),
    "kappa": lambda batch: patterning.kappa(batch.encoded),
    "SCD": lambda batch: patterning.scd(batch.encoded, group1=["E", "D"], group2=["R", "K"]),
    "SHD": lambda batch: patterning.shd(batch.sequences),
    "fraction_aromatic": lambda batch: composition.fraction_aromatic(batch.counts),
    "fraction_aliphatic": lambda batch: composition.fraction_aliphatic(batch.counts),
    "fraction_polar": lambda batch: composition.fraction_polar(batch.counts),
    "fraction_proline": lambda batch: composition.fraction_proline(batch.counts),
    "hydrophobicity": lambda batch: composition.hydrophobicity(batch.counts),
    "fraction_chain_expanding": lambda batch: composition.fraction_chain_expanding(batch.counts),
    "fraction_disorder_promoting": lambda batch: composition.fraction_disorder_promoting(batch.counts),
    "isoelectric_point": lambda batch: charge.isoelectric_point(batch.counts),
    "PPII_propensity": lambda batch: composition.polyproline_II_propensity(batch.counts),
    "RK_ratio": lambda batch: composition.log_residue_ratio(batch.counts, "R", "K"),
    "ED_ratio": lambda batch: composition.log_residue_ratio(batch.counts, "E", "D"),
    "complexity": lambda batch: composition.seg_complexity(batch.counts),
}

#: The names :func:`compute` accepts: every scalar property of
#: :class:`~sparrow.protein.Protein`, in the order ``Protein`` defines them.
PROPERTY_NAMES = tuple(_PROPERTIES)


def compute(
    sequences: str | Sequence[str],
    names: str | Iterable[str] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Several ``Protein`` properties for a whole batch, by name.

    Each value is the one the ``Protein`` property of that name gives for that
    sequence, to the last bit: the same function is called with the same
    arguments, only once for the batch rather than once per sequence. The batch
    is encoded and counted once, however many properties are asked for.

    Parameters
    ----------
    sequences : str or sequence of str
        The batch. Lower-case residues are upper-cased first.
    names : str, iterable of str, or None, optional
        The properties to compute, by their ``Protein`` names (see
        :data:`PROPERTY_NAMES`) - for example ``["FCR", "kappa", "SCD"]``. One
        name may be passed on its own. Default None, which computes all of them.

    Returns
    -------
    dict of str to numpy.ndarray
        One ``(n_sequences,)`` array per property, in the order asked for; a
        name asked for twice appears once.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If a name is not one of :data:`PROPERTY_NAMES` (nothing is computed),
        or a sequence contains a non-standard residue.

    Examples
    --------
    >>> table = compute(["EEEEEKKKKK", "EKEKEKEKEK"], ["FCR", "kappa"])
    >>> table["FCR"].tolist()
    [1.0, 1.0]
    >>> bool(table["kappa"][0] > table["kappa"][1])
    True
    """
    if names is None:
        wanted = list(PROPERTY_NAMES)
    elif isinstance(names, str):
        wanted = [names]
    else:
        wanted = list(dict.fromkeys(names))

    unknown = [name for name in wanted if name not in _PROPERTIES]
    if unknown:
        raise SparrowException(f"Unknown property name(s) {unknown}; compute() knows {list(PROPERTY_NAMES)}")

    batch_sequences = encoding.as_batch(sequences)
    encoded = encode_batch(batch_sequences)
    batch = _PreparedBatch(sequences=batch_sequences, encoded=encoded, counts=count_residues(encoded))
    return {name: _PROPERTIES[name](batch) for name in wanted}
