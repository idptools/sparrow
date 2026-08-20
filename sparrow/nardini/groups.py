"""Residue groupings and feature names used by the NARDINI analysis.

NARDINI coarse-grains the 20 standard amino acids into a small number of
biophysically-motivated types, then asks how non-randomly each type (and each
pair of types) is distributed along a sequence. This module defines those
groupings and the derived feature names.

Two groupings are provided:

* :data:`TYPES_8` -- the default 8-type scheme used in the NARDINI paper and in
  the NARDINI+ notebook, in which histidine is grouped with the polar residues.
* :data:`TYPES_9` -- the 9-type scheme (``TYPEALL_9x9`` in the reference
  ``nardini`` package) in which histidine is broken out as its own type.

Both are exact partitions of the 20 standard amino acids: every residue belongs
to exactly one type. That property is what lets the patterning calculations work
on a compact integer label array rather than on strings (see
:mod:`sparrow.nardini.patterning`), and it is asserted at import time.
"""

from collections import OrderedDict

from sparrow.data.amino_acids import VALID_AMINO_ACIDS

__all__ = [
    "TYPES_8",
    "TYPES_9",
    "DEFAULT_TYPESET",
    "get_typeset",
    "type_names",
    "patterning_feature_names",
]


#: The default NARDINI 8-type grouping. Histidine sits with the polar residues.
TYPES_8 = OrderedDict(
    [
        ("pol", tuple("STNQCH")),
        ("hyd", tuple("ILMV")),
        ("pos", tuple("RK")),
        ("neg", tuple("ED")),
        ("aro", tuple("FWY")),
        ("ala", ("A",)),
        ("pro", ("P",)),
        ("gly", ("G",)),
    ]
)

#: The 9-type grouping, with histidine broken out as its own type.
TYPES_9 = OrderedDict(
    [
        ("pol", tuple("STNQC")),
        ("hyd", tuple("ILMV")),
        ("pos", tuple("RK")),
        ("neg", tuple("ED")),
        ("aro", tuple("FWY")),
        ("ala", ("A",)),
        ("pro", ("P",)),
        ("gly", ("G",)),
        ("his", ("H",)),
    ]
)

#: Name of the grouping used unless another is requested.
DEFAULT_TYPESET = "8"

_TYPESETS = {"8": TYPES_8, "9": TYPES_9}


def _validate_partition(typeset, label):
    """Assert that a grouping is an exact partition of the 20 standard residues."""
    seen = []
    for residues in typeset.values():
        seen.extend(residues)

    duplicates = {r for r in seen if seen.count(r) > 1}
    if duplicates:
        raise ValueError(f"{label} assigns these residues to more than one type: {sorted(duplicates)}")

    missing = set(VALID_AMINO_ACIDS) - set(seen)
    if missing:
        raise ValueError(f"{label} does not cover these residues: {sorted(missing)}")

    extra = set(seen) - set(VALID_AMINO_ACIDS)
    if extra:
        raise ValueError(f"{label} contains non-standard residues: {sorted(extra)}")


_validate_partition(TYPES_8, "TYPES_8")
_validate_partition(TYPES_9, "TYPES_9")


def get_typeset(typeset=DEFAULT_TYPESET):
    """Return a NARDINI residue grouping by name.

    Parameters
    ----------
    typeset : {'8', '9'} or dict, optional
        Which grouping to use. ``'8'`` (default) is the standard NARDINI scheme;
        ``'9'`` breaks histidine out as its own type. A custom mapping of
        ``{type_name: residues}`` may also be passed, in which case it is
        validated as an exact partition of the 20 standard amino acids and
        returned as an ``OrderedDict``.

    Returns
    -------
    collections.OrderedDict
        Ordered mapping from type name to a tuple of one-letter residue codes.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``typeset`` names an unknown grouping.
    ValueError
        If a custom grouping is not an exact partition of the standard residues.

    Examples
    --------
    >>> list(get_typeset('8'))
    ['pol', 'hyd', 'pos', 'neg', 'aro', 'ala', 'pro', 'gly']
    """
    # local import to keep this module importable without the wider package
    from sparrow.sparrow_exceptions import SparrowException

    if isinstance(typeset, dict):
        custom = OrderedDict((name, tuple(residues)) for name, residues in typeset.items())
        _validate_partition(custom, "custom typeset")
        return custom

    key = str(typeset)
    if key not in _TYPESETS:
        raise SparrowException(
            f"Unknown NARDINI typeset '{typeset}'. Must be one of "
            f"{sorted(_TYPESETS)}, or a dict defining a custom partition."
        )
    return _TYPESETS[key]


def type_names(typeset=DEFAULT_TYPESET):
    """Return the ordered type names for a grouping.

    Parameters
    ----------
    typeset : {'8', '9'} or dict, optional
        Grouping selector, see :func:`get_typeset`.

    Returns
    -------
    tuple of str
        Type names in their canonical order.
    """
    return tuple(get_typeset(typeset).keys())


def patterning_feature_names(typeset=DEFAULT_TYPESET):
    """Return the ordered NARDINI patterning feature names.

    Features are the upper triangle (including the diagonal) of the
    type-by-type matrix, read row-major, and named ``"<type1>-<type2>"``. For the
    default 8-type grouping this gives 36 features, beginning ``'pol-pol'``,
    ``'pol-hyd'``, ... and ending ``'gly-gly'``.

    Diagonal entries (``'pol-pol'``) report the *omega*-style asymmetry of one
    type against everything else; off-diagonal entries (``'pol-hyd'``) report the
    *kappa*-style asymmetry between two types.

    Parameters
    ----------
    typeset : {'8', '9'} or dict, optional
        Grouping selector, see :func:`get_typeset`.

    Returns
    -------
    tuple of str
        Feature names in canonical order.

    Examples
    --------
    >>> names = patterning_feature_names('8')
    >>> len(names), names[0], names[-1]
    (36, 'pol-pol', 'gly-gly')
    """
    names = type_names(typeset)
    return tuple(f"{names[i]}-{names[j]}" for i in range(len(names)) for j in range(i, len(names)))
