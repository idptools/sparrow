"""How each of sparrow's analyses is drawn, and the tracks made from what it returns.

What an analysis returns decides how it is drawn: a value for each residue is
a line, a 1 or 0 for each residue is a box over each run of 1s (or a site for
each 1), and a list of regions is a box over each.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from sparrow.batching import _Request
from sparrow.sparrow_exceptions import SparrowException
from sparrow.visualize.protein_visualizer._inputs import _domain_rows, _site_track
from sparrow.visualize.proteinvis import LinearTrack, Track

if TYPE_CHECKING:
    from sparrow.protein import Protein


@dataclass(frozen=True)
class _Look:
    """How one of sparrow's analyses is drawn.

    Attributes
    ----------
    label : str
        The track's label, unless the analysis's arguments name it (see
        :func:`_label`).
    y_limits : tuple of float or None
        For a value per residue with a fixed range, where the y-axis runs;
        None for the range of the values.
    ones_are_sites : bool
        Whether a 1 for a residue marks a single site, rather than part of a
        region.
    """

    label: str
    y_limits: tuple[float, float] | None = None
    ones_are_sites: bool = False


# sparrow's analyses that give something for each residue, and how each is
# drawn. Any other analysis that gives a value for each residue is drawn too,
# labelled with its name
_PROBABILITY = (0.0, 1.0)
_PLDDT_RANGE = (0.0, 100.0)
_LOOKS: dict[str, _Look] = {
    "predictor.disorder": _Look("Disorder", _PROBABILITY),
    "predictor.binary_disorder": _Look("IDRs"),
    "predictor.disorder_domains": _Look("IDRs"),
    "predictor.pLDDT": _Look("pLDDT", _PLDDT_RANGE),
    "predictor.dssp_helicity": _Look("Helix", _PROBABILITY),
    "predictor.dssp_extended": _Look("Strand", _PROBABILITY),
    "predictor.dssp_coil": _Look("Coil", _PROBABILITY),
    "predictor.transmembrane_regions": _Look("TM"),
    "predictor.mitochondrial_targeting_sequence": _Look("MTS"),
    "predictor.nuclear_export_signal": _Look("NES", _PROBABILITY),
    "predictor.nuclear_import_signal": _Look("NLS", _PROBABILITY),
    "predictor.transactivation_domains": _Look("TAD", _PROBABILITY),
    "predictor.pscore": _Look("PScore"),
    "predictor.serine_phosphorylation": _Look("pS", _PROBABILITY, ones_are_sites=True),
    "predictor.threonine_phosphorylation": _Look("pT", _PROBABILITY, ones_are_sites=True),
    "predictor.tyrosine_phosphorylation": _Look("pY", _PROBABILITY, ones_are_sites=True),
    "linear_sequence_profile": _Look("Profile"),
    "linear_composition_profile": _Look("Fraction"),
    "linear_property_profile": _Look("Property"),
    "low_complexity_mask": _Look("LCDs"),
    "low_complexity_domains": _Look("LCDs"),
    "plaac_prion_like_domains": _Look("PLDs"),
}

# The label of each of linear_sequence_profile's modes; most profile a fraction
# of the residues in each window
_PROFILE_LABELS = {
    "FCR": "FCR",
    "NCPR": "NCPR",
    "aromatic": "Fraction aromatic",
    "aliphatic": "Fraction aliphatic",
    "polar": "Fraction polar",
    "proline": "Fraction proline",
    "positive": "Fraction positive",
    "negative": "Fraction negative",
    "hydrophobicity": "Hydrophobicity",
    "seg-complexity": "SEG complexity",
    "kappa": "Kappa",
}

# The analyses that return regions as [subsequence, start, end], start
# counting from 0 and end left out
_REGION_LISTS = ("low_complexity_domains", "plaac_prion_like_domains")


def _analysis_tracks(protein: Protein, request: _Request) -> list[Track]:
    """Compute one of a protein's analyses and make the track that draws it.

    Parameters
    ----------
    protein : sparrow.protein.Protein
        The protein, which computes the analysis and caches it.
    request : _Request
        The analysis and its arguments, checked.

    Returns
    -------
    list of Track
        A line for a value per residue, boxes for regions, or marks for
        sites: one track, unless regions overlap, when they take as many
        rows as they need.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the analysis does not give something for each residue.
    """
    owner = protein if request.accessor is None else getattr(protein, request.accessor)
    result = getattr(owner, request.attribute)
    if not request.is_property:
        result = result(**request.arguments)

    label = _label(request)
    look = _LOOKS.get(request.name, _Look(request.name))
    arguments = request.every_argument

    # the arguments that make a method return positions or regions in
    # place of something for each residue, all counting from 0
    if arguments.get("return_sites_only"):
        return [_site_track(label, [int(position) + 1 for position in result])]
    if arguments.get("return_boundaries"):
        return list(_domain_rows(label, [(int(start) + 1, int(end)) for start, end in result]))
    if request.name in _REGION_LISTS:
        if not arguments.get("simple", True):
            raise SparrowException(
                f"{request.name} with simple=False returns PLAAC's full result rather than regions, so it "
                "cannot be drawn; leave simple as True."
            )
        return list(_domain_rows(label, [(int(start) + 1, int(end)) for _, start, end in result]))

    values = _per_residue(request, result, len(protein))
    if not _is_mask(values):
        return [LinearTrack(label, values, y_limits=look.y_limits)]
    if look.ones_are_sites:
        return [_site_track(label, [int(position) + 1 for position in np.flatnonzero(values)])]
    return list(_domain_rows(label, _runs_of_ones(values)))


def _label(request: _Request) -> str:
    """The label of an analysis's track.

    Parameters
    ----------
    request : _Request
        The analysis and its arguments.

    Returns
    -------
    str
        For a linear profile, what it profiles, such as ``'NCPR'`` or
        ``'Fraction N+Q'``; for a low-complexity analysis, with the residues
        it looks for; otherwise the label sparrow gives it, or its name.
    """
    arguments = request.every_argument
    if request.name == "linear_sequence_profile":
        return _PROFILE_LABELS.get(arguments["mode"], str(arguments["mode"]))
    if request.name == "linear_property_profile":
        return str(arguments["mode"])
    if request.name == "linear_composition_profile":
        residues = "+".join(sorted(str(residue) for residue in arguments["composition_list"]))
        return f"{_LOOKS[request.name].label} {residues}"

    label = _LOOKS[request.name].label if request.name in _LOOKS else request.name
    if request.name in ("low_complexity_mask", "low_complexity_domains"):
        residues = arguments.get("kwargs", {}).get("residue_selector")
        if residues:
            label = f"{label} ({residues})"
    return label


def _per_residue(request: _Request, result: object, length: int) -> npt.NDArray[np.generic]:
    """Check that an analysis gave one number for each residue.

    Parameters
    ----------
    request : _Request
        The analysis, for the error message.
    result : object
        What it returned.
    length : int
        The number of residues.

    Returns
    -------
    numpy.ndarray
        The numbers, as returned, in a 1-D array.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the analysis gave one value for the whole protein, something other
        than numbers, or a number of values other than ``length``.
    """
    called = request.name if not request.arguments else f"{request.name} with {request.arguments}"
    if isinstance(result, numbers.Number):
        raise SparrowException(
            f"{called} is one value for the whole protein, so it cannot be drawn along it. "
            "ProteinVisualizer.available_analyses() lists those that can."
        )

    values = np.asarray(result) if isinstance(result, (list, np.ndarray)) else None
    if values is None or values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
        what = f"a {type(result).__name__}" if values is None else f"an array of shape {values.shape}"
        raise SparrowException(
            f"{called} returns {what}, not one value for each residue, so it cannot be drawn along the protein. "
            "Draw regions of your own with domains=, and sites with sites=."
        )
    if len(values) != length:
        raise SparrowException(
            f"{called} returns {len(values)} values for a protein of {length} residues, so they cannot be drawn "
            "one for each residue. For a linear profile, end_mode='extend-ends' gives one for each residue."
        )
    return values


def _is_mask(values: npt.NDArray[np.generic]) -> bool:
    """Whether per-residue values are a 1 or 0 for each residue, as whole numbers.

    Parameters
    ----------
    values : numpy.ndarray
        The values.

    Returns
    -------
    bool
        True for whole numbers or booleans that are all 0 or 1. Fractions that
        happen to be 0 or 1 are values, not a mask.
    """
    whole = np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_)
    return bool(whole and np.isin(values, (0, 1)).all())


def _runs_of_ones(mask: npt.NDArray[np.generic]) -> list[tuple[int, int]]:
    """The runs of 1s in a mask, as regions.

    Parameters
    ----------
    mask : numpy.ndarray
        A 1 or 0 for each residue.

    Returns
    -------
    list of tuple of int
        ``(start, end)`` of each run, counting from 1, both ends in the run.
    """
    # a 0 either side, so every run has a rise and a fall
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    changes = np.flatnonzero(np.diff(padded))
    rises, falls = changes[0::2], changes[1::2]
    # a rise at index i of the differences is residue i + 1, counting from 1,
    # the first of a run, and a fall at index j is just after residue j, the last
    return [(int(rise) + 1, int(fall)) for rise, fall in zip(rises, falls)]
