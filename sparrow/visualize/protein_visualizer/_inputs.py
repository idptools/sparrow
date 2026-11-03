"""Reading what ``Protein.visualize`` is given, and making the tracks it describes.

This reads the order of the kinds of track, and the signals, sites, domains,
sequence and highlights given by hand, each checked as it is read.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union, cast

import numpy as np

from sparrow.sparrow_exceptions import SparrowException
from sparrow.visualize.proteinvis import Domain, DomainTrack, Highlight, SequenceTrack, Site, SiteTrack

if TYPE_CHECKING:
    from sparrow.protein import Protein

#: The kinds of track a figure is built from, in the order they are drawn, top
#: to bottom, unless ``order`` says otherwise
GROUPS = ("domains", "sites", "analyses", "signals", "tracks", "sequence")

# What sequence= and highlights= take to mean each predicted IDR
_IDRS = "idrs"

# A domain given as a tuple is (start, end) or (start, end, label)
_DOMAIN_TUPLE_LENGTHS = (2, 3)

# What each argument takes. The positions in a tuple are checked when its
# domain, region or highlight is made, so a tuple's contents are typed Any
SitesInput = Union[Mapping[int, str], Iterable[Union[int, Site]]]
DomainsInput = Iterable[Union[Domain, Sequence[Any]]]
RegionsInput = Union[Literal["idrs"], Iterable[Sequence[Any]]]
SequenceInput = Union[bool, Literal["idrs"], Iterable[Sequence[Any]], None]
HighlightsInput = Union[Literal["idrs"], Iterable[Union[Highlight, Sequence[Any]]], None]

_Contents = TypeVar("_Contents")


def _checked_order(order: Sequence[str], given: Sequence[str]) -> list[str]:
    """Check the order the kinds of track are drawn in, and return the kinds given, in that order.

    Parameters
    ----------
    order : sequence of str
        The order asked for.
    given : sequence of str
        The kinds that were given.

    Returns
    -------
    list of str
        The kinds given, in ``order``.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``order`` names a kind that does not exist, names one twice, or
        leaves out one that was given.
    """
    if isinstance(order, str) or len(set(order)) != len(order) or not set(order) <= set(GROUPS):
        raise SparrowException(f"order must name each of {GROUPS} at most once, not {order!r}.")
    left_out = [group for group in given if group not in order]
    if left_out:
        raise SparrowException(f"{left_out} were given but order {tuple(order)} leaves them out.")
    return [group for group in order if group in given]


def _checked_mapping(value: Mapping[str, _Contents], group: str) -> Mapping[str, _Contents]:
    """Check that what was given for one kind of track is a dict of labels and their contents.

    Parameters
    ----------
    value : dict
        What was given.
    group : str
        The kind, for the error message.

    Returns
    -------
    Mapping
        ``value``.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``value`` is not a dict whose keys are strings.
    """
    if not isinstance(value, Mapping) or not all(isinstance(label, str) for label in value):
        raise SparrowException(
            f"{group} must be a dict of each track's label and what it draws, such as {{'Label': ...}}, "
            f"not a {type(value).__name__}."
        )
    return value


def _is_tuple_like(item: object) -> bool:
    """Whether a region or domain was given as a tuple, a list or a row of a numpy array.

    Parameters
    ----------
    item : object
        The region or domain.

    Returns
    -------
    bool
        True for a tuple, list or 1-D array; False for a string, which is a
        sequence of characters rather than of positions.
    """
    if isinstance(item, np.ndarray):
        return item.ndim == 1
    return isinstance(item, Sequence) and not isinstance(item, str)


def _region(item: Sequence[Any], what: str) -> tuple[int, int]:
    """Check that a region is a ``(start, end)`` pair.

    Parameters
    ----------
    item : sequence
        The region.
    what : str
        The argument it was given in, for the error message.

    Returns
    -------
    tuple of int
        ``(start, end)``. Whether these are residues of the protein is checked
        when the region is added to its track.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``item`` is not a pair.
    """
    if not _is_tuple_like(item) or len(item) != 2:
        raise SparrowException(f"Each region of {what} must be (start, end), not {item!r}.")
    start, end = item
    return start, end


def _site_track(label: str, sites: SitesInput) -> SiteTrack:
    """Make a site track of the sites given.

    Parameters
    ----------
    label : str
        The track's label.
    sites : dict or iterable
        A dict of positions and their labels, or positions and
        :class:`~sparrow.visualize.proteinvis.Site` in any mix.

    Returns
    -------
    SiteTrack

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``sites`` is neither, or a site is not valid.
    """
    track = SiteTrack(label)
    if isinstance(sites, Mapping):
        # a Mapping is Iterable too, so only the dict form of SitesInput gets here
        for position, site_label in cast(Mapping[int, str], sites).items():
            track.add_site(position, site_label)
        return track
    if isinstance(sites, str) or not isinstance(sites, Iterable):
        raise SparrowException(
            f"The sites of {label!r} must be a list of positions, or a dict of positions and their labels, not "
            f"{sites!r}."
        )
    for item in sites:
        if isinstance(item, Site):
            track.add_site(item.position, item.label, item.color)
        else:
            track.add_site(item)
    return track


def _as_domain(item: Union[Domain, Sequence[Any]], label: str) -> Domain:
    """A domain given as ``(start, end)``, ``(start, end, label)`` or a Domain.

    Parameters
    ----------
    item : Domain or sequence
        The domain.
    label : str
        The label of its track, for the error message.

    Returns
    -------
    Domain

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``item`` is none of these, or not a valid domain.
    """
    if isinstance(item, Domain):
        return item
    if not _is_tuple_like(item) or len(item) not in _DOMAIN_TUPLE_LENGTHS:
        raise SparrowException(
            f"Each domain of {label!r} must be (start, end), (start, end, label) or a Domain, not {item!r}."
        )
    return Domain(*item)


def _domain_rows(label: str, domains: DomainsInput) -> list[DomainTrack]:
    """Lay domains out on as few domain tracks as they fit on without overlapping.

    Each domain goes on the first row whose domains all end before it starts,
    taking the domains in order of where they start, which needs as few rows
    as any arrangement can.

    Parameters
    ----------
    label : str
        The label of the first row; the rest have none.
    domains : iterable
        The domains, each as :func:`_as_domain` takes it.

    Returns
    -------
    list of DomainTrack
        At least one row, empty if there are no domains.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``domains`` is not a list of domains.
    """
    if isinstance(domains, (str, Mapping)) or not isinstance(domains, Iterable):
        raise SparrowException(f"The domains of {label!r} must be a list of domains, not {domains!r}.")

    rows = [DomainTrack(label)]
    checked = [_as_domain(item, label) for item in domains]
    for domain in sorted(checked, key=lambda domain: (domain.start, domain.end)):
        row = next((row for row in rows if not row.domains or row.domains[-1].end < domain.start), None)
        if row is None:
            row = DomainTrack()
            rows.append(row)
        row.add_domain(**asdict(domain))
    return rows


def _sequence_track(protein: Protein, sequence: Union[Literal[True], RegionsInput]) -> SequenceTrack:
    """Make the track that writes the sequence of the regions asked for.

    Parameters
    ----------
    protein : sparrow.protein.Protein
        The protein.
    sequence : True, 'idrs' or list
        The whole protein, each predicted IDR, or ``(start, end)`` regions.

    Returns
    -------
    SequenceTrack
    """
    track = SequenceTrack()
    if sequence is True:
        track.add_region(1, len(protein))
        return track
    for start, end in _regions(protein, sequence, "sequence"):
        track.add_region(start, end)
    return track


def _highlights(protein: Protein, highlights: HighlightsInput) -> list[Highlight]:
    """The highlights asked for.

    Parameters
    ----------
    protein : sparrow.protein.Protein
        The protein.
    highlights : 'idrs', list or None
        As for :meth:`ProteinVisualizer.visual_protein`.

    Returns
    -------
    list of Highlight
    """
    if highlights is None:
        return []
    if isinstance(highlights, str):
        return [Highlight(start, end) for start, end in _regions(protein, highlights, "highlights")]
    checked = []
    for item in highlights:
        if isinstance(item, Highlight):
            checked.append(item)
        else:
            start, end = _region(item, "highlights")
            checked.append(Highlight(start, end))
    return checked


def _regions(protein: Protein, regions: RegionsInput, what: str) -> list[tuple[int, int]]:
    """Regions given as a list of ``(start, end)``, or as ``'idrs'`` for each predicted IDR.

    Parameters
    ----------
    protein : sparrow.protein.Protein
        The protein.
    regions : 'idrs' or list
        The regions.
    what : str
        The argument they were given as, for the error message.

    Returns
    -------
    list of tuple of int
        ``(start, end)`` of each region, counting from 1, both ends in the region.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``regions`` is a string other than ``'idrs'``, or a region is not
        a ``(start, end)`` pair.
    """
    if isinstance(regions, str):
        if regions != _IDRS:
            raise SparrowException(f"{what} can be {_IDRS!r} or a list of (start, end) regions, not {regions!r}.")
        # metapredict's IDRs count from 0 and leave out their end
        boundaries = protein.predictor.disorder_domains(return_boundaries=True)
        return [(int(start) + 1, int(end)) for start, end in boundaries]
    return [_region(item, what) for item in regions]
