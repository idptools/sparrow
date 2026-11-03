"""The features drawn on a protein: a domain, a site, a region whose sequence is written, and a highlight."""

from __future__ import annotations

import numbers
from dataclasses import dataclass

from sparrow.sparrow_exceptions import SparrowException
from sparrow.visualize.proteinvis._checks import _check_color, _check_position, _check_positive, _check_region
from sparrow.visualize.proteinvis._style import (
    DEFAULT_DOMAIN_COLOR,
    DEFAULT_DOMAIN_HEIGHT,
    DEFAULT_HIGHLIGHT_ALPHA,
    DEFAULT_HIGHLIGHT_COLOR,
    DEFAULT_SITE_COLOR,
    INK_COLOR,
    LABEL_POSITIONS,
    LabelPosition,
)


@dataclass(frozen=True)
class Domain:
    """One domain of a :class:`DomainTrack`: a box over residues ``start`` to ``end``.

    Make one with :meth:`DomainTrack.add_domain`.

    Parameters
    ----------
    start, end : int
        The first and last residue of the domain, counting from 1. Both are
        in the domain.
    label : str, optional
        Text written with the domain. Default ``''``, none.
    color : str, optional
        The fill of the box, any color matplotlib can draw. Default white.
    height : float, optional
        The height of the box, as a fraction of its track's height, greater
        than 0 and at most 1. Default 0.6.
    label_position : {'inside', 'above', 'below'}, optional
        Where the label is written: in the box (the default), or centered
        above or below it.
    label_color : str, optional
        The color of the label. Default black.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If any of these is not as described.
    """

    start: int
    end: int
    label: str = ""
    color: str = DEFAULT_DOMAIN_COLOR
    height: float = DEFAULT_DOMAIN_HEIGHT
    label_position: LabelPosition = "inside"
    label_color: str = INK_COLOR

    def __post_init__(self) -> None:
        _check_region(self.start, self.end, "a domain")
        if not isinstance(self.label, str):
            raise SparrowException(f"A domain's label must be a str, not {self.label!r}.")
        _check_color(self.color, "A domain's color")
        _check_color(self.label_color, "A domain's label_color")
        if _check_positive(self.height, "A domain's height") > 1:
            raise SparrowException(
                f"A domain's height is a fraction of its track's height, so it must be at most 1, not {self.height!r}."
            )
        if self.label_position not in LABEL_POSITIONS:
            raise SparrowException(
                f"A domain's label_position must be one of {LABEL_POSITIONS}, not {self.label_position!r}."
            )


@dataclass(frozen=True)
class Site:
    """One site of a :class:`SiteTrack`: a single residue, marked by a stem and a dot.

    Make one with :meth:`SiteTrack.add_site`.

    Parameters
    ----------
    position : int
        The residue, counting from 1.
    label : str, optional
        Text written above the marker. Default ``''``, none.
    color : str, optional
        The color of the stem and marker, any color matplotlib can draw.
        Default black.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If any of these is not as described.
    """

    position: int
    label: str = ""
    color: str = DEFAULT_SITE_COLOR

    def __post_init__(self) -> None:
        _check_position(self.position, "A site's position")
        if not isinstance(self.label, str):
            raise SparrowException(f"A site's label must be a str, not {self.label!r}.")
        _check_color(self.color, "A site's color")


@dataclass(frozen=True)
class SequenceRegion:
    """One region of a :class:`SequenceTrack`, residues ``start`` to ``end``, whose sequence is written.

    Make one with :meth:`SequenceTrack.add_region`.

    Parameters
    ----------
    start, end : int
        The first and last residue of the region, counting from 1. Both are
        in the region.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If either is not a residue position, or ``end`` is before ``start``.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        _check_region(self.start, self.end, "a sequence region")


@dataclass(frozen=True)
class Highlight:
    """A band of color behind every track of a protein, over residues ``start`` to ``end``.

    Make one with :meth:`VisualProtein.add_highlight`.

    Parameters
    ----------
    start, end : int
        The first and last residue the band covers, counting from 1.
    color : str, optional
        The band's color, any color matplotlib can draw. Default red.
    alpha : float, optional
        The band's opacity, from 0 (invisible) to 1 (solid). Default 0.15.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If any of these is not as described.
    """

    start: int
    end: int
    color: str = DEFAULT_HIGHLIGHT_COLOR
    alpha: float = DEFAULT_HIGHLIGHT_ALPHA

    def __post_init__(self) -> None:
        _check_region(self.start, self.end, "a highlight")
        _check_color(self.color, "A highlight's color")
        alpha = self.alpha
        if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real) or not 0 <= alpha <= 1:
            raise SparrowException(f"A highlight's alpha must be a number from 0 to 1, not {alpha!r}.")
