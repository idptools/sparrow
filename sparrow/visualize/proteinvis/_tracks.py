"""The tracks a protein is drawn as: domains, sites, a value for each residue, a heatmap, and the sequence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Union

import matplotlib
import numpy as np
import numpy.typing as npt

from sparrow.data.amino_acids import AA_COLOR
from sparrow.sparrow_exceptions import SparrowException
from sparrow.visualize.proteinvis._checks import (
    _check_color,
    _check_count,
    _check_limits,
    _check_positive,
    _first_overlap,
    _per_residue_values,
)
from sparrow.visualize.proteinvis._features import Domain, SequenceRegion, Site
from sparrow.visualize.proteinvis._style import (
    DEFAULT_CHARACTERS_PER_BLOCK,
    DEFAULT_COLORMAP,
    DEFAULT_DOMAIN_COLOR,
    DEFAULT_DOMAIN_HEIGHT,
    DEFAULT_DOMAIN_TRACK_HEIGHT_IN,
    DEFAULT_GRID_ROW_HEIGHT_IN,
    DEFAULT_LINE_WIDTH_PT,
    DEFAULT_LINEAR_TRACK_HEIGHT_IN,
    DEFAULT_SEQUENCE_FONT_SIZE_PT,
    DEFAULT_SITE_COLOR,
    DEFAULT_SITE_MARKER_SIZE_PT,
    DEFAULT_SITE_TRACK_HEIGHT_IN,
    INK_COLOR,
    PLOT_TYPES,
    LabelPosition,
    PlotType,
)

if TYPE_CHECKING:
    from sparrow.visualize.proteinvis._visual_protein import VisualProtein


class DomainTrack:
    """A track of domains: boxes on a line that stands for the chain.

    Domains in one track cannot overlap. Draw overlapping domains, such as
    IDRs and the motifs within them, on tracks of their own.

    Parameters
    ----------
    name : str, optional
        The track's label, written to its left. Default ``''``, none.
    height_in : float, optional
        The track's height, in inches. Default 0.3.
    show_backbone : bool, optional
        Whether to draw the line that stands for the chain, from residue 1 to
        the last. Default True.
    label_font_size_pt : float or None, optional
        The size of the domains' labels, in points. Default None, the size of
        every other label in the figure.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If any of these is not as described.
    """

    def __init__(
        self,
        name: str = "",
        height_in: float = DEFAULT_DOMAIN_TRACK_HEIGHT_IN,
        show_backbone: bool = True,
        label_font_size_pt: float | None = None,
    ) -> None:
        if not isinstance(name, str):
            raise SparrowException(f"A track's name must be a str, not {name!r}.")
        self.name = name
        self.height_in = _check_positive(height_in, "A domain track's height_in")
        self.show_backbone = bool(show_backbone)
        self.label_font_size_pt = (
            None if label_font_size_pt is None else _check_positive(label_font_size_pt, "label_font_size_pt")
        )
        self.domains: list[Domain] = []

    def add_domain(
        self,
        start: int,
        end: int,
        label: str = "",
        color: str = DEFAULT_DOMAIN_COLOR,
        height: float = DEFAULT_DOMAIN_HEIGHT,
        label_position: LabelPosition = "inside",
        label_color: str = INK_COLOR,
    ) -> Domain:
        """Add a domain over residues ``start`` to ``end``.

        Parameters
        ----------
        start, end : int
            The first and last residue of the domain, counting from 1. Both
            are in the domain.
        label : str, optional
            Text written with the domain. Default ``''``, none.
        color : str, optional
            The fill of the box, any color matplotlib can draw. Default white.
        height : float, optional
            The height of the box, as a fraction of the track's height,
            greater than 0 and at most 1. Default 0.6.
        label_position : {'inside', 'above', 'below'}, optional
            Where the label is written: in the box (the default), or centered
            above or below it.
        label_color : str, optional
            The color of the label. Default black.

        Returns
        -------
        Domain
            The domain added.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If any of these is not as described, or the domain overlaps one
            already in the track.
        """
        domain = Domain(start, end, label, color, height, label_position, label_color)
        clash = _first_overlap(domain.start, domain.end, [(other.start, other.end) for other in self.domains])
        if clash is not None:
            raise SparrowException(
                f"The domain from {domain.start} to {domain.end} overlaps the one from {clash[0]} to {clash[1]} "
                "already in this track. Put overlapping domains on tracks of their own."
            )
        self.domains.append(domain)
        return domain

    def validate(self, protein: VisualProtein) -> None:
        """Check that every domain lies within ``protein``.

        Parameters
        ----------
        protein : VisualProtein
            The protein the track is drawn for.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If a domain ends after the protein's last residue.
        """
        for domain in self.domains:
            if domain.end > protein.length:
                raise SparrowException(
                    f"The domain from {domain.start} to {domain.end} in track {self.name!r} ends after the last "
                    f"residue of {protein.name or 'the protein'} ({protein.length})."
                )


class SiteTrack:
    """A track of single residues, such as phosphosites or mutations, each a stem and a dot above the chain.

    Parameters
    ----------
    name : str, optional
        The track's label, written to its left. Default ``''``, none.
    height_in : float, optional
        The track's height, in inches. Default 0.3.
    show_backbone : bool, optional
        Whether to draw the line that stands for the chain, from residue 1 to
        the last. Default True.
    marker_size_pt : float, optional
        The diameter of each site's marker, in points. Default 3.
    label_font_size_pt : float or None, optional
        The size of the sites' labels, in points. Default None, the size of
        every other label in the figure.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If any of these is not as described.
    """

    def __init__(
        self,
        name: str = "",
        height_in: float = DEFAULT_SITE_TRACK_HEIGHT_IN,
        show_backbone: bool = True,
        marker_size_pt: float = DEFAULT_SITE_MARKER_SIZE_PT,
        label_font_size_pt: float | None = None,
    ) -> None:
        if not isinstance(name, str):
            raise SparrowException(f"A track's name must be a str, not {name!r}.")
        self.name = name
        self.height_in = _check_positive(height_in, "A site track's height_in")
        self.show_backbone = bool(show_backbone)
        self.marker_size_pt = _check_positive(marker_size_pt, "A site track's marker_size_pt")
        self.label_font_size_pt = (
            None if label_font_size_pt is None else _check_positive(label_font_size_pt, "label_font_size_pt")
        )
        self.sites: list[Site] = []

    def add_site(self, position: int, label: str = "", color: str = DEFAULT_SITE_COLOR) -> Site:
        """Add a site at residue ``position``.

        Parameters
        ----------
        position : int
            The residue, counting from 1.
        label : str, optional
            Text written above the marker. Default ``''``, none.
        color : str, optional
            The color of the stem and marker. Default black.

        Returns
        -------
        Site
            The site added.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If any of these is not as described, or the track already has a
            site at ``position``.
        """
        site = Site(position, label, color)
        if any(other.position == site.position for other in self.sites):
            raise SparrowException(
                f"This track already has a site at residue {site.position}. Put a second site at the same residue "
                "on a track of its own."
            )
        self.sites.append(site)
        return site

    def validate(self, protein: VisualProtein) -> None:
        """Check that every site lies within ``protein``.

        Parameters
        ----------
        protein : VisualProtein
            The protein the track is drawn for.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If a site is after the protein's last residue.
        """
        for site in self.sites:
            if site.position > protein.length:
                raise SparrowException(
                    f"The site at residue {site.position} in track {self.name!r} is after the last residue of "
                    f"{protein.name or 'the protein'} ({protein.length})."
                )


class LinearTrack:
    """A track of one value per residue, drawn as a line or as bars, on a y-axis of its own.

    Parameters
    ----------
    name : str
        The track's label, written to the left of its y-axis. Give the units
        here, if the values have them.
    values : array_like of float
        One value for each residue of the protein, in order. A residue with no
        value can be NaN, and the line breaks there.
    plot_type : {'line', 'bar'}, optional
        Draw the values as a line (the default), or as one bar for each
        residue, from zero.
    color : str, optional
        The color of the line or bars. Default black.
    line_width_pt : float, optional
        The width of the line, in points. Default 0.75. Bars ignore it.
    y_limits : tuple of float or None, optional
        The ``(lower, upper)`` limits of the y-axis. Default None: the range
        of the values, rounded outwards, and for bars, widened to include zero.
    height_in : float, optional
        The track's height, in inches. Default 0.5.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If any of these is not as described.
    """

    def __init__(
        self,
        name: str,
        values: npt.ArrayLike,
        plot_type: PlotType = "line",
        color: str = INK_COLOR,
        line_width_pt: float = DEFAULT_LINE_WIDTH_PT,
        y_limits: tuple[float, float] | None = None,
        height_in: float = DEFAULT_LINEAR_TRACK_HEIGHT_IN,
    ) -> None:
        if not isinstance(name, str):
            raise SparrowException(f"A track's name must be a str, not {name!r}.")
        if plot_type not in PLOT_TYPES:
            raise SparrowException(f"A linear track's plot_type must be one of {PLOT_TYPES}, not {plot_type!r}.")
        _check_color(color, "A linear track's color")

        self.name = name
        self.values = _per_residue_values(values, f"The values of linear track {name!r}")
        if self.values.ndim != 1:
            raise SparrowException(
                f"The values of linear track {name!r} must be one value for each residue, not an array of shape "
                f"{self.values.shape}. Use a GridTrack for several values for each residue."
            )
        self.plot_type = plot_type
        self.color = color
        self.line_width_pt = _check_positive(line_width_pt, "A linear track's line_width_pt")
        self.y_limits = None if y_limits is None else _check_limits(y_limits, "A linear track's y_limits")
        self.height_in = _check_positive(height_in, "A linear track's height_in")

    def validate(self, protein: VisualProtein) -> None:
        """Check that there is one value for each residue of ``protein``.

        Parameters
        ----------
        protein : VisualProtein
            The protein the track is drawn for.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If the number of values is not the protein's length.
        """
        if len(self.values) != protein.length:
            raise SparrowException(
                f"Linear track {self.name!r} has {len(self.values)} values, but {protein.name or 'the protein'} has "
                f"{protein.length} residues: there must be one value for each."
            )


class GridTrack:
    """A heatmap of per-residue values, one row for each kind of value, each with its own colormap.

    Parameters
    ----------
    name : str
        The track's label, written to the left of its row names.
    values : array_like of float
        The values, as an array of shape ``(rows, residues)``: one row for each
        kind of value, top to bottom, holding one value for each residue. A
        list of per-residue profiles is such an array, and a single profile
        is one row. A residue with no value can be NaN, and is left blank.
    colormaps : str or sequence of str, optional
        The name of the colormap of every row, or of each row in turn, as
        matplotlib names them. Default ``'viridis'``.
    limits : sequence of (tuple of float or None), or None, optional
        For each row, the ``(lower, upper)`` values its colormap runs between,
        or None for the row's own range. Default None, every row's own range.
        Values outside the limits are drawn in the color of the nearer limit.
    row_names : sequence of str or None, optional
        The name of each row, written to its left. Default None, no names.
    row_height_in : float, optional
        The height of each row, in inches. Default 0.1.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If any of these is not as described, a colormap is not one matplotlib
        has, or a row with no limits given is all NaN.
    """

    def __init__(
        self,
        name: str,
        values: npt.ArrayLike,
        colormaps: str | Sequence[str] = DEFAULT_COLORMAP,
        limits: Sequence[tuple[float, float] | None] | None = None,
        row_names: Sequence[str] | None = None,
        row_height_in: float = DEFAULT_GRID_ROW_HEIGHT_IN,
    ) -> None:
        if not isinstance(name, str):
            raise SparrowException(f"A track's name must be a str, not {name!r}.")
        self.name = name

        array = _per_residue_values(values, f"The values of grid track {name!r}")
        if array.ndim == 1:
            array = array[np.newaxis, :]
        if array.ndim != 2:
            raise SparrowException(
                f"The values of grid track {name!r} must have the shape (rows, residues), not {array.shape}."
            )
        self.values = array
        n_rows = array.shape[0]

        self.colormaps = self._checked_colormaps(colormaps, n_rows)
        self.limits = self._checked_limits(limits, n_rows)
        self.row_names = self._checked_row_names(row_names, n_rows)
        self.row_height_in = _check_positive(row_height_in, "A grid track's row_height_in")

    @property
    def n_rows(self) -> int:
        """int : The number of rows."""
        return int(self.values.shape[0])

    @property
    def height_in(self) -> float:
        """float : The track's height in inches, which is the height of a row times the number of rows."""
        return self.row_height_in * self.n_rows

    def _checked_colormaps(self, colormaps: str | Sequence[str], n_rows: int) -> list[str]:
        """The colormap of each row, checked to be one matplotlib has.

        Parameters
        ----------
        colormaps : str or sequence of str
            One name for every row, or one for each row.
        n_rows : int
            The number of rows.

        Returns
        -------
        list of str
            The name of each row's colormap.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If there is not one name, or one for each row, or matplotlib has
            no colormap of a name.
        """
        names = [colormaps] * n_rows if isinstance(colormaps, str) else list(colormaps)
        if len(names) != n_rows:
            raise SparrowException(
                f"Grid track {self.name!r} has {n_rows} rows but {len(names)} colormaps: give one for every row, or "
                "one for each."
            )
        for colormap in names:
            if colormap not in matplotlib.colormaps:
                raise SparrowException(f"matplotlib has no colormap named {colormap!r} (grid track {self.name!r}).")
        return names

    def _checked_limits(
        self, limits: Sequence[tuple[float, float] | None] | None, n_rows: int
    ) -> list[tuple[float, float]]:
        """The values each row's colormap runs between, the row's own range where none are given.

        Parameters
        ----------
        limits : sequence of (tuple of float or None), or None
            For each row, ``(lower, upper)`` or None.
        n_rows : int
            The number of rows.

        Returns
        -------
        list of tuple of float
            The ``(lower, upper)`` of each row.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If there are limits but not one for each row, or limits that are
            not ``(lower, upper)``, or a row with no limits is all NaN.
        """
        given: list[tuple[float, float] | None] = [None] * n_rows if limits is None else list(limits)
        if len(given) != n_rows:
            raise SparrowException(
                f"Grid track {self.name!r} has {n_rows} rows but {len(given)} limits: give one for each row, None "
                "for a row's own range."
            )

        checked = []
        for row_index, row_limits in enumerate(given):
            if row_limits is not None:
                checked.append(_check_limits(row_limits, f"The limits of row {row_index} of grid track {self.name!r}"))
                continue
            row = self.values[row_index]
            if np.isnan(row).all():
                raise SparrowException(
                    f"Row {row_index} of grid track {self.name!r} is all NaN, so it has no range for its colormap. "
                    "Give its limits."
                )
            checked.append((float(np.nanmin(row)), float(np.nanmax(row))))
        return checked

    def _checked_row_names(self, row_names: Sequence[str] | None, n_rows: int) -> list[str]:
        """The name of each row, checked to be one str for each.

        Parameters
        ----------
        row_names : sequence of str or None
            The names, or None for none.
        n_rows : int
            The number of rows.

        Returns
        -------
        list of str
            The name of each row, ``''`` for each if none were given.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If there are names but not one str for each row.
        """
        if row_names is None:
            return [""] * n_rows
        names = list(row_names)
        if len(names) != n_rows or not all(isinstance(row_name, str) for row_name in names):
            raise SparrowException(
                f"Grid track {self.name!r} has {n_rows} rows, so it needs {n_rows} row names, each a str."
            )
        return names

    def validate(self, protein: VisualProtein) -> None:
        """Check that each row has one value for each residue of ``protein``.

        Parameters
        ----------
        protein : VisualProtein
            The protein the track is drawn for.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If the number of values in a row is not the protein's length.
        """
        n_residues = self.values.shape[1]
        if n_residues != protein.length:
            raise SparrowException(
                f"Each row of grid track {self.name!r} has {n_residues} values, but {protein.name or 'the protein'} "
                f"has {protein.length} residues: there must be one value for each. The values are (rows, residues), "
                "so an array of (residues, rows) needs transposing."
            )


class SequenceTrack:
    """A track of the sequence of chosen regions, each written within its region.

    Each region's sequence is written in blocks of ``chars_per_block``
    residues, and wraps onto as many lines as it needs to stay within the
    region, so it lies within a highlight over the same residues. Each line
    holds as many whole blocks as fit across the region; a region too narrow
    for one block holds as many residues on a line as fit. The lines are
    centered across the region as one column, each starting at the same
    place, so the blocks line up from one line to the next. The track is as
    tall as its region with the most lines, so narrow regions, or a long
    protein in a narrow figure, make it taller.

    The letters are colored by residue (:data:`~sparrow.data.amino_acids.AA_COLOR`),
    and are not aligned to their residues. A region narrower than one letter
    cannot hold its sequence, and :func:`plot_proteins` warns of it.

    Parameters
    ----------
    name : str, optional
        The track's label, written to its left. Default ``''``, none.
    font_size_pt : float, optional
        The size the sequence is written at, in points. Default 4.
    chars_per_block : int, optional
        The number of residues in each block. Default 10.
    blocks_per_line : int or None, optional
        The most blocks on each line; a region with room for fewer gets
        fewer. Default None, as many as fit across each region.
    colors : dict or None, optional
        Colors for residues, by one-letter code, in place of their colors in
        :data:`~sparrow.data.amino_acids.AA_COLOR`. A residue that has no
        color in either is black. Default None.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If any of these is not as described.
    """

    def __init__(
        self,
        name: str = "",
        font_size_pt: float = DEFAULT_SEQUENCE_FONT_SIZE_PT,
        chars_per_block: int = DEFAULT_CHARACTERS_PER_BLOCK,
        blocks_per_line: int | None = None,
        colors: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(name, str):
            raise SparrowException(f"A track's name must be a str, not {name!r}.")
        self.name = name
        self.font_size_pt = _check_positive(font_size_pt, "A sequence track's font_size_pt")
        self.chars_per_block = _check_count(chars_per_block, "A sequence track's chars_per_block")
        self.blocks_per_line = None if blocks_per_line is None else _check_count(blocks_per_line, "blocks_per_line")

        self.colors = dict(AA_COLOR)
        for residue, color in (colors or {}).items():
            _check_color(color, f"The color of residue {residue!r}")
            self.colors[residue] = color

        self.regions: list[SequenceRegion] = []

    def add_region(self, start: int, end: int) -> SequenceRegion:
        """Add a region whose sequence is written, residues ``start`` to ``end``.

        Parameters
        ----------
        start, end : int
            The first and last residue of the region, counting from 1. Both
            are in the region.

        Returns
        -------
        SequenceRegion
            The region added.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If either is not a residue position, ``end`` is before ``start``,
            or the region overlaps one already in the track.
        """
        region = SequenceRegion(start, end)
        clash = _first_overlap(region.start, region.end, [(other.start, other.end) for other in self.regions])
        if clash is not None:
            raise SparrowException(
                f"The region from {region.start} to {region.end} overlaps the one from {clash[0]} to {clash[1]} "
                "already in this track."
            )
        self.regions.append(region)
        return region

    def validate(self, protein: VisualProtein) -> None:
        """Check that ``protein`` has a sequence, and that every region lies within it.

        Parameters
        ----------
        protein : VisualProtein
            The protein the track is drawn for.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If the protein was made without a sequence, or a region ends after
            its last residue.
        """
        if protein.sequence is None:
            raise SparrowException(
                f"Sequence track {self.name!r} writes the sequence, but {protein.name or 'the protein'} was made "
                "without one: make the VisualProtein with sequence=..."
            )
        for region in self.regions:
            if region.end > protein.length:
                raise SparrowException(
                    f"The region from {region.start} to {region.end} in sequence track {self.name!r} ends after the "
                    f"last residue of {protein.name or 'the protein'} ({protein.length})."
                )


#: Any of the tracks a :class:`VisualProtein` can hold
Track = Union[DomainTrack, SiteTrack, LinearTrack, GridTrack, SequenceTrack]

_TRACK_TYPES = (DomainTrack, SiteTrack, LinearTrack, GridTrack, SequenceTrack)
