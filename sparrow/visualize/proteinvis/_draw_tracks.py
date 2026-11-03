"""Drawing each kind of track on axes of its own: domains, sites, a line or bars, and a heatmap.

The sequence of a sequence track is laid out and written by
``_sequence_layout``, since where it goes
depends on the width of a residue, which is known only once the figure is laid
out. A heatmap is drawn then too, for the same reason.
"""

from __future__ import annotations

import math

import matplotlib
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import nonsingular

from sparrow.visualize.proteinvis._features import Domain
from sparrow.visualize.proteinvis._style import (
    _POINTS_PER_INCH,
    BACKBONE_WIDTH_PT,
    DOMAIN_EDGE_WIDTH_PT,
    INK_COLOR,
    SITE_STEM_WIDTH_PT,
)
from sparrow.visualize.proteinvis._tracks import DomainTrack, GridTrack, LinearTrack, SiteTrack, Track
from sparrow.visualize.proteinvis._visual_protein import VisualProtein

# The space between a domain's box and a label above or below it, in points
_DOMAIN_LABEL_PAD_PT = 1.0

# Where a site track's chain line and its sites' markers sit, as fractions of
# the track's height from its bottom, leaving room above the markers for labels
_SITE_BACKBONE_Y = 0.2
_SITE_MARKER_Y = 0.55

# The space between a site's marker and its label, in points
_SITE_LABEL_PAD_PT = 1.0

# The number of intervals matplotlib may divide a linear track's values into
# when it rounds their range outwards to the track's limits
_Y_TICK_INTERVALS = 4

# The steps matplotlib may space ticks by, on either axis: 1, 2, 2.5 or 5 times a power of ten
_TICK_STEPS = (1, 2, 2.5, 5, 10)

# How far zero must be from both limits of a linear track, as a fraction of its
# range, to be given a tick of its own between them
_ZERO_TICK_CLEARANCE = 0.25

# How far a linear track whose values are all the same is widened either side
# of them, as a fraction of the value, so the track has a range to draw on
_FLAT_RANGE_EXPANSION = 0.05

# A heatmap is saved as an image, and Preview - like anything else that draws a
# PDF with Apple's PDFKit - smooths an image it enlarges, whatever the PDF asks,
# blurring each cell into its neighbours across one pixel of the image. Each
# cell is therefore a block of identical pixels, enough for the image to have at
# least this many pixels to an inch of the figure, so the blur reaches only
# 1/600 of an inch into each cell, which is sharp in Preview at 800%. Twice
# this is no sharper to the eye, and makes the PDF four times the size
_HEATMAP_PIXELS_PER_INCH = 600


def _hide_y_axis(ax: Axes) -> None:
    """Hide a track's y-axis, for a track whose height has no scale.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The track's axes.
    """
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)


def _draw_domain_label(ax: Axes, domain: Domain, font_size_pt: float) -> None:
    """Write a domain's label in, above or below its box.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The track's axes, whose y-axis runs from 0 to 1.
    domain : Domain
        The domain.
    font_size_pt : float
        The size of the label, in points.
    """
    middle = (domain.start + domain.end) / 2
    half_height = domain.height / 2
    if domain.label_position == "inside":
        anchor, offset_pt, vertical_alignment = 0.5, 0.0, "center"
    elif domain.label_position == "above":
        anchor, offset_pt, vertical_alignment = 0.5 + half_height, _DOMAIN_LABEL_PAD_PT, "bottom"
    else:
        anchor, offset_pt, vertical_alignment = 0.5 - half_height, -_DOMAIN_LABEL_PAD_PT, "top"

    ax.annotate(
        domain.label,
        xy=(middle, anchor),
        xytext=(0, offset_pt),
        textcoords="offset points",
        ha="center",
        va=vertical_alignment,
        color=domain.label_color,
        fontsize=font_size_pt,
        annotation_clip=False,
        zorder=3,
    )


def _draw_backbone(ax: Axes, protein: VisualProtein, y: float) -> None:
    """Draw the line that stands for the chain, from the first residue's left edge to the last one's right edge.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The track's axes, whose y-axis runs from 0 to 1.
    protein : VisualProtein
        The protein it is drawn for.
    y : float
        The height of the line, as a fraction of the track's height.
    """
    ax.plot(
        [0.5, protein.length + 0.5],
        [y, y],
        color=INK_COLOR,
        linewidth=BACKBONE_WIDTH_PT,
        solid_capstyle="butt",
        zorder=1,
    )


def _draw_site_track(ax: Axes, track: SiteTrack, protein: VisualProtein, font_size_pt: float) -> None:
    """Draw a site track: the line for the chain, and for each site a stem up to a marker, with its label above.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The track's axes.
    track : SiteTrack
        The track.
    protein : VisualProtein
        The protein it is drawn for.
    font_size_pt : float
        The size of every label in the figure, in points.
    """
    ax.set_ylim(0, 1)
    if track.show_backbone:
        _draw_backbone(ax, protein, _SITE_BACKBONE_Y)

    if track.sites:
        positions = [site.position for site in track.sites]
        colors = [site.color for site in track.sites]
        # not clipped, so a site at either end keeps its whole marker
        ax.vlines(
            positions,
            _SITE_BACKBONE_Y,
            _SITE_MARKER_Y,
            colors=colors,
            linewidth=SITE_STEM_WIDTH_PT,
            zorder=2,
            clip_on=False,
        )
        # a scatter marker's size is its area, in square points
        ax.scatter(
            positions,
            [_SITE_MARKER_Y] * len(positions),
            s=track.marker_size_pt**2,
            c=colors,
            linewidths=0,
            zorder=3,
            clip_on=False,
        )

    label_font_size_pt = track.label_font_size_pt or font_size_pt
    for site in track.sites:
        if site.label:
            ax.annotate(
                site.label,
                xy=(site.position, _SITE_MARKER_Y),
                xytext=(0, track.marker_size_pt / 2 + _SITE_LABEL_PAD_PT),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=label_font_size_pt,
                annotation_clip=False,
                zorder=3,
            )

    _hide_y_axis(ax)


def _draw_domain_track(ax: Axes, track: DomainTrack, protein: VisualProtein, font_size_pt: float) -> None:
    """Draw a domain track: the line for the chain, and a box for each domain.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The track's axes.
    track : DomainTrack
        The track.
    protein : VisualProtein
        The protein it is drawn for.
    font_size_pt : float
        The size of every label in the figure, in points.
    """
    ax.set_ylim(0, 1)
    if track.show_backbone:
        _draw_backbone(ax, protein, 0.5)

    label_font_size_pt = track.label_font_size_pt or font_size_pt
    for domain in track.domains:
        box = Rectangle(
            (domain.start - 0.5, 0.5 - domain.height / 2),
            domain.end - domain.start + 1,
            domain.height,
            facecolor=domain.color,
            edgecolor=INK_COLOR,
            linewidth=DOMAIN_EDGE_WIDTH_PT,
            zorder=2,
        )
        ax.add_patch(box)
        if domain.label:
            _draw_domain_label(ax, domain, label_font_size_pt)

    _hide_y_axis(ax)


def _rounded_limits(values: npt.NDArray[np.float64], include_zero: bool) -> tuple[float, float]:
    """The range of a linear track's values, rounded outwards to round numbers.

    Parameters
    ----------
    values : numpy.ndarray
        The values, finite or NaN.
    include_zero : bool
        Whether to widen the range to include zero, as bars drawn from zero need.

    Returns
    -------
    tuple of float
        ``(lower, upper)``.
    """
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
    # values that are all the same have no range, so widen it a little either side
    lower, upper = nonsingular(lower, upper, expander=_FLAT_RANGE_EXPANSION)
    # the ticks matplotlib would choose run from at or below the lowest value
    # to at or above the highest, at round numbers
    ticks = MaxNLocator(nbins=_Y_TICK_INTERVALS, steps=list(_TICK_STEPS)).tick_values(lower, upper)
    return float(ticks[0]), float(ticks[-1])


def _y_ticks(lower: float, upper: float) -> list[float]:
    """The ticks of a linear track: its two limits, and zero if it lies well between them.

    Parameters
    ----------
    lower, upper : float
        The limits of the y-axis.

    Returns
    -------
    list of float
        The ticks, lowest first.
    """
    # zero is labelled only far enough from both limits that its label clears theirs
    clearance = _ZERO_TICK_CLEARANCE * (upper - lower)
    if lower + clearance <= 0 <= upper - clearance:
        return [lower, 0.0, upper]
    return [lower, upper]


def _draw_linear_track(ax: Axes, track: LinearTrack) -> None:
    """Draw a linear track: its values, as a line or bars, on a y-axis of their own.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The track's axes.
    track : LinearTrack
        The track.
    """
    residues = np.arange(1, len(track.values) + 1)
    if track.plot_type == "line":
        ax.plot(residues, track.values, color=track.color, linewidth=track.line_width_pt)
    else:
        ax.bar(residues, track.values, width=1.0, color=track.color, linewidth=0)

    if track.y_limits is not None:
        lower, upper = track.y_limits
    else:
        lower, upper = _rounded_limits(track.values, include_zero=track.plot_type == "bar")
    ax.set_ylim(lower, upper)
    ax.set_yticks(_y_ticks(lower, upper))
    # the labels of the two limits sit within the track, the lower one above its
    # tick and the upper one below its tick, so they clear the labels of the
    # tracks above and below
    tick_labels = ax.get_yticklabels()
    tick_labels[0].set_verticalalignment("bottom")
    tick_labels[-1].set_verticalalignment("top")


def _draw_grid_track(ax: Axes, track: GridTrack) -> None:
    """Set up a grid track, with each row's name to its left; its heatmap is drawn once the figure is laid out.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The track's axes.
    track : GridTrack
        The track.
    """
    ax.set_ylim(track.n_rows, 0)
    ax.set_yticks(np.arange(track.n_rows) + 0.5, track.row_names)
    # the row names sit against the heatmap, with no tick marks
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def _grid_colors(track: GridTrack) -> npt.NDArray[np.uint8]:
    """Color each value of a grid track by its row's colormap and limits.

    Parameters
    ----------
    track : GridTrack
        The track.

    Returns
    -------
    numpy.ndarray
        The colors as RGBA bytes, of shape ``(rows, residues, 4)``. A residue
        with no value is transparent.
    """
    n_residues = track.values.shape[1]
    colors = np.zeros((track.n_rows, n_residues, 4), dtype=np.uint8)
    for row_index in range(track.n_rows):
        lower, upper = track.limits[row_index]
        colormap = matplotlib.colormaps[track.colormaps[row_index]]
        colors[row_index] = colormap(Normalize(vmin=lower, vmax=upper)(track.values[row_index]), bytes=True)
    return colors


def _heatmap_pixels_across(length_in: float) -> int:
    """Return how many pixels of a heatmap's image one cell spans, along one side.

    Parameters
    ----------
    length_in : float
        The length of the cell's side, in inches.

    Returns
    -------
    int
        The number of pixels, enough for ``_HEATMAP_PIXELS_PER_INCH``, and at least one.
    """
    return max(1, math.ceil(length_in * _HEATMAP_PIXELS_PER_INCH))


def _draw_heatmap(ax: Axes, track: GridTrack, residue_width_pt: float) -> None:
    """Draw a grid track's heatmap, each row of values as a strip of color.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The track's axes.
    track : GridTrack
        The track.
    residue_width_pt : float
        The width of a residue, in points.
    """
    n_residues = track.values.shape[1]
    colors = _grid_colors(track)

    # each cell is a block of identical pixels, so a viewer that smooths the
    # image blurs only the edge of each cell (see _HEATMAP_PIXELS_PER_INCH)
    pixels_per_row = _heatmap_pixels_across(track.row_height_in)
    pixels_per_residue = _heatmap_pixels_across(residue_width_pt / _POINTS_PER_INCH)
    blocks = np.repeat(np.repeat(colors, pixels_per_row, axis=0), pixels_per_residue, axis=1)

    # drawn as one image, residue i spanning i - 0.5 to i + 0.5, and the first
    # row on top. With no interpolation, a PDF or SVG holds the image as it is,
    # so each cell lines up with its residue, with no seams between cells
    ax.imshow(blocks, extent=(0.5, n_residues + 0.5, track.n_rows, 0), aspect="auto", interpolation="none")


def _draw_track(ax: Axes, track: Track, protein: VisualProtein, font_size_pt: float) -> None:
    """Draw a track of any kind, apart from the text of a sequence track, which is written once the figure is laid out.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The track's axes.
    track : Track
        The track.
    protein : VisualProtein
        The protein it is drawn for.
    font_size_pt : float
        The size of every label in the figure, in points.
    """
    # the axes are transparent, so highlights drawn behind them show through
    ax.set_facecolor("none")
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)

    if isinstance(track, DomainTrack):
        _draw_domain_track(ax, track, protein, font_size_pt)
    elif isinstance(track, SiteTrack):
        _draw_site_track(ax, track, protein, font_size_pt)
    elif isinstance(track, LinearTrack):
        _draw_linear_track(ax, track)
    elif isinstance(track, GridTrack):
        _draw_grid_track(ax, track)
    else:
        ax.set_ylim(0, 1)
        _hide_y_axis(ax)

    ax.set_ylabel(track.name, rotation=0, ha="right", va="center")


def _show_x_axis(ax: Axes, show: bool) -> None:
    """Show a track's x-axis, with the residue numbers, or hide it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The track's axes.
    show : bool
        Whether to show it: only the bottom track does.
    """
    ax.spines["bottom"].set_visible(show)
    ax.tick_params(axis="x", bottom=show, labelbottom=show)
    if show:
        ax.set_xlabel("Residue")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=list(_TICK_STEPS)))
