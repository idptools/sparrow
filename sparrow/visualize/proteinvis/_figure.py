"""Drawing a whole figure, every protein's tracks stacked on one residue axis, and saving it."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.textpath import text_to_path
from matplotlib.transforms import blended_transform_factory, offset_copy

from sparrow.sparrow_exceptions import SparrowException
from sparrow.visualize.proteinvis._checks import _check_positive
from sparrow.visualize.proteinvis._draw_tracks import _draw_heatmap, _draw_track, _show_x_axis
from sparrow.visualize.proteinvis._sequence_layout import (
    _draw_sequence_text,
    _lay_out_sequence,
    _RegionText,
    _warn_if_sequence_spills,
)
from sparrow.visualize.proteinvis._style import (
    _EDITABLE_TEXT,
    _POINTS_PER_INCH,
    DEFAULT_DPI,
    DEFAULT_FIGURE_WIDTH_IN,
    DEFAULT_FONT_SIZE_PT,
    FONT_FAMILY,
    _rc_params,
)
from sparrow.visualize.proteinvis._tracks import GridTrack, SequenceTrack, SiteTrack
from sparrow.visualize.proteinvis._visual_protein import VisualProtein

# The space around the whole figure, between the tracks of one protein, and
# between one protein and the next, in inches
_MARGIN_IN = 0.05
_TRACK_GAP_IN = 0.06
_PROTEIN_GAP_IN = 0.2

# The space between a track's label and its axis (or its tick labels), in points
_LABEL_PAD_PT = 4.0

# The space between the top of a protein's first track and its name, and the
# height of the line the name is written on above that, in multiples of the
# font size. The track's top tick label is flush with its top, so without the
# space the name sits right on it.
_NAME_PAD = 0.5
_NAME_LINE_HEIGHT = 1.5

# How far below a site track the residue axis is drawn when the site track is
# the bottom one, in points. The chain line sits near the bottom of the track,
# so without this it reads as one doubled line with the axis; this puts it
# about as far above the axis as a domain track's chain line, 0.15 inches.
_SITE_AXIS_DROP_PT = 6.0

# The height a figure is made at before its tracks are laid out, in inches;
# it is replaced by the height they need
_PROVISIONAL_HEIGHT_IN = 4.0


def _text_width_pt(text: str, font_size_pt: float) -> float:
    """The width of a line of text in the figure's font, in points.

    Parameters
    ----------
    text : str
        The text.
    font_size_pt : float
        Its size, in points.

    Returns
    -------
    float
        Its width.
    """
    font = FontProperties(family=list(FONT_FAMILY), size=font_size_pt)
    width_pt, _, _ = text_to_path.get_text_width_height_descent(text, font, ismath=False)
    return float(width_pt)


@dataclass(frozen=True)
class _Margins:
    """The space the labels need around the tracks, in points.

    Attributes
    ----------
    left_pt : float
        From the left edge of the figure to the left edge of the tracks.
    label_offset_pt : float
        From the left edge of the tracks to the right edge of the track
        labels, which clear every track's tick labels.
    right_pt : float
        From the right edge of the tracks to the right edge of the figure.
    bottom_pt : float
        From the bottom of the last track to the bottom of the figure.
    """

    left_pt: float
    label_offset_pt: float
    right_pt: float
    bottom_pt: float


def _measure_margins(figure: Figure, axes: Sequence[Axes], longest: int, font_size_pt: float) -> _Margins:
    """Measure the space the tracks' labels need, once they are drawn.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure, with every track drawn.
    axes : sequence of matplotlib.axes.Axes
        Every track's axes, top to bottom.
    longest : int
        The length of the longest protein.
    font_size_pt : float
        The size of every label, in points.

    Returns
    -------
    _Margins
        The margins.
    """
    figure.draw_without_rendering()
    points_per_pixel = _POINTS_PER_INCH / figure.dpi

    # how far each track's tick labels reach to the left of it, and the widest track label
    tick_reach_pt = 0.0
    widest_label_pt = 0.0
    for ax in axes:
        tick_labels = [label for label in ax.get_yticklabels() if label.get_visible() and label.get_text()]
        if tick_labels:
            leftmost = min(label.get_window_extent().x0 for label in tick_labels)
            tick_reach_pt = max(tick_reach_pt, (ax.bbox.x0 - leftmost) * points_per_pixel)
        if ax.get_ylabel():
            widest_label_pt = max(widest_label_pt, ax.yaxis.label.get_window_extent().width * points_per_pixel)

    label_offset_pt = tick_reach_pt + _LABEL_PAD_PT
    margin_pt = _MARGIN_IN * _POINTS_PER_INCH
    left_pt = margin_pt + widest_label_pt + label_offset_pt

    # the residue numbers under the bottom track, and the axis label under them
    bottom_axes = axes[-1]
    x_axis_extent = bottom_axes.xaxis.get_tightbbox()
    assert x_axis_extent is not None, "the bottom track's x-axis is shown, so it has an extent"
    bottom_reach_pt = (bottom_axes.bbox.y0 - x_axis_extent.y0) * points_per_pixel

    # a residue number at the right end of the axis reaches half its width past it
    right_pt = margin_pt + _text_width_pt(str(longest), font_size_pt) / 2

    return _Margins(left_pt, label_offset_pt, right_pt, margin_pt + bottom_reach_pt)


def _place_tracks(
    figure: Figure,
    proteins: Sequence[VisualProtein],
    axes: Sequence[Sequence[Axes]],
    heights_in: Sequence[Sequence[float]],
    margins: _Margins,
    font_size_pt: float,
) -> None:
    """Stack the tracks down the figure, at their heights, and size the figure to fit them.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure.
    proteins : sequence of VisualProtein
        The proteins, top to bottom.
    axes : sequence of sequence of matplotlib.axes.Axes
        Each protein's tracks' axes, top to bottom.
    heights_in : sequence of sequence of float
        The height of each of those tracks, in inches.
    margins : _Margins
        The space the labels need.
    font_size_pt : float
        The size of every label, in points, which sets the height of the line
        each protein's name is written on.
    """
    name_height_in = font_size_pt * (_NAME_PAD + _NAME_LINE_HEIGHT) / _POINTS_PER_INCH

    # the top of each track, in inches from the top of the figure
    tops_in = []
    cursor_in = _MARGIN_IN
    for protein_index, protein in enumerate(proteins):
        if protein_index > 0:
            cursor_in += _PROTEIN_GAP_IN
        if protein.name:
            cursor_in += name_height_in
        protein_tops = []
        for track_index, height_in in enumerate(heights_in[protein_index]):
            if track_index > 0:
                cursor_in += _TRACK_GAP_IN
            protein_tops.append(cursor_in)
            cursor_in += height_in
        tops_in.append(protein_tops)

    width_in, _ = figure.get_size_inches()
    height_in = cursor_in + margins.bottom_pt / _POINTS_PER_INCH
    figure.set_size_inches(width_in, height_in)

    left_in = margins.left_pt / _POINTS_PER_INCH
    axes_width_in = width_in - left_in - margins.right_pt / _POINTS_PER_INCH
    for protein_index in range(len(proteins)):
        for ax, top_in, track_height_in in zip(axes[protein_index], tops_in[protein_index], heights_in[protein_index]):
            bottom_in = height_in - top_in - track_height_in
            ax.set_position(
                (left_in / width_in, bottom_in / height_in, axes_width_in / width_in, track_height_in / height_in)
            )
            # every track's label ends the same distance left of the tracks, clear of all their tick labels
            ax.yaxis.set_label_coords(-margins.label_offset_pt / _POINTS_PER_INCH / axes_width_in, 0.5)


def _draw_protein_name(figure: Figure, protein: VisualProtein, first_axes: Axes, font_size_pt: float) -> None:
    """Write a protein's name above its first track, at the left edge of the figure.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure.
    protein : VisualProtein
        The protein.
    first_axes : matplotlib.axes.Axes
        The axes of its first track.
    font_size_pt : float
        The size of every label in the figure, in points.
    """
    if not protein.name:
        return
    width_in, _ = figure.get_size_inches()
    # raised clear of the tick label at the top of the first track
    above_the_track = offset_copy(figure.transFigure, fig=figure, y=font_size_pt * _NAME_PAD, units="points")
    figure.text(
        _MARGIN_IN / width_in,
        first_axes.get_position().y1,
        protein.name,
        ha="left",
        va="bottom",
        transform=above_the_track,
    )


def _draw_highlights(figure: Figure, protein: VisualProtein, axes: Sequence[Axes]) -> None:
    """Draw each of a protein's highlights as one band behind all its tracks, and the gaps between them.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure, laid out.
    protein : VisualProtein
        The protein.
    axes : sequence of matplotlib.axes.Axes
        Its tracks' axes, top to bottom.
    """
    top = axes[0].get_position().y1
    bottom = axes[-1].get_position().y0
    # x in residues, y as a fraction of the figure's height
    residues_and_figure_height = blended_transform_factory(axes[0].transData, figure.transFigure)
    for highlight in protein.highlights:
        band = Rectangle(
            (highlight.start - 0.5, bottom),
            highlight.end - highlight.start + 1,
            top - bottom,
            transform=residues_and_figure_height,
            facecolor=highlight.color,
            alpha=highlight.alpha,
            linewidth=0,
            # behind the tracks, which are transparent
            zorder=-1,
        )
        figure.add_artist(band)


def _draw_proteins(
    figure: Figure, proteins: Sequence[VisualProtein], figure_width_in: float, font_size_pt: float
) -> list[list[Axes]]:
    """Draw every protein's tracks in a figure, lay them out, and size the figure to fit them.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure, empty.
    proteins : sequence of VisualProtein
        The proteins, top to bottom, each checked.
    figure_width_in : float
        The width of the figure, in inches.
    font_size_pt : float
        The size of every label, in points.

    Returns
    -------
    list of list of matplotlib.axes.Axes
        For each protein, the axes of each of its tracks, top to bottom.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the figure is too narrow for its labels.
    """
    longest = max(protein.length for protein in proteins)

    # 1. draw every track, on axes that share the residue axis; where the
    # axes go waits until the labels have been measured
    first_axes: Axes | None = None
    axes: list[list[Axes]] = []
    for protein in proteins:
        protein_axes = []
        for track in protein.tracks:
            # each axes gets a label of its own, so matplotlib never hands back one already made
            track_label = f"{protein.name} track {len(protein_axes)} of protein {len(axes)}"
            ax = figure.add_axes((0.0, 0.0, 1.0, 1.0), sharex=first_axes, label=track_label)
            if first_axes is None:
                first_axes = ax
            _draw_track(ax, track, protein, font_size_pt)
            protein_axes.append(ax)
        axes.append(protein_axes)

    all_axes = [ax for protein_axes in axes for ax in protein_axes]
    for ax in all_axes:
        _show_x_axis(ax, show=ax is all_axes[-1])
    # under a site track, the residue axis is moved down, clear of its chain line
    if isinstance(proteins[-1].tracks[-1], SiteTrack):
        all_axes[-1].spines["bottom"].set_position(("outward", _SITE_AXIS_DROP_PT))
    all_axes[0].set_xlim(0.5, longest + 0.5)

    # 2. measure the labels, which sets the width of the tracks, and so of a residue
    margins = _measure_margins(figure, all_axes, longest, font_size_pt)
    axes_width_pt = figure_width_in * _POINTS_PER_INCH - margins.left_pt - margins.right_pt
    if axes_width_pt <= 0:
        raise SparrowException(
            f"A figure {figure_width_in} inches wide leaves no room for the tracks beside their labels. Make it "
            "wider, or the labels shorter."
        )
    residue_width_pt = axes_width_pt / longest

    # 3. lay out each sequence track, whose height depends on the width of a residue
    sequence_texts: dict[int, list[_RegionText]] = {}
    heights_in: list[list[float]] = []
    for protein in proteins:
        protein_heights = []
        for track in protein.tracks:
            if isinstance(track, SequenceTrack):
                texts, height_in = _lay_out_sequence(track, protein, residue_width_pt)
                _warn_if_sequence_spills(track, texts)
                sequence_texts[id(track)] = texts
                protein_heights.append(height_in)
            else:
                protein_heights.append(track.height_in)
        heights_in.append(protein_heights)

    # 4. stack the tracks at their heights, then add what is placed relative
    # to them, or needs the width of a residue
    _place_tracks(figure, proteins, axes, heights_in, margins, font_size_pt)
    for protein, protein_axes in zip(proteins, axes):
        for track, ax in zip(protein.tracks, protein_axes):
            if isinstance(track, SequenceTrack):
                _draw_sequence_text(ax, figure, track, sequence_texts[id(track)])
            elif isinstance(track, GridTrack):
                _draw_heatmap(ax, track, residue_width_pt)
        _draw_protein_name(figure, protein, protein_axes[0], font_size_pt)
        _draw_highlights(figure, protein, protein_axes)

    # draw once more in this style, so the ticks matplotlib makes as it
    # draws are made in it too, and keep it when the figure is drawn later
    figure.draw_without_rendering()

    return axes


def plot_proteins(
    proteins: VisualProtein | Sequence[VisualProtein],
    figure_width_in: float = DEFAULT_FIGURE_WIDTH_IN,
    font_size_pt: float = DEFAULT_FONT_SIZE_PT,
    dpi: float = DEFAULT_DPI,
) -> tuple[Figure, list[list[Axes]]]:
    """Plot one or more proteins, each as its tracks stacked top to bottom along a shared residue axis.

    Residue 1 is on the left. Every protein is drawn on the same scale, so a
    shorter protein ends before the right edge, and the residue numbers are
    written once, under the last track. The figure is as wide as you ask,
    and as tall as its tracks: each track is the height it was made with, and
    the space for the labels is measured, so every label fits.

    Parameters
    ----------
    proteins : VisualProtein or sequence of VisualProtein
        The protein, or the proteins top to bottom.
    figure_width_in : float, optional
        The width of the figure, in inches. Default 6.5.
    font_size_pt : float, optional
        The size of every label, in points. Default 6. A sequence track and a
        domain track can set their own.
    dpi : float, optional
        The figure's resolution on screen, in dots per inch. Default 300. A
        PDF or SVG is drawn as shapes and text, whatever this is.

    Returns
    -------
    tuple
        ``(figure, axes)``: the matplotlib figure, and for each protein, the
        axes of each of its tracks, top to bottom. Save the figure with
        :func:`save_figure` to keep its text editable.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If there are no proteins, a protein has no tracks, a track does not
        fit its protein, or the figure is too narrow for its labels.

    Warns
    -----
    sparrow.sparrow_exceptions.SparrowWarning
        If a region of a sequence track is narrower than one letter, so its
        sequence spills out of it.
    """
    protein_list = [proteins] if isinstance(proteins, VisualProtein) else list(proteins)
    if not protein_list:
        raise SparrowException("There are no proteins to plot.")
    for protein in protein_list:
        if not isinstance(protein, VisualProtein):
            raise SparrowException(f"Each protein to plot must be a VisualProtein, not {type(protein).__name__}.")
        protein.validate()
    figure_width_in = _check_positive(figure_width_in, "figure_width_in")
    font_size_pt = _check_positive(font_size_pt, "font_size_pt")
    dpi = _check_positive(dpi, "dpi")

    # The figure is made before the style is applied, since pyplot loads its
    # backend when it makes its first figure, and a backend can change
    # rcParams as it loads: Jupyter's inline backend turns on interactive
    # mode, which shows each figure under its cell. Made inside the
    # rc_context, that change would be undone as the context ends, and no
    # later figure in the notebook would be shown.
    figure = plt.figure(figsize=(figure_width_in, _PROVISIONAL_HEIGHT_IN), dpi=dpi)
    try:
        with matplotlib.rc_context(_rc_params(font_size_pt)):
            axes = _draw_proteins(figure, protein_list, figure_width_in, font_size_pt)
    except BaseException:
        # a figure that could not be drawn is not left open in pyplot, where a
        # notebook would show it, half drawn, under the error
        plt.close(figure)
        raise
    return figure, axes


def save_figure(figure: Figure, path: str | os.PathLike[str], **savefig_kwargs: Any) -> None:
    """Save a figure with its text kept as text, so it can be edited in Illustrator.

    matplotlib decides how to save text when it saves, not when it draws, so
    a figure from :func:`plot_proteins` saved with ``figure.savefig`` keeps
    its text editable only if your own matplotlib settings say so. This saves
    it as TrueType (Type 42) in a PDF or PostScript file and as text in an
    SVG, whatever your settings.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure.
    path : str or os.PathLike
        Where to save it. The extension chooses the format, as for
        ``figure.savefig``.
    **savefig_kwargs
        Anything else ``figure.savefig`` takes, such as ``dpi`` or
        ``transparent``.
    """
    with matplotlib.rc_context(_EDITABLE_TEXT):
        figure.savefig(path, **savefig_kwargs)
