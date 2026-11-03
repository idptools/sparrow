"""The look of every figure: its fonts, sizes and colors, and the defaults of each track.

The matplotlib settings a figure is drawn in are made here too, by
``_rc_params``; they apply only while a figure is drawn or saved.
"""

from __future__ import annotations

from typing import Any, Literal

# ..............................................................
# The look of a figure

#: The font of every label, Arial, with matplotlib's own DejaVu Sans where Arial is not installed
FONT_FAMILY = ("Arial", "DejaVu Sans")

#: The font the sequence is written in. It is monospaced, so the blocks line up, and ships with matplotlib
SEQUENCE_FONT_FAMILY = "DejaVu Sans Mono"

#: The size of every label, in points, when none is given
DEFAULT_FONT_SIZE_PT = 6.0

#: The width of a figure, in inches, when none is given
DEFAULT_FIGURE_WIDTH_IN = 6.5

#: The resolution of a figure on screen, in dots per inch, when none is given
DEFAULT_DPI = 300

#: The color of every line and label
INK_COLOR = "black"

# The width of the axis lines and ticks, and the length of the ticks, in points
_AXIS_LINE_WIDTH_PT = 0.75
_TICK_LENGTH_PT = 2.5

# The space between a tick and its label, and between an axis and its label, in points
_TICK_PAD_PT = 2.0

# Text saved as text rather than as outlines - TrueType (Type 42) in PDF and
# PostScript, and plain text in SVG - so it can be edited in Illustrator
_EDITABLE_TEXT: dict[str, Any] = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}

# ..............................................................
# Track defaults

#: The height of a domain track, in inches
DEFAULT_DOMAIN_TRACK_HEIGHT_IN = 0.3

#: The height of a domain's box, as a fraction of its track's height
DEFAULT_DOMAIN_HEIGHT = 0.6

#: The fill of a domain's box
DEFAULT_DOMAIN_COLOR = "white"

#: The width of the line that stands for the chain, in points
BACKBONE_WIDTH_PT = 1.0

#: The width of the outline of a domain's box, in points
DOMAIN_EDGE_WIDTH_PT = 0.5

#: The height of a site track, in inches
DEFAULT_SITE_TRACK_HEIGHT_IN = 0.3

#: The diameter of a site's marker, in points
DEFAULT_SITE_MARKER_SIZE_PT = 3.0

#: The color of a site's stem and marker
DEFAULT_SITE_COLOR = INK_COLOR

#: The width of a site's stem, in points
SITE_STEM_WIDTH_PT = 0.5

#: The height of a linear track, in inches
DEFAULT_LINEAR_TRACK_HEIGHT_IN = 0.5

#: The width of a linear track's line, in points
DEFAULT_LINE_WIDTH_PT = 0.75

#: The height of each row of a grid track, in inches
DEFAULT_GRID_ROW_HEIGHT_IN = 0.1

#: The colormap of each row of a grid track
DEFAULT_COLORMAP = "viridis"

#: The size the sequence is written at, in points
DEFAULT_SEQUENCE_FONT_SIZE_PT = 4.0

#: The number of residues in each block of the sequence, as :func:`~sparrow.visualize.sequence_visuals.show_sequence` writes it
DEFAULT_CHARACTERS_PER_BLOCK = 10

#: The color and opacity of a highlight
DEFAULT_HIGHLIGHT_COLOR = "red"
DEFAULT_HIGHLIGHT_ALPHA = 0.15

#: Where a domain's label can be written
LABEL_POSITIONS = ("inside", "above", "below")

#: How a linear track can be drawn
PLOT_TYPES = ("line", "bar")

LabelPosition = Literal["inside", "above", "below"]
PlotType = Literal["line", "bar"]

# The number of points in an inch, which every size in a figure is measured in
_POINTS_PER_INCH = 72.0


def _rc_params(font_size_pt: float) -> dict[str, Any]:
    """The matplotlib settings every figure is drawn with.

    Parameters
    ----------
    font_size_pt : float
        The size of every label, in points.

    Returns
    -------
    dict
        rcParams: Arial at ``font_size_pt``, black lines and labels, no grid,
        and text saved as text.
    """
    return {
        "font.family": list(FONT_FAMILY),
        "font.size": font_size_pt,
        "axes.labelsize": font_size_pt,
        "xtick.labelsize": font_size_pt,
        "ytick.labelsize": font_size_pt,
        "text.color": INK_COLOR,
        "axes.edgecolor": INK_COLOR,
        "axes.labelcolor": INK_COLOR,
        "xtick.color": INK_COLOR,
        "ytick.color": INK_COLOR,
        "axes.linewidth": _AXIS_LINE_WIDTH_PT,
        "xtick.major.width": _AXIS_LINE_WIDTH_PT,
        "ytick.major.width": _AXIS_LINE_WIDTH_PT,
        "xtick.major.size": _TICK_LENGTH_PT,
        "ytick.major.size": _TICK_LENGTH_PT,
        "xtick.major.pad": _TICK_PAD_PT,
        "ytick.major.pad": _TICK_PAD_PT,
        "axes.labelpad": _TICK_PAD_PT,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "axes.grid": False,
        **_EDITABLE_TEXT,
    }
