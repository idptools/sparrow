"""Linear plots of proteins: domains, per-residue values, heatmaps and sequence along a shared residue axis.

A plot is built from a :class:`VisualProtein` for each protein, which holds the
tracks to draw for it, top to bottom:

* :class:`DomainTrack` - regions drawn as boxes on a line that stands for the
  chain, such as folded domains or IDRs;
* :class:`SiteTrack` - single residues, such as phosphosites or mutations,
  each marked by a stem and a dot above the chain;
* :class:`LinearTrack` - one value per residue, drawn as a line or as bars, on
  a y-axis of its own;
* :class:`GridTrack` - one or more per-residue values, each drawn as a row of a
  heatmap with its own colormap;
* :class:`SequenceTrack` - the sequence of chosen regions, written within each
  region.

:func:`plot_proteins` draws one or more of them, residue 1 on the left, each
track on axes of its own and every protein on the same residue scale.
:func:`save_figure` saves the figure with its text kept as text, so it can be
edited in Illustrator.

.. code-block:: python

    from sparrow.visualize.proteinvis import (
        DomainTrack, LinearTrack, VisualProtein, plot_proteins, save_figure,
    )

    protein = VisualProtein("Med15", sequence=sequence)

    domains = DomainTrack("Folded")
    domains.add_domain(6, 91, label="KIX", color="#ffe699")
    protein.add_track(domains)

    protein.add_track(LinearTrack("Disorder", disorder, y_limits=(0, 1)))
    protein.add_highlight(6, 91)

    figure, axes = plot_proteins(protein)
    save_figure(figure, "med15.pdf")

Positions count from 1, and a region from ``start`` to ``end`` includes both,
so residue ``i`` spans ``i - 0.5`` to ``i + 0.5`` on the x-axis.

Figures are drawn in Arial, falling back to matplotlib's own DejaVu Sans where
Arial is not installed, with black lines and no grid. These settings apply only
while :func:`plot_proteins` draws and :func:`save_figure` saves, so importing
this module leaves matplotlib's settings as they were.
"""

# The parts of a figure, each in a module of its own by what it does. Every
# public name is imported here, so everything is imported from
# sparrow.visualize.proteinvis, never from the modules themselves:
#
# * _style - the look of every figure: its fonts, sizes and colors, the
#   defaults of each track, and the matplotlib settings it is drawn in
# * _checks - the checks of the arguments every part is made with
# * _features - Domain, Site, SequenceRegion and Highlight
# * _tracks - the five kinds of track, and Track, any one of them
# * _visual_protein - VisualProtein, a protein and its tracks
# * _draw_tracks - drawing each kind of track on its axes, apart from the
#   sequence
# * _sequence_layout - laying out and writing the sequence of a sequence track
# * _figure - laying out and drawing the whole figure (plot_proteins), and
#   saving it (save_figure)

from sparrow.visualize.proteinvis._features import Domain, Highlight, SequenceRegion, Site
from sparrow.visualize.proteinvis._figure import plot_proteins, save_figure
from sparrow.visualize.proteinvis._style import (
    BACKBONE_WIDTH_PT,
    DEFAULT_CHARACTERS_PER_BLOCK,
    DEFAULT_COLORMAP,
    DEFAULT_DOMAIN_COLOR,
    DEFAULT_DOMAIN_HEIGHT,
    DEFAULT_DOMAIN_TRACK_HEIGHT_IN,
    DEFAULT_DPI,
    DEFAULT_FIGURE_WIDTH_IN,
    DEFAULT_FONT_SIZE_PT,
    DEFAULT_GRID_ROW_HEIGHT_IN,
    DEFAULT_HIGHLIGHT_ALPHA,
    DEFAULT_HIGHLIGHT_COLOR,
    DEFAULT_LINE_WIDTH_PT,
    DEFAULT_LINEAR_TRACK_HEIGHT_IN,
    DEFAULT_SEQUENCE_FONT_SIZE_PT,
    DEFAULT_SITE_COLOR,
    DEFAULT_SITE_MARKER_SIZE_PT,
    DEFAULT_SITE_TRACK_HEIGHT_IN,
    DOMAIN_EDGE_WIDTH_PT,
    FONT_FAMILY,
    INK_COLOR,
    LABEL_POSITIONS,
    PLOT_TYPES,
    SEQUENCE_FONT_FAMILY,
    SITE_STEM_WIDTH_PT,
    LabelPosition,
    PlotType,
)
from sparrow.visualize.proteinvis._tracks import DomainTrack, GridTrack, LinearTrack, SequenceTrack, SiteTrack, Track
from sparrow.visualize.proteinvis._visual_protein import VisualProtein

__all__ = [
    "Domain",
    "DomainTrack",
    "GridTrack",
    "Highlight",
    "LinearTrack",
    "SequenceRegion",
    "SequenceTrack",
    "Site",
    "SiteTrack",
    "VisualProtein",
    "plot_proteins",
    "save_figure",
]
