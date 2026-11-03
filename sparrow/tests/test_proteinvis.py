"""Tests for sparrow.visualize.proteinvis, the linear plots of proteins.

Each track is drawn on axes of its own, with residues along x, so the tests
read back what was drawn - the boxes, lines, image and letters - and check it
against the residues it stands for, as well as the layout, the checks on every
argument, and that matplotlib's own settings are left as they were.
"""

import ast
import importlib
import os
import pkgutil
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from sparrow.data.amino_acids import AA_COLOR  # noqa: E402
from sparrow.sparrow_exceptions import SparrowException, SparrowWarning  # noqa: E402
from sparrow.visualize import proteinvis  # noqa: E402
from sparrow.visualize.proteinvis import _draw_tracks, _sequence_layout  # noqa: E402
from sparrow.visualize.proteinvis import (  # noqa: E402
    DomainTrack,
    GridTrack,
    LinearTrack,
    SequenceTrack,
    SiteTrack,
    VisualProtein,
    plot_proteins,
    save_figure,
)

SEQUENCE = "MEEEKKKKSSSTTTDDDQQQQNNNNGGGGSSSSLLLVVVAAAPPPPRRRRKKKKEEEEDDDDWWFFYY"
LONG_SEQUENCE = "ACDEFGHIKLMNPQRSTVWY" * 50

# A matplotlib backend that, as Jupyter's inline backend does, turns on
# interactive mode as it is loaded
_BACKEND_THAT_TURNS_ON_INTERACTIVE_MODE = (
    "import matplotlib\n"
    "from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas\n"
    "matplotlib.interactive(True)\n"
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure a test makes."""
    yield
    plt.close("all")


def _protein_with_domains():
    """A protein of SEQUENCE, with one domain track of two domains."""
    protein = VisualProtein("Test", sequence=SEQUENCE)
    track = DomainTrack("Domains")
    track.add_domain(5, 12, label="A", color="red")
    track.add_domain(30, 30, label="B")
    protein.add_track(track)
    return protein


def _boxes(ax):
    """The domain boxes drawn on a track's axes."""
    return [patch for patch in ax.patches if isinstance(patch, Rectangle)]


# ..............................................................
# matplotlib's settings


def test_importing_leaves_matplotlib_settings_alone():
    """The old module set Arial and Type 42 fonts for the whole session on import."""
    script = (
        "import matplotlib\n"
        "before = dict(matplotlib.rcParams)\n"
        "import sparrow.visualize.proteinvis\n"
        "print(before == dict(matplotlib.rcParams))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "True"


def test_plotting_and_saving_leave_matplotlib_settings_alone(tmp_path):
    before = dict(matplotlib.rcParams)
    figure, _ = plot_proteins(_protein_with_domains())
    save_figure(figure, tmp_path / "figure.pdf")
    assert dict(matplotlib.rcParams) == before


def test_the_backend_loaded_for_the_first_figure_keeps_its_settings(tmp_path):
    """Jupyter's inline backend turns on interactive mode as pyplot loads it, so each figure is shown under its cell.

    pyplot loads its backend when it makes its first figure. Made inside the
    figure's style, the first figure of a notebook undid the change as the
    style ended, and no later figure in the notebook was shown.
    """
    (tmp_path / "settings_backend.py").write_text(_BACKEND_THAT_TURNS_ON_INTERACTIVE_MODE)
    script = (
        "import matplotlib\n"
        "from sparrow.visualize.proteinvis import LinearTrack, VisualProtein, plot_proteins\n"
        "protein = VisualProtein('Test', length=10)\n"
        "protein.add_track(LinearTrack('Values', list(range(10))))\n"
        "plot_proteins(protein)\n"
        "print(matplotlib.is_interactive())\n"
    )
    python_path = os.pathsep.join(path for path in [str(tmp_path), os.environ.get("PYTHONPATH", "")] if path)
    environment = {**os.environ, "MPLBACKEND": "module://settings_backend", "PYTHONPATH": python_path}
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, env=environment
    )
    assert result.stdout.strip() == "True"


def test_text_is_arial_or_its_fallback():
    figure, axes = plot_proteins(_protein_with_domains())
    ax = axes[0][0]
    for text in [ax.yaxis.label, *ax.get_xticklabels(), ax.xaxis.label]:
        assert text.get_fontname() in ("Arial", "DejaVu Sans")


def test_save_figure_keeps_text_editable(tmp_path):
    """Saved as TrueType (Type 42), not as Type 3 outlines, so Illustrator can edit it."""
    figure, _ = plot_proteins(_protein_with_domains())
    path = tmp_path / "figure.pdf"
    save_figure(figure, path)
    data = path.read_bytes()
    assert b"/Type3" not in data
    assert b"/FontFile2" in data


# ..............................................................
# Layout


def test_residues_run_left_to_right_from_residue_one():
    figure, axes = plot_proteins(_protein_with_domains())
    ax = axes[0][0]
    assert ax.get_xlim() == (0.5, len(SEQUENCE) + 0.5)
    assert not ax.xaxis_inverted()


def test_each_track_is_the_height_it_was_made_with():
    protein = _protein_with_domains()
    protein.add_track(LinearTrack("Values", np.linspace(0, 1, len(SEQUENCE)), height_in=0.7))
    protein.add_track(GridTrack("Grid", np.zeros((3, len(SEQUENCE))), row_height_in=0.2))

    figure, axes = plot_proteins(protein)
    _, figure_height_in = figure.get_size_inches()
    heights_in = [ax.get_position().height * figure_height_in for ax in axes[0]]
    assert heights_in == pytest.approx([proteinvis.DEFAULT_DOMAIN_TRACK_HEIGHT_IN, 0.7, 0.6])


def test_tracks_are_stacked_top_to_bottom_without_overlapping():
    protein = _protein_with_domains()
    protein.add_track(LinearTrack("Values", np.linspace(0, 1, len(SEQUENCE))))
    second = VisualProtein("Second", length=40)
    second.add_track(LinearTrack("More", np.ones(40)))

    figure, axes = plot_proteins([protein, second])
    positions = [ax.get_position() for protein_axes in axes for ax in protein_axes]
    for upper, lower in zip(positions, positions[1:]):
        assert upper.y0 > lower.y1


def test_every_protein_is_drawn_on_the_same_scale():
    long_protein = VisualProtein("Long", length=200)
    long_protein.add_track(DomainTrack("Chain"))
    short_protein = VisualProtein("Short", length=50)
    short_protein.add_track(DomainTrack("Chain"))

    figure, axes = plot_proteins([long_protein, short_protein])
    assert [len(protein_axes) for protein_axes in axes] == [1, 1]
    assert axes[1][0].get_xlim() == (0.5, 200.5)
    # the shorter chain ends at its own last residue
    assert list(axes[1][0].lines[0].get_xdata()) == [0.5, 50.5]


def test_only_the_bottom_track_shows_the_residue_numbers():
    protein = _protein_with_domains()
    protein.add_track(LinearTrack("Values", np.linspace(0, 1, len(SEQUENCE))))
    figure, axes = plot_proteins(protein)
    top, bottom = axes[0]
    assert not top.spines["bottom"].get_visible()
    assert bottom.spines["bottom"].get_visible()
    assert bottom.get_xlabel() == "Residue"


def test_every_label_fits_in_the_figure():
    protein = VisualProtein("A protein with a long name", sequence=SEQUENCE)
    protein.add_track(LinearTrack("A long label (kcal/mol)", np.linspace(-1000, 1000, len(SEQUENCE))))
    protein.add_track(GridTrack("Grid", np.zeros((2, len(SEQUENCE))), row_names=["Short", "A much longer row"]))

    figure, axes = plot_proteins(protein, figure_width_in=3)
    figure_box = figure.bbox
    for ax in axes[0]:
        for text in [ax.yaxis.label, *ax.get_yticklabels(), *ax.get_xticklabels()]:
            if text.get_visible() and text.get_text():
                extent = text.get_window_extent()
                assert extent.x0 >= figure_box.x0 - 1
                assert extent.x1 <= figure_box.x1 + 1


def test_track_labels_line_up():
    protein = _protein_with_domains()
    protein.add_track(LinearTrack("Values", np.linspace(0, 100, len(SEQUENCE))))
    figure, axes = plot_proteins(protein)
    right_edges = [ax.yaxis.label.get_window_extent().x1 for ax in axes[0]]
    assert right_edges[0] == pytest.approx(right_edges[1], abs=1)


def test_a_figure_too_narrow_for_its_labels_is_refused():
    protein = VisualProtein("P", length=10)
    protein.add_track(LinearTrack("A very long label indeed for so narrow a figure", np.ones(10)))
    open_figures = plt.get_fignums()
    with pytest.raises(SparrowException, match="no room"):
        plot_proteins(protein, figure_width_in=0.5)
    # the figure it could not draw is not left open
    assert plt.get_fignums() == open_figures


# ..............................................................
# Domain tracks


def test_a_domain_box_spans_its_residues():
    figure, axes = plot_proteins(_protein_with_domains())
    first, single = _boxes(axes[0][0])
    assert (first.get_x(), first.get_width()) == (4.5, 8)
    # a domain of one residue is one residue wide, not zero
    assert (single.get_x(), single.get_width()) == (29.5, 1)


def test_the_backbone_spans_the_chain():
    figure, axes = plot_proteins(_protein_with_domains())
    backbone = axes[0][0].lines[0]
    assert list(backbone.get_xdata()) == [0.5, len(SEQUENCE) + 0.5]


def test_the_backbone_can_be_left_out():
    protein = VisualProtein("P", length=20)
    protein.add_track(DomainTrack("No chain", show_backbone=False))
    figure, axes = plot_proteins(protein)
    assert len(axes[0][0].lines) == 0


def test_domain_labels_are_written_where_asked():
    protein = VisualProtein("P", length=100)
    track = DomainTrack("Domains")
    track.add_domain(10, 20, label="in")
    track.add_domain(30, 40, label="up", label_position="above")
    track.add_domain(50, 60, label="down", label_position="below")
    protein.add_track(track)

    figure, axes = plot_proteins(protein)
    labels = {text.get_text(): text for text in axes[0][0].texts}
    assert labels["in"].xy == (15, 0.5)
    assert labels["up"].xy[1] > 0.5 and labels["up"].get_va() == "bottom"
    assert labels["down"].xy[1] < 0.5 and labels["down"].get_va() == "top"


def test_overlapping_domains_in_one_track_are_refused():
    track = DomainTrack("Domains")
    track.add_domain(10, 20)
    with pytest.raises(SparrowException, match="overlaps"):
        track.add_domain(20, 30)
    # touching is not overlapping
    track.add_domain(21, 30)


def test_a_domain_added_after_its_track_is_still_checked():
    protein = VisualProtein("P", length=20)
    track = DomainTrack("Domains")
    protein.add_track(track)
    track.add_domain(15, 25)
    with pytest.raises(SparrowException, match="ends after the last residue"):
        plot_proteins(protein)


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"start": 0, "end": 5}, "whole number from 1"),
        ({"start": 6, "end": 5}, "before its start"),
        ({"start": 1.5, "end": 5}, "whole number from 1"),
        ({"start": 1, "end": 5, "color": "not a color"}, "color matplotlib can draw"),
        ({"start": 1, "end": 5, "height": 1.5}, "at most 1"),
        ({"start": 1, "end": 5, "label_position": "left"}, "label_position"),
    ],
)
def test_bad_domains_are_refused(arguments, message):
    with pytest.raises(SparrowException, match=message):
        DomainTrack("Domains").add_domain(**arguments)


def test_numpy_integers_are_positions():
    track = DomainTrack("Domains")
    track.add_domain(np.int64(3), np.int64(9))
    assert (track.domains[0].start, track.domains[0].end) == (3, 9)


# ..............................................................
# Site tracks


def _site_figure(track, length=len(SEQUENCE)):
    """Plot a protein of ``length`` residues with ``track``, and return its axes."""
    protein = VisualProtein("P", length=length)
    protein.add_track(track)
    figure, axes = plot_proteins(protein)
    return axes[0][0]


def test_each_site_is_a_stem_up_to_a_marker_at_its_residue():
    track = SiteTrack("Sites")
    track.add_site(1)
    track.add_site(20, color="red")
    track.add_site(len(SEQUENCE))
    ax = _site_figure(track)

    stems, markers = ax.collections
    for segment, position in zip(stems.get_segments(), [1, 20, len(SEQUENCE)]):
        (x0, y0), (x1, y1) = segment
        assert x0 == x1 == position
        assert (y0, y1) == (_draw_tracks._SITE_BACKBONE_Y, _draw_tracks._SITE_MARKER_Y)
    np.testing.assert_array_equal(markers.get_offsets()[:, 0], [1, 20, len(SEQUENCE)])
    assert matplotlib.colors.to_hex(markers.get_facecolors()[1]) == matplotlib.colors.to_hex("red")
    # a site at either end keeps its whole marker
    assert not markers.get_clip_on()
    # the chain runs the length of the protein, beneath the markers
    (backbone,) = ax.lines
    assert list(backbone.get_xdata()) == [0.5, len(SEQUENCE) + 0.5]


def test_a_sites_label_is_written_above_its_marker():
    track = SiteTrack("Sites")
    track.add_site(10, label="S10")
    track.add_site(30)
    ax = _site_figure(track)
    (label,) = ax.texts
    assert label.get_text() == "S10"
    assert label.xy == (10, _draw_tracks._SITE_MARKER_Y)
    assert label.get_verticalalignment() == "bottom"


def test_a_site_track_with_no_sites_is_the_chain_alone():
    ax = _site_figure(SiteTrack("Sites"))
    assert len(ax.collections) == 0
    assert len(ax.lines) == 1


def _chain_above_residue_numbers_pt(figure, ax, chain_y):
    """How far a bottom track's chain line, ``chain_y`` up its axes, sits above the residue numbers, in points."""
    figure.draw_without_rendering()
    chain_px = ax.transAxes.transform((0, chain_y))[1]
    numbers_top_px = max(label.get_window_extent().y1 for label in ax.get_xticklabels() if label.get_text())
    return (chain_px - numbers_top_px) * 72 / figure.dpi


def test_under_a_site_track_the_residue_axis_is_as_far_below_its_chain_as_under_a_domain_track():
    """A site track's chain line sits near the bottom of its track, so right above the axis the two read as one line."""
    site_protein = VisualProtein("P", length=50)
    site_protein.add_track(SiteTrack("Sites"))
    site_figure, site_axes = plot_proteins(site_protein)

    domain_protein = VisualProtein("P", length=50)
    domain_protein.add_track(DomainTrack("Domains"))
    domain_figure, domain_axes = plot_proteins(domain_protein)

    site_gap_pt = _chain_above_residue_numbers_pt(site_figure, site_axes[0][0], _draw_tracks._SITE_BACKBONE_Y)
    domain_gap_pt = _chain_above_residue_numbers_pt(domain_figure, domain_axes[0][0], 0.5)
    assert site_gap_pt == pytest.approx(domain_gap_pt, abs=1.0)


def test_a_site_track_above_another_leaves_its_axis_where_it_was():
    protein = VisualProtein("P", length=50)
    protein.add_track(SiteTrack("Sites"))
    protein.add_track(DomainTrack("Domains"))
    figure, axes = plot_proteins(protein)
    for ax in axes[0]:
        assert ax.spines["bottom"].get_position() == ("outward", 0.0)


def test_two_sites_at_one_residue_are_refused():
    track = SiteTrack("Sites")
    track.add_site(5)
    with pytest.raises(SparrowException, match="already has a site at residue 5"):
        track.add_site(5, label="again")


def test_a_site_after_the_last_residue_is_refused():
    track = SiteTrack("Sites")
    track.add_site(len(SEQUENCE) + 1)
    protein = VisualProtein("P", length=len(SEQUENCE))
    with pytest.raises(SparrowException, match="after the last residue"):
        protein.add_track(track)


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"position": 0}, "residue position"),
        ({"position": 2.5}, "residue position"),
        ({"position": 3, "label": 7}, "label must be a str"),
        ({"position": 3, "color": "not-a-color"}, "color"),
    ],
)
def test_bad_sites_are_refused(arguments, message):
    with pytest.raises(SparrowException, match=message):
        SiteTrack("Sites").add_site(**arguments)


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"height_in": 0}, "height_in"),
        ({"marker_size_pt": -1}, "marker_size_pt"),
        ({"name": 3}, "name must be a str"),
    ],
)
def test_bad_site_track_arguments_are_refused(arguments, message):
    with pytest.raises(SparrowException, match=message):
        SiteTrack(**arguments)


# ..............................................................
# Linear tracks


def test_a_linear_track_draws_its_values_unscaled():
    values = np.linspace(-0.3, 0.7, len(SEQUENCE))
    protein = VisualProtein("P", sequence=SEQUENCE)
    protein.add_track(LinearTrack("Values", values))
    figure, axes = plot_proteins(protein)
    line = axes[0][0].lines[0]
    np.testing.assert_array_equal(line.get_xdata(), np.arange(1, len(SEQUENCE) + 1))
    np.testing.assert_array_equal(line.get_ydata(), values)


def test_a_linear_track_rounds_its_range_outwards():
    values = np.linspace(0.03, 0.97, 30)
    protein = VisualProtein("P", length=30)
    protein.add_track(LinearTrack("Values", values))
    figure, axes = plot_proteins(protein)
    assert axes[0][0].get_ylim() == pytest.approx((0, 1))
    assert list(axes[0][0].get_yticks()) == pytest.approx([0, 1])


def test_zero_is_ticked_only_well_between_the_limits():
    assert _draw_tracks._y_ticks(-0.4, 0.6) == [-0.4, 0.0, 0.6]
    # zero right by the upper limit would crowd its label
    assert _draw_tracks._y_ticks(-20, 5) == [-20, 5]
    assert _draw_tracks._y_ticks(0, 1) == [0, 1]


def test_tick_labels_stay_within_their_track():
    """So two linear tracks, one above the other, cannot have overlapping tick labels."""
    protein = VisualProtein("P", length=30)
    protein.add_track(LinearTrack("Upper", np.linspace(0, 1, 30)))
    protein.add_track(LinearTrack("Lower", np.linspace(-0.4, 0.4, 30)))
    figure, axes = plot_proteins(protein)
    for ax in axes[0]:
        for label in ax.get_yticklabels():
            extent = label.get_window_extent()
            assert extent.y0 >= ax.bbox.y0 - 0.5
            assert extent.y1 <= ax.bbox.y1 + 0.5


def test_y_limits_are_used_as_given():
    protein = VisualProtein("P", length=30)
    protein.add_track(LinearTrack("Values", np.full(30, 0.5), y_limits=(0, 2)))
    figure, axes = plot_proteins(protein)
    assert axes[0][0].get_ylim() == (0, 2)


def test_bars_are_drawn_from_zero():
    values = np.linspace(0.2, 0.8, 30)
    protein = VisualProtein("P", length=30)
    protein.add_track(LinearTrack("Values", values, plot_type="bar"))
    figure, axes = plot_proteins(protein)
    ax = axes[0][0]
    assert ax.get_ylim()[0] == 0
    bars = ax.patches
    assert len(bars) == 30
    assert [bar.get_x() for bar in bars[:2]] == pytest.approx([0.5, 1.5])
    assert bars[0].get_height() == pytest.approx(0.2)


def test_linear_track_values_with_gaps_are_drawn():
    values = np.linspace(0, 1, 30)
    values[10:15] = np.nan
    protein = VisualProtein("P", length=30)
    protein.add_track(LinearTrack("Values", values))
    plot_proteins(protein)


@pytest.mark.parametrize(
    "values, message",
    [
        (np.ones(29), "one value for each"),
        (np.ones((2, 30)), "Use a GridTrack"),
        (np.full(30, np.nan), "all NaN"),
        (np.r_[np.ones(29), np.inf], "infinite"),
        (["a"] * 30, "must be numbers"),
    ],
)
def test_bad_linear_track_values_are_refused(values, message):
    protein = VisualProtein("P", length=30)
    with pytest.raises(SparrowException, match=message):
        protein.add_track(LinearTrack("Values", values))


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"plot_type": "scatter"}, "plot_type"),
        ({"y_limits": (1, 0)}, "lower below upper"),
        ({"color": "nope"}, "color"),
        ({"height_in": 0}, "greater than 0"),
    ],
)
def test_bad_linear_track_arguments_are_refused(arguments, message):
    with pytest.raises(SparrowException, match=message):
        LinearTrack("Values", np.ones(10), **arguments)


# ..............................................................
# Grid tracks


def _heatmap_cells(image, n_rows, n_residues):
    """The color of each cell of a heatmap, of shape (rows, residues, 4), checking each cell is one block of color."""
    pixels = np.asarray(image.get_array())
    pixels_per_row = pixels.shape[0] // n_rows
    pixels_per_residue = pixels.shape[1] // n_residues
    assert pixels.shape == (n_rows * pixels_per_row, n_residues * pixels_per_residue, 4)
    cells = pixels[::pixels_per_row, ::pixels_per_residue]
    blocks = np.repeat(np.repeat(cells, pixels_per_row, axis=0), pixels_per_residue, axis=1)
    np.testing.assert_array_equal(pixels, blocks)
    return cells


def test_a_grid_track_colors_each_row_by_its_own_colormap_and_limits():
    rows = np.array([np.linspace(0, 1, 30), np.linspace(-1, 1, 30)])
    rows[1, 5] = np.nan
    protein = VisualProtein("P", length=30)
    protein.add_track(GridTrack("Grid", rows, colormaps=["Greens", "seismic"], limits=[(0, 0.5), None]))

    figure, axes = plot_proteins(protein)
    (image,) = axes[0][0].images
    assert image.get_extent() == [0.5, 30.5, 2, 0]
    cells = _heatmap_cells(image, n_rows=2, n_residues=30)

    expected_first = matplotlib.colormaps["Greens"](Normalize(0, 0.5)(rows[0]), bytes=True)
    np.testing.assert_array_equal(cells[0], expected_first)
    expected_second = matplotlib.colormaps["seismic"](Normalize(-1, 1)(rows[1]), bytes=True)
    np.testing.assert_array_equal(cells[1][~np.isnan(rows[1])], expected_second[~np.isnan(rows[1])])
    # a residue with no value is left transparent
    assert cells[1, 5, 3] == 0


@pytest.mark.parametrize("n_residues", [10, 300, 20000])
def test_a_heatmap_has_enough_pixels_that_smoothing_it_blurs_only_the_edge_of_each_cell(n_residues):
    protein = VisualProtein("P", length=n_residues)
    protein.add_track(GridTrack("Grid", np.zeros((2, n_residues)), row_height_in=0.15))

    figure, axes = plot_proteins(protein)
    ax = axes[0][0]
    (image,) = ax.images
    n_pixel_rows, n_pixel_columns, _ = np.asarray(image.get_array()).shape
    figure_width_in, figure_height_in = figure.get_size_inches()
    width_in = ax.get_position().width * figure_width_in
    height_in = ax.get_position().height * figure_height_in
    # allowing for rounding in the axes' position
    least_pixels_per_inch = _draw_tracks._HEATMAP_PIXELS_PER_INCH * (1 - 1e-9)
    assert n_pixel_rows / height_in >= least_pixels_per_inch
    assert n_pixel_columns / width_in >= least_pixels_per_inch
    # each cell is a whole number of pixels, at least one, even for a residue
    # narrower than a pixel
    assert n_pixel_rows % 2 == 0
    assert n_pixel_columns % n_residues == 0


def test_a_grid_track_names_its_rows_top_to_bottom():
    protein = VisualProtein("P", length=10)
    protein.add_track(GridTrack("Grid", np.zeros((2, 10)), row_names=["first", "second"]))
    figure, axes = plot_proteins(protein)
    ax = axes[0][0]
    assert [label.get_text() for label in ax.get_yticklabels()] == ["first", "second"]
    # the first row is on top
    assert ax.get_ylim() == (2, 0)


def test_a_single_profile_is_one_row():
    track = GridTrack("Grid", np.linspace(0, 1, 10))
    assert track.n_rows == 1


def test_transposed_grid_values_are_refused_with_the_reason():
    protein = VisualProtein("P", length=30)
    with pytest.raises(SparrowException, match="transposing"):
        protein.add_track(GridTrack("Grid", np.zeros((30, 3))))


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"colormaps": ["Greens"]}, "2 rows but 1 colormaps"),
        ({"colormaps": "not-a-colormap"}, "no colormap"),
        ({"limits": [(0, 1)]}, "2 rows but 1 limits"),
        ({"limits": [(1, 0), None]}, "lower below upper"),
        ({"row_names": ["one"]}, "needs 2 row names"),
    ],
)
def test_bad_grid_track_arguments_are_refused(arguments, message):
    with pytest.raises(SparrowException, match=message):
        GridTrack("Grid", np.zeros((2, 10)), **arguments)


def test_an_all_nan_grid_row_needs_its_limits():
    rows = np.array([np.ones(10), np.full(10, np.nan)])
    with pytest.raises(SparrowException, match="all NaN"):
        GridTrack("Grid", rows)
    GridTrack("Grid", rows, limits=[None, (0, 1)])


# ..............................................................
# Sequence tracks


def _protein_for(track, sequence=SEQUENCE):
    """A protein of ``sequence`` holding ``track``."""
    protein = VisualProtein("P", sequence=sequence)
    protein.add_track(track)
    return protein


def _sequence_figure(track):
    """Plot a protein of SEQUENCE with ``track``, and return the figure and the track's letters, in order."""
    figure, axes = plot_proteins(_protein_for(track))
    return figure, axes[0][0].texts


def test_a_region_is_written_in_full_and_in_order():
    track = SequenceTrack("Sequence")
    track.add_region(5, 40)
    figure, letters = _sequence_figure(track)
    assert "".join(letter.get_text() for letter in letters) == SEQUENCE[4:40]


def test_a_region_wraps_into_blocks_of_the_size_asked():
    track = SequenceTrack("Sequence", chars_per_block=5, blocks_per_line=2)
    track.add_region(1, 23)
    figure, axes = plot_proteins(_protein_for(track))
    renderer = figure.canvas.get_renderer()
    letters = axes[0][0].texts
    positions = [letter.get_window_extent(renderer) for letter in letters]
    tops = sorted({round(position.y0, 3) for position in positions}, reverse=True)
    # 23 residues in lines of 10 is three lines
    assert len(tops) == 3
    first_line = [position for position in positions if round(position.y0, 3) == tops[0]]
    assert len(first_line) == 10
    # a blank cell separates the two blocks of a line
    centers = [(position.x0 + position.x1) / 2 for position in first_line]
    gaps = np.diff(centers)
    assert gaps[4] == pytest.approx(2 * gaps[0], rel=1e-3)


def test_a_long_region_fits_more_blocks_on_a_line():
    track = SequenceTrack("Sequence")
    track.add_region(1, 300)
    track.add_region(500, 510)
    texts, _ = _sequence_layout._lay_out_sequence(track, _protein_for(track, LONG_SEQUENCE), residue_width_pt=1.0)
    long_region, short_region = texts
    assert long_region.n_lines < 300 // proteinvis.DEFAULT_CHARACTERS_PER_BLOCK
    # a region too narrow for one block holds as many residues on a line as fit
    pitch_pt = _sequence_layout._character_pitch_pt(track.font_size_pt)
    letters_per_line = int((11 * 1.0 - 2 * _sequence_layout._SEQUENCE_INSET_PT) // pitch_pt)
    assert 0 < letters_per_line < proteinvis.DEFAULT_CHARACTERS_PER_BLOCK
    assert short_region.n_lines == -(-11 // letters_per_line)


def test_each_regions_sequence_is_centered_across_the_region():
    track = SequenceTrack("Sequence")
    track.add_region(1, 30)  # one block to a line
    track.add_region(100, 400)  # several blocks to a line
    track.add_region(500, 515)  # too narrow for a block
    residue_width_pt = 1.0
    texts, _ = _sequence_layout._lay_out_sequence(track, _protein_for(track, LONG_SEQUENCE), residue_width_pt)
    pitch_pt = _sequence_layout._character_pitch_pt(track.font_size_pt)
    for text in texts:
        assert text.n_lines > 1
        region_width_pt = (text.region.end - text.region.start + 1) * residue_width_pt
        column_left_pt = min(x_pt for _, x_pt, _ in text.characters) - pitch_pt / 2
        column_right_pt = max(x_pt for _, x_pt, _ in text.characters) + pitch_pt / 2
        assert column_left_pt == pytest.approx(region_width_pt - column_right_pt)
        # every line starts at the same place, so the blocks line up in columns
        line_starts_pt = {}
        for _, x_pt, line in text.characters:
            line_starts_pt[line] = min(x_pt, line_starts_pt.get(line, x_pt))
        assert max(line_starts_pt.values()) == pytest.approx(min(line_starts_pt.values()))


def _letters_outside_their_regions(figure, ax, track):
    """The letters of a sequence track that reach outside their region, as drawn."""
    renderer = figure.canvas.get_renderer()
    letters = iter(ax.texts)
    outside = []
    for region in sorted(track.regions, key=lambda region: region.start):
        # the region spans from the left edge of its first residue to the right edge of its last
        left, _ = ax.transData.transform((region.start - 0.5, 0))
        right, _ = ax.transData.transform((region.end + 0.5, 0))
        for _ in range(region.end - region.start + 1):
            letter = next(letters)
            extent = letter.get_window_extent(renderer)
            if extent.x0 < left or extent.x1 > right:
                outside.append((region.start, letter.get_text()))
    return outside


@pytest.mark.parametrize("blocks_per_line", [None, 20])
def test_each_regions_sequence_stays_within_the_region(blocks_per_line):
    # on a protein of 1000 residues a residue is under half a point wide, so
    # these regions range from a few letters wide to many blocks
    track = SequenceTrack("Sequence", blocks_per_line=blocks_per_line)
    for start, end in [(1, 30), (100, 112), (120, 132), (200, 260), (300, 999)]:
        track.add_region(start, end)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SparrowWarning)
        figure, axes = plot_proteins(_protein_for(track, LONG_SEQUENCE))
    assert _letters_outside_their_regions(figure, axes[0][0], track) == []


def test_a_narrower_figure_wraps_the_sequence_onto_more_lines():
    track = SequenceTrack("Sequence")
    track.add_region(1, 200)
    heights_in = []
    for figure_width_in in (6.5, 3.5):
        figure, axes = plot_proteins(_protein_for(track, LONG_SEQUENCE), figure_width_in=figure_width_in)
        _, figure_height_in = figure.get_size_inches()
        heights_in.append(axes[0][0].get_position().height * figure_height_in)
        assert _letters_outside_their_regions(figure, axes[0][0], track) == []
    assert heights_in[1] > heights_in[0]


def test_letters_are_colored_by_residue_and_by_the_colors_given():
    track = SequenceTrack("Sequence", colors={"K": "orange"})
    track.add_region(1, 8)
    figure, letters = _sequence_figure(track)
    colors = {letter.get_text(): matplotlib.colors.to_hex(letter.get_color()) for letter in letters}
    assert colors["E"] == AA_COLOR["E"]
    assert colors["K"] == matplotlib.colors.to_hex("orange")


def test_a_sequence_track_is_as_tall_as_its_most_lines():
    track = SequenceTrack("Sequence", blocks_per_line=1)
    track.add_region(1, 30)
    figure, axes = plot_proteins(_protein_for(track))
    _, figure_height_in = figure.get_size_inches()
    height_pt = axes[0][0].get_position().height * figure_height_in * 72
    line_height_pt = track.font_size_pt * _sequence_layout._SEQUENCE_LINE_HEIGHT
    assert height_pt == pytest.approx(2 * _sequence_layout._SEQUENCE_PAD_PT + 3 * line_height_pt)


def test_a_region_narrower_than_a_letter_warns():
    # on a protein of 1000 residues, three residues are under a letter wide
    track = SequenceTrack("Sequence")
    track.add_region(100, 102)
    track.add_region(200, 260)
    with pytest.warns(SparrowWarning, match=r"narrower than one letter \(residues 100 to 102\)"):
        plot_proteins(_protein_for(track, LONG_SEQUENCE))


def test_the_warning_points_at_the_line_that_plotted():
    track = SequenceTrack("Sequence")
    track.add_region(100, 102)
    with pytest.warns(SparrowWarning, match="narrower than one letter") as record:
        plot_proteins(_protein_for(track, LONG_SEQUENCE))
    (warning,) = [warning for warning in record if issubclass(warning.category, SparrowWarning)]
    assert warning.filename == __file__


def test_a_figure_that_fails_after_it_is_made_is_not_left_open():
    """A notebook shows every figure left open, so one that failed would be shown half drawn under the error."""
    track = SequenceTrack("Sequence")
    track.add_region(100, 102)
    open_figures = plt.get_fignums()
    with warnings.catch_warnings():
        # the warning is raised once the figure is made and its tracks are drawn
        warnings.simplefilter("error", SparrowWarning)
        with pytest.raises(SparrowWarning, match="narrower than one letter"):
            plot_proteins(_protein_for(track, LONG_SEQUENCE))
    assert plt.get_fignums() == open_figures


def test_sequence_that_fits_does_not_warn():
    track = SequenceTrack("Sequence")
    track.add_region(1, 20)
    track.add_region(40, 60)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SparrowWarning)
        _sequence_figure(track)


def test_a_sequence_track_needs_the_sequence():
    protein = VisualProtein("P", length=30)
    track = SequenceTrack("Sequence")
    track.add_region(1, 10)
    with pytest.raises(SparrowException, match="without one"):
        protein.add_track(track)


def test_overlapping_sequence_regions_are_refused():
    track = SequenceTrack("Sequence")
    track.add_region(1, 10)
    with pytest.raises(SparrowException, match="overlaps"):
        track.add_region(5, 15)


# ..............................................................
# Proteins and highlights


def test_a_highlight_is_one_band_behind_all_of_a_proteins_tracks():
    protein = _protein_with_domains()
    protein.add_track(LinearTrack("Values", np.linspace(0, 1, len(SEQUENCE))))
    protein.add_highlight(10, 20, color="blue", alpha=0.3)

    figure, axes = plot_proteins(protein)
    (band,) = [artist for artist in figure.artists if isinstance(artist, Rectangle)]
    assert (band.get_x(), band.get_width()) == (9.5, 11)
    top, bottom = axes[0]
    assert band.get_y() == pytest.approx(bottom.get_position().y0)
    assert band.get_y() + band.get_height() == pytest.approx(top.get_position().y1)
    assert band.get_zorder() < top.get_zorder()


def test_a_proteins_name_is_written_above_its_tracks():
    figure, axes = plot_proteins(_protein_with_domains())
    (name,) = [text for text in figure.texts if text.get_text() == "Test"]
    figure.draw_without_rendering()
    assert name.get_window_extent().y0 > axes[0][0].get_window_extent().y1


def test_a_proteins_name_is_clear_of_the_tick_label_at_the_top_of_its_first_track():
    """A long name reaches over the tick labels, and written right on the track it sat on the top one."""
    protein = VisualProtein("A protein with a long name", length=50)
    protein.add_track(LinearTrack("Values", np.linspace(0, 1, 50), y_limits=(0, 1)))
    figure, axes = plot_proteins(protein)
    figure.draw_without_rendering()
    (name,) = [text for text in figure.texts if text.get_text() == "A protein with a long name"]
    tick_label_tops = [label.get_window_extent().y1 for label in axes[0][0].get_yticklabels() if label.get_text()]
    gap_pt = (name.get_window_extent().y0 - max(tick_label_tops)) * 72 / figure.dpi
    assert gap_pt >= proteinvis.DEFAULT_FONT_SIZE_PT / 4


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"name": "P"}, "length or its sequence"),
        ({"name": "P", "length": 0}, "whole number from 1"),
        ({"name": "P", "length": 5, "sequence": "MEEK"}, "4 residues"),
        ({"name": "P", "sequence": ""}, "at least one residue"),
    ],
)
def test_bad_proteins_are_refused(arguments, message):
    with pytest.raises(SparrowException, match=message):
        VisualProtein(**arguments)


def test_a_protein_with_no_tracks_is_refused():
    with pytest.raises(SparrowException, match="no tracks"):
        plot_proteins(VisualProtein("P", length=10))


def test_only_tracks_can_be_added():
    with pytest.raises(SparrowException, match="must be a DomainTrack"):
        VisualProtein("P", length=10).add_track("not a track")


def test_a_highlight_must_lie_within_the_protein():
    with pytest.raises(SparrowException, match="ends after the last residue"):
        VisualProtein("P", length=10).add_highlight(5, 11)


# ..............................................................
# The package


def _public_names_defined_in(module):
    """The public names a module defines itself, rather than imports."""
    tree = ast.parse(Path(module.__file__).read_text())
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {name for name in names if not name.startswith("_")}


def test_every_public_name_of_the_package_is_imported_from_it():
    """The code is in private modules, by what it does, and everything public is imported from proteinvis."""
    defined = set()
    for module_info in pkgutil.iter_modules(proteinvis.__path__):
        module = importlib.import_module(f"{proteinvis.__name__}.{module_info.name}")
        for name in _public_names_defined_in(module):
            assert getattr(proteinvis, name, None) is getattr(module, name), f"{module.__name__}.{name}"
            defined.add(name)
    assert set(proteinvis.__all__) <= defined
