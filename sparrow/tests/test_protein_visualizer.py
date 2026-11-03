"""Tests for Protein.visualize, the plots of a Protein drawn from its own analyses.

Each test reads back what was drawn - the line, boxes, sites and letters on
each track's axes - and checks it against what the protein itself computes,
or against the values, sites and domains it was given.
"""

import ast
import importlib
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
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from sparrow import Protein  # noqa: E402
from sparrow.sparrow_exceptions import SparrowException  # noqa: E402
from sparrow.visualize import protein_visualizer  # noqa: E402
from sparrow.visualize.protein_visualizer import ProteinVisualizer  # noqa: E402
from sparrow.visualize.protein_visualizer import _analyses  # noqa: E402
from sparrow.visualize.proteinvis import (  # noqa: E402
    Domain,
    LinearTrack,
    Site,
    VisualProtein,
    plot_proteins,
)

# a disordered N-terminus with a Q-rich stretch, ubiquitin, which is folded,
# and a disordered C-terminus, so there are two IDRs and a low-complexity domain
DISORDERED_N_TERMINUS = "MSEEKKRSDSEEKKPSTPSSGSQQQQQQAQQQQQQSQQQQPGSG"
UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
DISORDERED_C_TERMINUS = "TTSQPSSAEEDDKRKRSTPTSPGSYGSQQQNNQQQNNSQQ"
SEQUENCE = DISORDERED_N_TERMINUS + UBIQUITIN + DISORDERED_C_TERMINUS
LENGTH = len(SEQUENCE)

# the page that documents Protein.visualize, with a table of the analyses it labels
DOCS_PAGE = Path(__file__).resolve().parents[2] / "docs" / "protein" / "visualization.rst"

# the arguments the analyses that need them are drawn with, in the test of every analysis
ARGUMENTS = {
    "linear_sequence_profile": {"mode": "FCR"},
    "linear_composition_profile": {"composition_list": ["Q"]},
    "linear_property_profile": {"mode": "hydropathy-kyte-1982"},
    "low_complexity_mask": {"residue_selector": "Q"},
    "low_complexity_domains": {"residue_selector": "Q"},
}


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure a test makes."""
    yield
    plt.close("all")


@pytest.fixture
def protein():
    return Protein(SEQUENCE)


def _labels(axes):
    """The label of each track, top to bottom."""
    return [ax.yaxis.label.get_text() for ax in axes]


def _boxes(ax):
    """The boxes on a domain track, as (start, end) residues, in order along the protein."""
    boxes = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
    return sorted((round(box.get_x() + 0.5), round(box.get_x() + box.get_width() - 0.5)) for box in boxes)


def _sites(ax):
    """The residues marked on a site track, in order."""
    (markers,) = ax.collections[1:]
    return sorted(int(round(x)) for x, _ in markers.get_offsets())


def _runs(mask):
    """The runs of 1s in a mask, as (start, end) residues counting from 1, worked out one residue at a time."""
    runs = []
    for index, value in enumerate(mask):
        if value and (index == 0 or not mask[index - 1]):
            runs.append([index + 1, index + 1])
        elif value:
            runs[-1][1] = index + 1
    return [tuple(run) for run in runs]


# ..............................................................
# The accessor


def test_visualize_is_made_once_and_kept(protein):
    assert isinstance(protein.visualize, ProteinVisualizer)
    assert protein.visualize is protein.visualize


def test_importing_sparrow_does_not_import_matplotlib():
    """matplotlib is imported only when a protein is first drawn, so importing sparrow stays fast."""
    script = "import sys, sparrow\nprint('matplotlib' in sys.modules)\n"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_batch_does_not_take_visualize_as_a_value():
    import sparrow

    with pytest.raises(SparrowException, match="not something a Protein computes"):
        sparrow.batch([SEQUENCE], ["visualize"])


@pytest.mark.skipif(not DOCS_PAGE.exists(), reason="the docs are not here, as in an installed copy of sparrow")
def test_the_docs_list_every_analysis_sparrow_knows(protein):
    """docs/protein/visualization.rst has a table of the analyses; each must be in it."""
    text = DOCS_PAGE.read_text()
    missing = [name for name in protein.visualize.available_analyses() if f"``{name}``" not in text]
    assert missing == []


def test_every_analysis_sparrow_knows_can_be_drawn(protein):
    names = [{name: ARGUMENTS.get(name)} for name in protein.visualize.available_analyses()]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        figure, axes = protein.visualize.show_protein(names)
    assert len(axes) == len(names)


# ..............................................................
# What an analysis is drawn as


def test_a_profile_is_a_line_of_the_proteins_own_values(protein):
    figure, axes = protein.visualize.show_protein({"linear_sequence_profile": {"mode": "NCPR", "window_size": 5}})
    (line,) = axes[0].lines
    np.testing.assert_allclose(line.get_ydata(), protein.linear_sequence_profile("NCPR", window_size=5))
    np.testing.assert_array_equal(line.get_xdata(), np.arange(1, LENGTH + 1))
    assert _labels(axes) == ["NCPR"]


def test_a_composition_profile_is_labelled_by_its_residues(protein):
    figure, axes = protein.visualize.show_protein({"linear_composition_profile": {"composition_list": ["Q", "N"]}})
    assert _labels(axes) == ["Fraction N+Q"]


def test_disorder_is_a_line_from_zero_to_one(protein):
    figure, axes = protein.visualize.show_protein("predictor.disorder")
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), protein.predictor.disorder())
    assert axes[0].get_ylim() == (0.0, 1.0)
    assert _labels(axes) == ["Disorder"]


def test_a_mask_is_a_box_over_each_run_of_ones(protein):
    figure, axes = protein.visualize.show_protein({"low_complexity_mask": {"residue_selector": "Q"}})
    mask = protein.low_complexity_mask(residue_selector="Q")
    assert _runs(mask)
    assert _boxes(axes[0]) == _runs(mask)
    assert _labels(axes) == ["LCDs (Q)"]


def test_regions_are_boxes_over_the_same_residues_as_their_mask(protein):
    figure, axes = protein.visualize.show_protein(
        [{"low_complexity_domains": {"residue_selector": "Q"}}, {"low_complexity_mask": {"residue_selector": "Q"}}]
    )
    assert _boxes(axes[0]) == _boxes(axes[1])
    assert len(_boxes(axes[0])) > 0


def test_predicted_idrs_are_boxes_however_they_are_asked_for(protein):
    figure, axes = protein.visualize.show_protein(
        ["predictor.disorder_domains", {"predictor.disorder_domains": {"return_boundaries": True}}]
    )
    boundaries = protein.predictor.disorder_domains(return_boundaries=True)
    expected = [(start + 1, end) for start, end in boundaries]
    assert len(expected) > 1
    assert _boxes(axes[0]) == expected
    assert _boxes(axes[1]) == expected


def test_phosphosites_are_marked_as_sites_however_they_are_asked_for(protein):
    figure, axes = protein.visualize.show_protein(
        [
            "predictor.threonine_phosphorylation",
            {"predictor.threonine_phosphorylation": {"return_sites_only": True}},
            {"predictor.threonine_phosphorylation": {"raw_values": True}},
        ]
    )
    expected = [int(index) + 1 for index in np.flatnonzero(protein.predictor.threonine_phosphorylation())]
    assert expected
    assert _sites(axes[0]) == expected
    assert _sites(axes[1]) == expected
    # the probabilities themselves are a line
    np.testing.assert_allclose(
        axes[2].lines[0].get_ydata(), protein.predictor.threonine_phosphorylation(raw_values=True)
    )
    assert axes[2].get_ylim() == (0.0, 1.0)


def test_the_same_analysis_with_other_arguments_is_drawn_again(protein):
    figure, axes = protein.visualize.show_protein(
        [
            {"linear_sequence_profile": {"mode": "NCPR", "window_size": 5}},
            {"linear_sequence_profile": {"mode": "NCPR"}},
        ]
    )
    first, second = (ax.lines[0].get_ydata() for ax in axes)
    assert not np.allclose(first, second)


def test_an_analysis_of_one_value_is_refused(protein):
    with pytest.raises(SparrowException, match="one value for the whole protein"):
        protein.visualize.show_protein("FCR")


def test_an_analysis_of_the_wrong_length_is_refused(protein):
    with pytest.raises(SparrowException, match=f"values for a protein of {LENGTH} residues"):
        protein.visualize.show_protein({"linear_sequence_profile": {"mode": "NCPR", "end_mode": ""}})


def test_an_analysis_that_is_not_per_residue_is_refused(protein):
    with pytest.raises(SparrowException, match="returns a set"):
        protein.visualize.show_protein("elms")


def test_an_unknown_analysis_is_refused_with_close_names(protein):
    with pytest.raises(SparrowException, match="Did you mean 'predictor.disorder'"):
        protein.visualize.show_protein("predictor.disordr")


def test_an_analysis_that_returns_several_things_is_refused(protein):
    with pytest.raises(SparrowException, match="not one value for each residue"):
        protein.visualize.show_protein({"predictor.dssp_helicity": {"mode": "both"}})


@pytest.mark.parametrize(
    "mask, runs",
    [
        ([0, 1, 1, 0, 1], [(2, 3), (5, 5)]),
        ([1, 1, 1], [(1, 3)]),
        ([0, 0], []),
        ([1, 0, 0, 1], [(1, 1), (4, 4)]),
    ],
)
def test_runs_of_ones_count_from_one_and_include_both_ends(mask, runs):
    assert _analyses._runs_of_ones(np.array(mask)) == runs


def test_fractions_that_happen_to_be_zero_or_one_are_values_not_a_mask():
    assert _analyses._is_mask(np.array([0, 1, 1]))
    assert _analyses._is_mask(np.array([True, False]))
    assert not _analyses._is_mask(np.array([0.0, 1.0, 1.0]))
    assert not _analyses._is_mask(np.array([0, 2, 1]))


# ..............................................................
# Values, sites and domains of your own


def test_signals_are_lines_of_the_values_given(protein):
    values = np.linspace(-1, 1, LENGTH)
    figure, axes = protein.visualize.show_protein(signals={"Conservation": values})
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), values)
    assert _labels(axes) == ["Conservation"]


def test_a_signal_of_the_wrong_length_is_refused(protein):
    with pytest.raises(SparrowException, match="there must be one value for each"):
        protein.visualize.show_protein(signals={"Conservation": np.ones(LENGTH - 1)})


def test_sites_can_be_positions_labelled_positions_or_sites(protein):
    figure, axes = protein.visualize.show_protein(
        sites={
            "Plain": [3, 10, 20],
            "Labelled": {5: "S5", 40: "Q40"},
            "Colored": [Site(7, "K7", color="red"), 9],
        }
    )
    plain, labelled, colored = axes
    assert _sites(plain) == [3, 10, 20]
    assert _sites(labelled) == [5, 40]
    assert sorted(text.get_text() for text in labelled.texts) == ["Q40", "S5"]
    marker_colors = [to_hex(color) for color in colored.collections[1].get_facecolors()]
    assert marker_colors == [to_hex("red"), to_hex("black")]


def test_domains_can_be_pairs_labelled_or_domains(protein):
    figure, axes = protein.visualize.show_protein(
        domains={"Domains": [(1, 10), (20, 30, "B"), Domain(40, 50, "C", color="#ffe699")]}
    )
    ax = axes[0]
    assert _boxes(ax) == [(1, 10), (20, 30), (40, 50)]
    assert sorted(text.get_text() for text in ax.texts) == ["B", "C"]
    colors = sorted(to_hex(patch.get_facecolor()) for patch in ax.patches)
    assert colors == sorted([to_hex("white"), to_hex("white"), "#ffe699"])


def test_overlapping_domains_go_on_as_few_rows_as_they_need(protein):
    domains = [(1, 30, "A"), (10, 20, "B"), (25, 40, "C"), (35, 50, "D"), (60, 70, "E")]
    figure, axes = protein.visualize.show_protein(domains={"Domains": domains})
    # A, C and E cannot share a row with the domains that overlap them, but A
    # and D can, and B and C can
    assert [_boxes(ax) for ax in axes] == [[(1, 30), (35, 50), (60, 70)], [(10, 20), (25, 40)]]
    # only the first row is labelled
    assert _labels(axes) == ["Domains", ""]


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"domains": {"D": [(1,)]}}, "must be \\(start, end\\)"),
        ({"domains": {"D": [(1, 2, "A", "B")]}}, "must be \\(start, end\\)"),
        ({"domains": {"D": [(10, 5)]}}, "before"),
        ({"domains": {"D": [(1, LENGTH + 1)]}}, "ends after the last residue"),
        ({"domains": {"D": "not a list"}}, "must be a list of domains"),
        ({"domains": [(1, 10)]}, "must be a dict"),
        ({"sites": {"S": [0]}}, "residue position"),
        ({"sites": {"S": [LENGTH + 1]}}, "after the last residue"),
        ({"sites": {"S": [3, 3]}}, "already has a site at residue 3"),
        ({"sites": {"S": "3"}}, "must be a list of positions"),
        ({"signals": {"S": "abc"}}, "must be numbers"),
        ({"sequence": "everything"}, "can be 'idrs' or a list"),
        ({"sequence": [(1, 2, 3)]}, "must be \\(start, end\\)"),
        ({"highlights": [(1, LENGTH + 1)]}, "ends after the last residue"),
    ],
)
def test_bad_values_sites_domains_and_regions_are_refused(protein, arguments, message):
    with pytest.raises(SparrowException, match=message):
        protein.visualize.show_protein(**arguments)


def test_tracks_of_your_own_are_drawn_as_made(protein):
    bars = LinearTrack("Bars", protein.linear_sequence_profile("NCPR"), plot_type="bar")
    figure, axes = protein.visualize.show_protein(tracks=[bars])
    assert len(axes[0].patches) == LENGTH


# ..............................................................
# The sequence and highlights


def test_the_whole_sequence_can_be_written(protein):
    figure, axes = protein.visualize.show_protein(sequence=True)
    assert "".join(text.get_text() for text in axes[0].texts) == SEQUENCE


@pytest.mark.parametrize("regions", [[(1, 10), (50, 60)], [[1, 10], [50, 60]], np.array([[1, 10], [50, 60]])])
def test_the_sequence_of_regions_can_be_written(protein, regions):
    figure, axes = protein.visualize.show_protein(sequence=regions)
    assert "".join(text.get_text() for text in axes[0].texts) == SEQUENCE[0:10] + SEQUENCE[49:60]


def test_idrs_are_written_and_highlighted_where_they_are_predicted(protein):
    figure, axes = protein.visualize.show_protein(sequence="idrs", highlights="idrs")
    boundaries = protein.predictor.disorder_domains(return_boundaries=True)
    assert len(boundaries) > 1
    written = "".join(text.get_text() for text in axes[0].texts)
    assert written == "".join(SEQUENCE[start:end] for start, end in boundaries)
    visual = protein.visualize.visual_protein(sequence="idrs", highlights="idrs")
    assert [(highlight.start, highlight.end) for highlight in visual.highlights] == [
        (start + 1, end) for start, end in boundaries
    ]


def test_highlights_can_be_regions_or_highlights(protein):
    from sparrow.visualize.proteinvis import Highlight

    visual = protein.visualize.visual_protein(
        "predictor.disorder", highlights=[(1, 10), Highlight(20, 30, color="blue", alpha=0.3)]
    )
    assert [(highlight.start, highlight.end, highlight.color) for highlight in visual.highlights] == [
        (1, 10, "red"),
        (20, 30, "blue"),
    ]


# ..............................................................
# Order, the figure and saving it


def test_tracks_are_drawn_domains_sites_analyses_signals_tracks_then_sequence(protein):
    figure, axes = protein.visualize.show_protein(
        "predictor.disorder",
        signals={"Signal": np.zeros(LENGTH)},
        sites={"Sites": [5]},
        domains={"Domains": [(1, 10)]},
        tracks=[LinearTrack("Mine", np.ones(LENGTH))],
        sequence=[(1, 10)],
    )
    assert _labels(axes) == ["Domains", "Sites", "Disorder", "Signal", "Mine", ""]


def test_tracks_can_be_drawn_in_an_order_of_your_own(protein):
    figure, axes = protein.visualize.show_protein(
        "predictor.disorder",
        sites={"Sites": [5]},
        domains={"Domains": [(1, 10)]},
        order=("analyses", "sites", "domains"),
    )
    assert _labels(axes) == ["Disorder", "Sites", "Domains"]


@pytest.mark.parametrize(
    "order, message",
    [
        (("analyses",), "were given but order"),
        (("analyses", "domains", "domains"), "at most once"),
        (("analyses", "domains", "colors"), "at most once"),
        ("analyses", "at most once"),
    ],
)
def test_an_order_that_does_not_fit_is_refused(protein, order, message):
    with pytest.raises(SparrowException, match=message):
        protein.visualize.show_protein("predictor.disorder", domains={"Domains": [(1, 10)]}, order=order)


def test_nothing_to_draw_is_refused(protein):
    with pytest.raises(SparrowException, match="nothing to draw"):
        protein.visualize.show_protein()


def test_the_figure_has_the_width_and_name_asked_for(protein):
    figure, axes = protein.visualize.show_protein("predictor.disorder", name="Test", figure_width_in=4.0)
    assert figure.get_size_inches()[0] == pytest.approx(4.0)
    assert "Test" in [text.get_text() for text in figure.texts]


def test_a_visual_protein_can_be_drawn_with_others(protein):
    other = Protein(SEQUENCE[:50])
    proteins = [
        protein.visualize.visual_protein("predictor.disorder", name="Long"),
        other.visualize.visual_protein("predictor.disorder", name="Short"),
    ]
    assert all(isinstance(visual, VisualProtein) for visual in proteins)
    figure, axes = plot_proteins(proteins)
    assert [len(protein_axes) for protein_axes in axes] == [1, 1]


def test_save_protein_saves_editable_text_and_closes_the_figure(protein, tmp_path):
    open_figures = plt.get_fignums()
    path = tmp_path / "protein.pdf"
    protein.visualize.save_protein(path, "predictor.disorder", domains={"Domains": [(1, 10, "A")]})
    data = path.read_bytes()
    assert b"/FontFile2" in data
    assert b"/Type3" not in data
    assert plt.get_fignums() == open_figures


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
    """The code is in private modules, by what it does, and everything public is imported from protein_visualizer."""
    defined = set()
    for module_info in pkgutil.iter_modules(protein_visualizer.__path__):
        module = importlib.import_module(f"{protein_visualizer.__name__}.{module_info.name}")
        for name in _public_names_defined_in(module):
            assert getattr(protein_visualizer, name, None) is getattr(module, name), f"{module.__name__}.{name}"
            defined.add(name)
    assert set(protein_visualizer.__all__) <= defined
