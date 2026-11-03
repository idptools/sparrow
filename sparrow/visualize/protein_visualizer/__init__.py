"""Plots of a Protein along its length, drawn from its own analyses: ``Protein.visualize``.

``protein.visualize`` draws a protein with :mod:`sparrow.visualize.proteinvis`
from any analysis the protein can compute for each residue, named as
:func:`sparrow.batch` names it, together with values, sites and domains of
your own:

.. code-block:: python

    from sparrow import Protein

    protein = Protein(sequence)
    figure, axes = protein.visualize.show_protein(
        ["predictor.disorder", {"linear_sequence_profile": {"mode": "NCPR", "window_size": 9}}],
        domains={"Folded": [(6, 91, "KIX")]},
        sites={"Phosphosites": [12, 45, 78]},
        signals={"Conservation": conservation},
        sequence="idrs",
        highlights="idrs",
    )
    protein.visualize.save_protein("protein.pdf", ["predictor.disorder"])

How an analysis is drawn follows from what it returns:

* a value for each residue, such as ``predictor.disorder`` or
  ``linear_sequence_profile``, is a line on a y-axis of its own;
* a 1 or 0 for each residue, such as ``predictor.binary_disorder`` or
  ``low_complexity_mask``, is a box over each run of 1s, apart from the
  phosphorylation predictors', whose 1s are each marked as a site;
* a list of regions, such as ``low_complexity_domains`` or
  ``plaac_prion_like_domains``, is a box over each region.

An analysis that gives one value for the whole protein, such as ``FCR``, cannot
be drawn along it, and is refused.
"""

# Protein.visualize in three modules, by what each does. Every public name is
# imported here, so everything is imported from
# sparrow.visualize.protein_visualizer, never from the modules themselves:
#
# * _visualizer - ProteinVisualizer, the class Protein.visualize returns, and
#   its public methods
# * _inputs - reading the order of the kinds of track, and the signals, sites,
#   domains, sequence and highlights given by hand, into tracks
# * _analyses - how each of sparrow's analyses is labelled and drawn, and the
#   tracks made from what it returns
#
# The figure itself is drawn by sparrow.visualize.proteinvis.

from sparrow.visualize.protein_visualizer._inputs import (
    GROUPS,
    DomainsInput,
    HighlightsInput,
    RegionsInput,
    SequenceInput,
    SitesInput,
)
from sparrow.visualize.protein_visualizer._visualizer import ProteinVisualizer

__all__ = ["GROUPS", "ProteinVisualizer"]
