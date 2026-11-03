Plotting Proteins Along Their Length
====================================

:mod:`sparrow.visualize.proteinvis` draws a protein along its length - its
domains, any per-residue values you have for it, heatmaps of several values at
once, and the sequence of chosen regions - as a stack of tracks that share one
residue axis, residue 1 on the left. Several proteins can be drawn in one
figure, on the same scale.

For a ``Protein``, ``protein.visualize`` builds these tracks from the protein's
own analyses, with values, sites and domains of your own, in one call; see
:doc:`../protein/visualization`. This page covers the tracks themselves, for a
figure built by hand.

You build a :class:`~sparrow.visualize.proteinvis.VisualProtein`, add the
tracks to draw for it, top to bottom, and pass it to
:func:`~sparrow.visualize.proteinvis.plot_proteins`:

.. code-block:: python

   from sparrow import Protein
   from sparrow.visualize.proteinvis import (
       DomainTrack, GridTrack, LinearTrack, SequenceTrack, VisualProtein,
       plot_proteins, save_figure,
   )

   sequence = "MSAAPVQDKDTLSNAERAKNVNGLLQVLMDINTLNGGSSDTADKIRIHAKNFEAALFAKSSSKKEYMDSMNEKVAVMRNTYNTRKNAVTAAAANNNIKPV"
   protein = Protein(sequence)

   visual = VisualProtein("Med15 (1-100)", sequence=sequence)

   folded = DomainTrack("Folded")
   folded.add_domain(6, 91, label="KIX", color="#ffe699")
   visual.add_track(folded)

   visual.add_track(LinearTrack("Disorder", protein.predictor.disorder(), color="red", y_limits=(0, 1)))
   visual.add_track(LinearTrack("NCPR", protein.linear_sequence_profile("NCPR", window_size=9), plot_type="bar"))

   visual.add_track(GridTrack(
       "Composition",
       [protein.linear_sequence_profile("FCR", window_size=9),
        protein.linear_composition_profile(["Q", "N"], window_size=9)],
       colormaps=["Greys", "Greens"],
       limits=[(0, 0.6), (0, 0.6)],
       row_names=["FCR", "Q + N"],
   ))

   region = SequenceTrack("Sequence")
   region.add_region(1, 40)
   visual.add_track(region)

   visual.add_highlight(1, 40)

   figure, axes = plot_proteins(visual)
   save_figure(figure, "med15.pdf")

The tracks
----------

:class:`~sparrow.visualize.proteinvis.DomainTrack`
   Boxes on a line that stands for the chain: folded domains, IDRs, motifs.
   Add each with ``add_domain(start, end, ...)``, with a label written in,
   above or below its box. Domains in one track cannot overlap, so put
   overlapping ones - an IDR and the motifs within it, say - on tracks of their
   own.

:class:`~sparrow.visualize.proteinvis.SiteTrack`
   Single residues - phosphosites, mutations, cross-links - each marked by a
   stem up to a dot above the line for the chain. Add each with
   ``add_site(position, label, color)``; the label is written above its dot.
   A track holds one site at a residue, so put a second on a track of its own.

:class:`~sparrow.visualize.proteinvis.LinearTrack`
   One value per residue - a linear profile, a disorder score - as a line or as
   bars, on a y-axis of its own, so the values are read off the plot. Give the
   units in the track's name. The y-axis runs over the values, rounded
   outwards, unless you give ``y_limits``; a residue with no value can be NaN,
   and the line breaks there.

:class:`~sparrow.visualize.proteinvis.GridTrack`
   Several per-residue values at once, each a row of a heatmap with its own
   colormap and limits: a compact way to show many profiles. The values are an
   array of shape ``(rows, residues)``, which a list of profiles is.

:class:`~sparrow.visualize.proteinvis.SequenceTrack`
   The sequence of chosen regions, each written within its region in blocks of
   ten, colored by residue as ``show_sequence`` colors it. A residue is usually
   far narrower than a letter, so the letters are not aligned to their
   residues: each region's sequence wraps onto as many lines as it needs to
   stay within the region, so it lies within a highlight over the same
   residues. A line holds as many blocks as fit across the region, or, in a
   region too narrow for one block, as many residues as fit, and the lines are
   centered across the region as one column, so the blocks line up from one
   line to the next. The track is as tall as the region with the most lines, so narrow
   regions, or a long protein in a narrow figure, make it taller. Only a region
   narrower than one letter cannot hold its sequence;
   :func:`~sparrow.visualize.proteinvis.plot_proteins` warns of it, and a
   smaller ``font_size_pt`` or a wider figure fixes it.

:meth:`VisualProtein.add_highlight <sparrow.visualize.proteinvis.VisualProtein.add_highlight>`
adds a band of color over a run of residues, behind every one of the protein's
tracks, to guide the eye from one track to the next.

Positions count from 1, and a region from ``start`` to ``end`` includes both,
as in ``Protein``'s own regions. metapredict's domain boundaries count from 0
and leave out their end, so the domains it returns are added as
``add_domain(start + 1, end)``.

Size and style
--------------

The figure is as wide as ``figure_width_in`` (6.5 inches by default) and as
tall as its tracks, each the height it was made with, in inches
(``height_in``, or ``row_height_in`` for each row of a grid); a sequence track
is as tall as its region with the most lines. The space for the labels is
measured, so every label fits, and the track labels line up in one column.

Every label is in Arial, at ``font_size_pt`` (6 point by default), with black
lines and no grid; the sequence is written in DejaVu Sans Mono, which is
monospaced and ships with matplotlib. These settings apply only while the
figure is drawn and saved, so the rest of your session is untouched.

``plot_proteins`` returns the figure and, for each protein, the axes of each of
its tracks, so you can change anything matplotlib can afterwards.

Saving
------

Save with :func:`~sparrow.visualize.proteinvis.save_figure` rather than
``figure.savefig``. matplotlib decides how to save text when it saves, and
unless your own settings say otherwise it saves a PDF's text as shapes, which
Illustrator cannot edit. ``save_figure`` saves it as text, and takes anything
``savefig`` does:

.. code-block:: python

   save_figure(figure, "med15.pdf")
   save_figure(figure, "med15.png", dpi=600)

A heatmap is saved as an image, each cell a block of identical pixels, at least
600 to the inch, so in a PDF each cell lines up exactly with its residue.
Preview smooths any image it enlarges, whatever the PDF asks, and with a pixel
for each residue that blurs the cells into one another; with blocks, the
smoothing reaches only the very edge of each cell, so the heatmap is sharp in
Preview too.

Reference
---------

Every class and function is documented in full in
:mod:`sparrow.visualize.proteinvis`.
