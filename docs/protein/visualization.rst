Visualization
=============

A ``Protein`` can show its sequence as colored HTML, and plot itself along its
length from its own analyses.

The sequence as colored HTML
----------------------------

``show_sequence`` shows the sequence as HTML, each residue colored by its type,
in blocks - in a Jupyter notebook it renders where you call it.

.. code-block:: python

   from sparrow import Protein

   p = Protein("MEEEKKKKSSSTTTDDDQQQQNNNNGGGGSSSSAAAAPPPP")

   p.show_sequence(bold_residues=["E", "D"])   # colored HTML, rendered in a notebook

   # one unbroken line, no block spacing
   p.show_sequence(blocksize=-1, newline=-1)

   # get the HTML back instead of rendering it
   html = p.show_sequence(return_raw_string=True)

``blocksize`` and ``newline`` both accept ``-1`` to mean "the length of the
sequence": ``blocksize=-1`` suppresses the block spacing and ``newline=-1``
puts everything on one line. ``bold_positions``, ``bold_residues`` and
``opaque_positions`` pick out residues; ``colors`` replaces the default color of
each residue (:data:`~sparrow.data.amino_acids.AA_COLOR`).

The same function works on any sequence string, as
:func:`sparrow.visualize.sequence_visuals.show_sequence`.

Plotting the protein along its length
-------------------------------------

``p.visualize`` draws the protein along its length, residue 1 on the left, as
a stack of tracks: any of its own analyses that give something for each
residue, and values, sites and domains of your own. Name the analyses as
:func:`sparrow.batch` names them - a name on its own for its defaults, or in a
dict with its arguments - and give your own by the label of their track:

.. code-block:: python

   from sparrow import Protein

   p = Protein(sequence)

   figure, axes = p.visualize.show_protein(
       [
           "predictor.disorder",
           {"linear_sequence_profile": {"mode": "NCPR", "window_size": 9}},
           "predictor.threonine_phosphorylation",
       ],
       signals={"Conservation": conservation},             # one value for each residue
       sites={"Mutations": {120: "R120W", 400: "Q400*"}},  # residues, labelled or not
       domains={"Folded": [(6, 91, "KIX"), (158, 238, "ABD1")]},
       sequence="idrs",                                    # write each IDR's sequence
       highlights="idrs",                                  # and shade each IDR
       name="Med15",
   )

   # draw and save in one go, with the text editable in Illustrator
   p.visualize.save_protein("med15.pdf", ["predictor.disorder"], domains={"Folded": [(6, 91, "KIX")]})

Each analysis is computed by the protein and cached, as when you call it
yourself, and drawn according to what it returns:

* a value for each residue - ``predictor.disorder``, ``predictor.pLDDT``, a
  linear profile - is a line on a y-axis of its own, which runs from 0 to 1
  for a probability;
* a 1 or 0 for each residue - ``predictor.binary_disorder``, the DSSP classes,
  ``predictor.transmembrane_regions``, ``low_complexity_mask`` - is a box over
  each run of 1s, apart from the phosphorylation predictors', whose 1s are
  each marked as a site;
* regions - ``low_complexity_domains``, ``plaac_prion_like_domains``, or
  ``predictor.disorder_domains`` with ``return_boundaries=True`` - are boxes.

An analysis that gives one value for the whole protein, such as ``FCR``, cannot
be drawn along it, and is refused. ``p.visualize.available_analyses()`` lists
the analyses sparrow knows how to label, below; any other that gives a value
for each residue is drawn too, labelled with its name.

.. list-table::
   :header-rows: 1
   :widths: 44 28 28

   * - Analysis
     - Drawn as
     - Labelled
   * - ``predictor.disorder``
     - a line, 0 to 1
     - Disorder
   * - ``predictor.binary_disorder``, ``predictor.disorder_domains``
     - boxes
     - IDRs
   * - ``predictor.pLDDT``
     - a line, 0 to 100
     - pLDDT
   * - ``predictor.pLDDT2``
     - a line, 0 to 100
     - pLDDT2
   * - ``predictor.dssp_helicity``, ``predictor.dssp_extended``,
       ``predictor.dssp_coil``
     - boxes, or with ``mode='probability'`` a line, 0 to 1
     - Helix, Strand, Coil
   * - ``predictor.transmembrane_regions``
     - boxes
     - TM
   * - ``predictor.mitochondrial_targeting_sequence``
     - boxes
     - MTS
   * - ``predictor.nuclear_export_signal``, ``predictor.nuclear_import_signal``
     - a line, 0 to 1
     - NES, NLS
   * - ``predictor.transactivation_domains``
     - a line, 0 to 1
     - TAD
   * - ``predictor.pscore``
     - a line
     - PScore
   * - ``predictor.serine_phosphorylation``,
       ``predictor.threonine_phosphorylation``,
       ``predictor.tyrosine_phosphorylation``
     - sites, or with ``raw_values=True`` a line, 0 to 1
     - pS, pT, pY
   * - ``linear_sequence_profile``
     - a line
     - its ``mode``: FCR, NCPR, Fraction aromatic, ...
   * - ``linear_composition_profile``
     - a line
     - Fraction and its residues, such as Fraction N+Q
   * - ``linear_property_profile``
     - a line
     - its ``mode``, the AAindex property
   * - ``low_complexity_mask``, ``low_complexity_domains``
     - boxes
     - LCDs and the residues, such as LCDs (Q)
   * - ``plaac_prion_like_domains``
     - boxes
     - PLDs

Your own data goes in by the label of its track:

* ``signals`` - values, one for each residue, each set drawn as a line; a
  residue with no value can be NaN;
* ``sites`` - residues, as a list of positions, or a dict of positions and
  their labels, or :class:`~sparrow.visualize.proteinvis.Site` for a color;
* ``domains`` - regions, each ``(start, end)``, ``(start, end, label)``, or a
  :class:`~sparrow.visualize.proteinvis.Domain` for its color and the rest.
  Domains that overlap go on rows of their own, as few as they fit on.

Positions count from 1, and a region from ``start`` to ``end`` includes both.
The regions metapredict and the low-complexity analyses return count from 0
and leave out their end; ``visualize`` converts them for you.

``sequence`` writes the sequence of the whole protein (``True``), of each
predicted IDR (``'idrs'``), or of a list of ``(start, end)`` regions, each
within its region, and ``highlights`` shades the predicted IDRs or a list of
regions behind every track. The tracks are drawn top to bottom in the order
domains, sites, analyses, signals, tracks, sequence; ``order`` gives another,
such as ``order=("analyses", "domains")``.

For full control of a track - bars for NCPR, a color, fixed limits - make it
yourself with :mod:`sparrow.visualize.proteinvis` and pass it in ``tracks``.
``visual_protein`` builds the tracks without drawing them, to change them, or
to plot the protein beside others with
:func:`~sparrow.visualize.proteinvis.plot_proteins`:

.. code-block:: python

   from sparrow.visualize.proteinvis import plot_proteins

   proteins = [Protein(s).visualize.visual_protein("predictor.disorder", name=name) for name, s in sequences.items()]
   figure, axes = plot_proteins(proteins)

:doc:`../power_user/plotting` covers the tracks, and the figure's size, style
and saving. The notebook ``demos/protein_visualization/protein_viz_2.ipynb``,
in sparrow's repository, walks through examples of everything on this page.

Reference
---------

.. automethod:: sparrow.protein.Protein.show_sequence

.. autoproperty:: sparrow.protein.Protein.visualize

.. autoclass:: sparrow.visualize.protein_visualizer.ProteinVisualizer
   :no-members:
   :no-inherited-members:
   :no-undoc-members:

.. autosummary::
   :nosignatures:

   sparrow.visualize.protein_visualizer.ProteinVisualizer.show_protein
   sparrow.visualize.protein_visualizer.ProteinVisualizer.save_protein
   sparrow.visualize.protein_visualizer.ProteinVisualizer.visual_protein
   sparrow.visualize.protein_visualizer.ProteinVisualizer.available_analyses

.. automethod:: sparrow.visualize.protein_visualizer.ProteinVisualizer.show_protein

.. automethod:: sparrow.visualize.protein_visualizer.ProteinVisualizer.save_protein

.. automethod:: sparrow.visualize.protein_visualizer.ProteinVisualizer.visual_protein

.. automethod:: sparrow.visualize.protein_visualizer.ProteinVisualizer.available_analyses
