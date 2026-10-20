Visualization
=============

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

Reference
---------

.. automethod:: sparrow.protein.Protein.show_sequence
