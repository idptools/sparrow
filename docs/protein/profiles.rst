Linear Profiles
===============

A linear profile is a sequence parameter computed in a window that slides along
the sequence one residue at a time, giving one value per residue - the way to
*see* where along a sequence its charge, hydrophobicity or composition lies.
:doc:`../examples` plots one.

.. code-block:: python

   from sparrow import Protein

   p = Protein("MEEEKKKKSSSTTTDDDQQQQNNNNGGGGSSSSLLLVVVAAA")

   p.linear_sequence_profile(mode="NCPR", window_size=7)
   p.linear_composition_profile(["Q", "N"], window_size=8)
   p.linear_property_profile("hydropathy-kyte-1982", window_size=9)   # an AAindex scale

The three methods take the same ``window_size``, ``end_mode`` and ``smooth``
arguments, and differ in what each window is scored by:

* ``linear_sequence_profile`` - one of the built-in parameters, chosen by
  ``mode``: ``"FCR"``, ``"NCPR"``, ``"aromatic"``, ``"aliphatic"``,
  ``"polar"``, ``"proline"``, ``"positive"``, ``"negative"``,
  ``"hydrophobicity"`` (Kyte-Doolittle), ``"seg-complexity"`` or ``"kappa"``.
* ``linear_composition_profile`` - the fraction of the window made up of any
  residues you choose.
* ``linear_property_profile`` - the mean of any of the 569 published
  amino-acid scales sparrow ships from the AAindex database, named by a
  readable identifier such as ``"hydropathy-kyte-1982"`` or by its AAindex
  accession. Every scale is listed, with its source, on
  :doc:`../power_user/aaindex`.

The ends
--------

A window of ``w`` residues has ``len(sequence) - w + 1`` positions, so a profile
is shorter than its sequence unless its ends are filled in. ``end_mode`` says
how:

* ``"extend-ends"`` (the default) copies the first and last values outwards,
  so the profile has one value per residue;
* ``"zero-ends"`` fills them with zero;
* ``""`` leaves them off, giving the windows alone.

``smooth``, if given, smooths the profile with a Savitzky-Golay filter (a
third-order polynomial) over that many residues, which must be odd.

For a whole list of sequences at once, :mod:`sparrow.properties.profiles` gives
the same profiles, and can profile *any* of sparrow's batch properties; see
:doc:`../power_user/properties`.

Reference
---------

.. autosummary::
   :nosignatures:

   sparrow.protein.Protein.linear_sequence_profile
   sparrow.protein.Protein.linear_composition_profile
   sparrow.protein.Protein.linear_property_profile

.. automethod:: sparrow.protein.Protein.linear_sequence_profile

.. automethod:: sparrow.protein.Protein.linear_composition_profile

.. automethod:: sparrow.protein.Protein.linear_property_profile
