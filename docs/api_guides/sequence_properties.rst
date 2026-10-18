Sequence Properties in Batches (``sparrow.properties``)
=======================================================

Every sequence parameter sparrow calculates -- from FCR and kappa to the patch
fractions, SCD and the linear profiles -- is calculated in one place,
:mod:`sparrow.properties`. :class:`~sparrow.protein.Protein` and
:mod:`sparrow.nardini` both call into it rather than carrying arithmetic of
their own.

To get ``Protein`` objects with these values already computed, use
:func:`sparrow.batch` (:doc:`many_sequences`), which calls these functions for
you. This page is for when you want the arrays themselves.

Every function there takes a **batch** of sequences and returns an array with
one value per sequence. A ``Protein`` is simply a batch of one, so a whole
proteome can be analysed without making a ``Protein`` for each sequence, and the
numbers are the same as the ``Protein`` ones to the last bit.

Quick start
-----------

:func:`~sparrow.properties.compute` gives any of the ``Protein`` properties for a
whole batch, by name:

.. code-block:: python

   from sparrow.properties import compute

   sequences = ["MEEEKKKKSSSTTTDDDQQQQNNNN", "GRGRGGYGGRGGYGGSRGGYGG", "SPSPSPTPSPSP"]

   table = compute(sequences, ["FCR", "NCPR", "kappa", "SCD", "hydrophobicity"])
   table["kappa"]            # numpy array, one value per sequence

   everything = compute(sequences)   # all of them

The names are the ``Protein`` property names, listed in
:data:`~sparrow.properties.PROPERTY_NAMES`, and ``table[name][i]`` is exactly
``getattr(Protein(sequences[i]), name)``. The batch is encoded and counted once,
however many properties are asked for. For the 24,508 human IDRs, all 20
properties take 0.28 s this way, against 7.3 s with one ``Protein`` per IDR.

Because the result is a dictionary of arrays, writing it out needs nothing more
than NumPy:

.. code-block:: python

   import numpy as np

   names = list(table)
   np.savetxt("properties.tsv", np.column_stack([table[name] for name in names]),
              delimiter="\t", header="\t".join(names), comments="")

The batch functions
-------------------

Every function can also be called on its own, and imported from the package
itself. The ones that take arguments are the batch counterparts of the
``Protein`` methods:

.. list-table::
   :header-rows: 1

   * - ``Protein``
     - ``sparrow.properties``
   * - ``p.compute_residue_fractions(residues)``
     - :func:`~sparrow.properties.composition.fraction_in_set`
   * - ``p.charge_at_pH(pH, normalize)``
     - :func:`~sparrow.properties.charge.charge_at_pH`
   * - ``p.compute_kappa_x(group1, group2, window_size, flatten)``
     - :func:`~sparrow.properties.patterning.kappa_x`
   * - ``p.compute_iwd(residues)``
     - :func:`~sparrow.properties.patterning.iwd`
   * - ``p.compute_iwd_charged_weighted(charge)``
     - :func:`~sparrow.properties.patterning.iwd_charged_weighted`
   * - ``p.compute_bivariate_iwd_charged_weighted()``
     - :func:`~sparrow.properties.patterning.bivariate_iwd_charged_weighted`
   * - ``p.compute_SCD_x(group1, group2)``
     - :func:`~sparrow.properties.patterning.scd`
   * - ``p.compute_SHD_custom(scale)``
     - :func:`~sparrow.properties.patterning.shd`
   * - ``p.compute_patch_fraction(residues, ...)``
     - :func:`~sparrow.properties.patches.patch_fraction`
   * - ``p.linear_sequence_profile(mode, ...)``
     - :func:`~sparrow.properties.profiles.window_profile`
   * - ``p.linear_composition_profile(residues, ...)``
     - :func:`~sparrow.properties.profiles.fraction_profile`
   * - ``p.linear_property_profile(index, ...)``
     - :func:`~sparrow.properties.profiles.scale_profile`

.. code-block:: python

   from sparrow.properties import fraction_in_set, iwd, kappa_x, patch_fraction

   qn = fraction_in_set(sequences, ["Q", "N"])
   ed_rk = kappa_x(sequences, ["E", "D"], ["R", "K"], window_size=6, flatten=True)
   aromatic_clustering = iwd(sequences, ["F", "W", "Y"])
   s_patches = patch_fraction(sequences, "S")

The linear profiles return a list with one profile per sequence -- a list rather
than one array, because sequences of different lengths have profiles of
different lengths. :func:`~sparrow.properties.profiles.window_profile` profiles
*any* of the batch properties: every window of every sequence is handed to the
property function as one batch.

.. code-block:: python

   from sparrow.properties import composition, patterning, profiles

   fcr_profiles = profiles.window_profile(sequences, composition.fraction_charged, window_size=8)
   kappa_profiles = profiles.window_profile(sequences, patterning.kappa, window_size=12)

Counting once
^^^^^^^^^^^^^

Most of the composition properties are a few array operations on the batch's
residue counts. To compute several of them without counting the sequences each
time, count once and pass the counts:

.. code-block:: python

   from sparrow.properties import count_residues, fraction_charged, molecular_weight, net_charge_per_residue

   counts = count_residues(sequences)
   fcr = fraction_charged(counts)
   ncpr = net_charge_per_residue(counts)
   mass = molecular_weight(counts)

The functions in :mod:`~sparrow.properties.composition` and
:mod:`~sparrow.properties.charge` take counts; the rest take an encoded batch
from :func:`~sparrow.properties.encoding.encode_batch`, which likewise saves encoding the
sequences more than once. :func:`~sparrow.properties.compute` does both for you.

What a batch can be
-------------------

* **One sequence or many.** A single string is a batch of one; otherwise any
  sequence of strings. Lower-case residues are upper-cased.
* **Only the 20 standard amino acids.** Anything else raises a
  ``SparrowException`` that names the residue and, in a batch, the sequence.
  ``Protein(seq, validate=True)`` converts the common non-standard residues, and
  :func:`sparrow.properties.encoding.validate_protein_sequence` checks a sequence
  without computing anything. SHD is the one exception: it reads its scale
  character by character, so it works for any alphabet the scale covers.
* **Empty sequences** have every fraction, their SEG complexity, molecular weight,
  SCD, SHD and R/K and E/D ratios equal to 0.0, a mean hydrophobicity or PPII
  propensity of NaN (there is nothing to average), a kappa of -1, and an
  isoelectric point of 7.0 -- as does any sequence with no titratable residues,
  whose charge is zero at every pH. A profile needs a sequence at least as long as
  its window.

The same answer alone or in a batch
-----------------------------------

A sequence gets the same value -- to the last bit -- on its own, in a batch of
ten, or in a batch of a million. That is what lets ``Protein`` call these
functions with one sequence and still agree exactly with NARDINI, which calls
them with thousands. Three things make it so:

* Calculations that run along a sequence group the batch by exact length and
  score each group as one array, rather than padding every sequence to a common
  length; a padded row would be summed in a different order.
* Sums over residues are taken one row at a time, never as a matrix product,
  whose order of additions BLAS chooses by the number of rows.
* Count matrices are kept row-major, because NumPy sums the row of a row-major
  array pairwise but the row of a column-major one term by term.

``sparrow/tests/test_properties_batch.py`` checks every batch function against
one-sequence-at-a-time on batches of about 10 and 3,000 sequences, and
``sparrow/tests/test_properties_compute.py`` checks every
:func:`~sparrow.properties.compute` name against its ``Protein`` property.

Modules
-------

.. list-table::
   :header-rows: 1

   * - Module
     - Holds
   * - :mod:`~sparrow.properties.encoding`
     - validation, sequences as integer codes, residue-set lookup tables
   * - :mod:`~sparrow.properties.composition`
     - the residue count matrix, fractions, residue sets, ratios, mean scales,
       molecular weight, SEG complexity
   * - :mod:`~sparrow.properties.charge`
     - charge at any pH, the isoelectric point
   * - :mod:`~sparrow.properties.patches`
     - patch fractions
   * - :mod:`~sparrow.properties.windows`
     - sliding-window sums and counts
   * - :mod:`~sparrow.properties.patterning`
     - kappa, delta, the patterning percentile, IWD, SCD, SHD (see
       :doc:`patterning`)
   * - :mod:`~sparrow.properties.profiles`
     - linear profiles, their end modes and smoothing

Where things used to live
-------------------------

These modules were removed when their calculations moved here, and their import
paths were not kept:

.. list-table::
   :header-rows: 1

   * - Removed
     - Now
   * - ``sparrow.calculate_parameters``, ``sparrow.sequence_analysis.physical_properties``
     - :mod:`~sparrow.properties.composition`, :mod:`~sparrow.properties.charge`;
       ``calculate_linear_hydrophobicity`` is
       :func:`~sparrow.properties.encoding.scale_values`
   * - ``sparrow.sequence_analysis.patching``
     - :mod:`~sparrow.properties.patches`
   * - ``sparrow.patterning`` (``kappa``, ``iwd``, ``scd``, ``patterning``,
       ``vectorized``)
     - :mod:`~sparrow.properties.patterning` (see :doc:`patterning` for the
       renamed functions)
   * - ``sparrow.tools.track_tools``
     - :mod:`~sparrow.properties.profiles`
   * - the validators in ``sparrow.tools.general_tools``
     - :mod:`~sparrow.properties.encoding`

Reference
---------

.. py:module:: sparrow.properties

.. autofunction:: sparrow.properties.compute

.. autodata:: sparrow.properties.PROPERTY_NAMES
   :no-value:

.. automodule:: sparrow.properties.composition
   :members:
   :exclude-members: counts, lengths

.. automodule:: sparrow.properties.charge
   :members: charge_at_pH, isoelectric_point, DEFAULT_PH, PI_THRESHOLD

.. automodule:: sparrow.properties.encoding
   :members:
   :exclude-members: codes, lengths

.. automodule:: sparrow.properties.patches
   :members:

.. automodule:: sparrow.properties.profiles
   :members:

The residue sets and scales these functions use:

.. automodule:: sparrow.data.amino_acids
   :members: VALID_AMINO_ACIDS, POS, NEG, CHARGE, ARO, ALI, POLAR, NARDINI_POLAR,
             EXPANDING, DISORDER_PROMOTING, TITRATABLE_BASES, TITRATABLE_ACIDS,
             AA_PKA, AA_MOLECULAR_WEIGHT, AA_hydro_KD, AA_hydro_KD_normalized,
             AA_hydro_HPS, AA_PPII_HILSER
