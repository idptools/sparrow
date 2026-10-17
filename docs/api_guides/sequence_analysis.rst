Sequence Analysis (Functional Workflow)
=======================================

Use this workflow when you want direct functions and analysis objects in scripts
or pipelines without routing every operation through ``Protein`` methods.

When to Prefer Functional APIs
------------------------------

* You are processing many sequences in a pipeline with explicit intermediate data.
* You want to call lower-level utility functions directly.
* You need module-level control over alignment and scoring settings.

Minimal End-to-End Alignment Example
------------------------------------

.. code-block:: python

   from sparrow import Protein
   from sparrow.sequence_analysis.alignment import SequenceAlignment

   proteins = {
       "seq1": Protein("MGSQSSRSSSQQQ"),
       "seq2": Protein("MGSQSSRNNNQQQ"),
       "seq3": Protein("MGSQSSRSSSAAA"),
   }

   msa = SequenceAlignment(proteins, scoring_matrix="BLOSUM62")
   alignment = msa.alignment
   msa.save_msa("example_alignment.fasta")

Direct Parameter Calculations
-----------------------------

The calculations behind the ``Protein`` properties live in
:mod:`sparrow.properties`, and every one of them works on a batch: pass a list of
sequences and get back one value per sequence. A single string is a batch of one.

.. code-block:: python

   from sparrow.properties import charge, composition

   sequences = ["MGSQSSRSSSQQQ", "MEEEKKKKSSSTTTDDD"]
   fractions = composition.amino_acid_fractions(sequences)   # (2, 20)
   complexity = composition.seg_complexity(sequences)        # (2,)
   hydro = composition.hydrophobicity(sequences)             # Kyte-Doolittle mean
   pI = charge.isoelectric_point(sequences)

When you need several properties of the same batch, count it once and pass the
counts instead of the sequences:

.. code-block:: python

   counts = composition.count_residues(sequences)
   fcr = composition.fraction_charged(counts)
   ncpr = composition.net_charge_per_residue(counts)

NARDINI+ Feature Vectors
------------------------

Patterning and compositional feature vectors -- NARDINI+, and assignment to the
human GIN grammars -- live in :mod:`sparrow.nardini`. See :doc:`nardini`.

Patch Primitives
----------------

The patch metrics used by NARDINI+ are batch functions too, in
:mod:`sparrow.properties.patches`:

.. code-block:: python

   from sparrow.properties.patches import patch_fraction, rg_patch_fraction

   sequences = ["AAAAQQRGRGTTTAAAQQ", "GRGRGGYGGSSSS"]
   a_patch = patch_fraction(sequences, "A")      # one fraction per sequence
   rg_patch = rg_patch_fraction(sequences)

Protein vs Functional API
-------------------------

* Prefer ``Protein`` for interactive analysis and memoized repeated queries.
* Prefer functional modules for explicit data flow and pipeline composition.
* Mix both styles when needed: create ``Protein`` only where accessor behavior
  or object-level caching adds value.

Reference: Alignment Module
---------------------------

.. currentmodule:: sparrow.sequence_analysis.alignment

.. automodule:: sparrow.sequence_analysis.alignment
   :no-index:

Reference: Parameter Functions
------------------------------

Key function entry points:

* :func:`sparrow.properties.composition.amino_acid_fractions`
* :func:`sparrow.properties.composition.seg_complexity`
* :func:`sparrow.properties.composition.hydrophobicity`
* :func:`sparrow.properties.encoding.scale_values`
* :func:`sparrow.properties.patches.patch_fraction`
* :func:`sparrow.properties.patches.rg_patch_fraction`

Sequence properties
-------------------

The single implementation behind the corresponding
:class:`~sparrow.protein.Protein` properties *and* the matching NARDINI+
compositional features, so the two cannot disagree. Every function takes a batch.

.. automodule:: sparrow.properties
   :no-index:

.. automodule:: sparrow.properties.composition
   :members:

.. automodule:: sparrow.properties.charge
   :members: charge_at_pH, isoelectric_point, DEFAULT_PH, PI_THRESHOLD

.. automodule:: sparrow.properties.encoding
   :members:

.. automodule:: sparrow.properties.patches
   :members:
