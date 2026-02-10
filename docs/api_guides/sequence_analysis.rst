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

.. code-block:: python

   from sparrow import calculate_parameters

   seq = "MGSQSSRSSSQQQ"
   aa = calculate_parameters.calculate_aa_fractions(seq)
   complexity = calculate_parameters.calculate_seg_complexity(seq)
   hydro = calculate_parameters.calculate_hydrophobicity(seq, mode="KD")

Grammar Feature Vectors
-----------------------

Use ``sparrow.sequence_analysis.grammar`` when you want a single-sequence
feature-vector workflow with optional scramble-based z-scores.

.. code-block:: python

   from sparrow.sequence_analysis.grammar import (
       GrammarPatterningConfig,
       compute_feature_vector,
   )

   cfg = GrammarPatterningConfig(
       backend="kappa_cython",
       num_scrambles=200,
       seed=1,
   )
   vec = compute_feature_vector("MEEEKKKKSSSTTTDDD", patterning_config=cfg)

Patch Primitives
----------------

Patch metrics used by grammar are available directly as reusable primitives:

.. code-block:: python

   from sparrow.sequence_analysis.patching import patch_fraction, rg_patch_fraction

   seq = "AAAAQQRGRGTTTAAAQQ"
   a_patch = patch_fraction(seq, "A")
   rg_patch = rg_patch_fraction(seq)

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

* :func:`sparrow.calculate_parameters.calculate_aa_fractions`
* :func:`sparrow.calculate_parameters.calculate_seg_complexity`
* :func:`sparrow.calculate_parameters.calculate_hydrophobicity`
* :func:`sparrow.calculate_parameters.calculate_linear_hydrophobicity`
* :func:`sparrow.sequence_analysis.grammar.compute_feature_vector`
* :func:`sparrow.sequence_analysis.patching.patch_fraction`
* :func:`sparrow.sequence_analysis.patching.rg_patch_fraction`

.. automodule:: sparrow.calculate_parameters
   :no-index:

.. automodule:: sparrow.sequence_analysis.grammar
   :no-index:

.. automodule:: sparrow.sequence_analysis.patching
   :no-index:
