Protein API (Core Workflow)
===========================

The :class:`sparrow.protein.Protein` class is the primary user entrypoint.
It wraps sequence-derived calculations with lazy evaluation and memoization, so
values are computed only when requested and cached for repeated access.

Lifecycle and Accessors
-----------------------

``Protein`` exposes three major accessor namespaces that are created lazily:

* ``protein.predictor`` for predictor models.
* ``protein.polymeric`` for polymer-related metrics.
* ``protein.plugin`` for community contributed analyses.
* ``protein.elms`` for ELM motif annotations.

Basic Usage Examples
--------------------

Construction and optional validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sparrow import Protein

   p = Protein("MGSQSSRSSSQQQQQQQ")
   p_validated = Protein("MGSQXU--", validate=True)  # sequence normalization enabled

Composition and charge metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   p = Protein("MGSQSSRSSSQQQQQQQ")

   aa_fractions = p.amino_acid_fractions
   fcr = p.FCR
   ncpr = p.NCPR
   frac_pos = p.fraction_positive
   frac_neg = p.fraction_negative

Patterning and clustering
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   p = Protein("MEEEKKKKSSSTTTDDD")

   default_kappa = p.kappa
   default_scd = p.SCD

   custom_kappa = p.compute_kappa_x(group1="ED", group2="KR", window_size=6, flatten=True)
   acidic_iwd = p.compute_iwd(target_residues=["D", "E"])
   q_patch = p.compute_patch_fraction(residue_selector="Q")
   rg_patch = p.compute_rg_patch_fraction()

Profiles and low-complexity domains
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   p = Protein("QQQQQQAASSSSTTTTQQQQQ")

   ncpr_track = p.linear_sequence_profile(mode="NCPR", window_size=8)
   q_track = p.linear_composition_profile(composition_list=["Q"], window_size=8)
   lcds = p.low_complexity_domains(mode="holt", residue_selector="Q", minimum_length=8)

Curated API Index
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   sparrow.protein.Protein.sequence
   sparrow.protein.Protein.FCR
   sparrow.protein.Protein.NCPR
   sparrow.protein.Protein.kappa
   sparrow.protein.Protein.SCD
   sparrow.protein.Protein.compute_kappa_x
   sparrow.protein.Protein.compute_iwd
   sparrow.protein.Protein.compute_patch_fraction
   sparrow.protein.Protein.compute_rg_patch_fraction
   sparrow.protein.Protein.linear_sequence_profile
   sparrow.protein.Protein.low_complexity_domains
   sparrow.protein.Protein.predictor
   sparrow.protein.Protein.polymeric
   sparrow.protein.Protein.plugin
   sparrow.protein.Protein.elms

Gotchas and Edge Behavior
-------------------------

* ``kappa``/``compute_kappa_x`` can return ``-1`` for non-computable cases
  (for example, too-short sequences or absent group residues).
* Accessor objects (``predictor``, ``polymeric``, ``plugin``) are created on
  first access and then reused.
* Repeated calls to many methods return cached values where possible.

Full Class Reference
--------------------

.. autoclass:: sparrow.protein.Protein
   :no-index:
