The Protein Object
==================

:class:`sparrow.protein.Protein` is the one object you need. Create it from a
sequence and read everything off it -- parameters, properties, deep-learning
predictions, polymer dimensions, and visualizations. Values are computed lazily
on first access and cached, so you only pay for what you use.

This page lists **everything you can do with a protein, organized by what you
want to compute**. The deep-learning predictors and polymer-model properties are
reached through the ``predictor`` and ``polymeric`` accessors, but they are
documented here alongside everything else so you never have to hunt across pages.

.. code-block:: python

   from sparrow import Protein

   p = Protein("MGSQSSRSSSQQQQQQQ")
   p.FCR                      # a property
   p.compute_kappa_x("ED", "KR")        # a method
   p.predictor.disorder()     # a prediction
   p.polymeric.predicted_nu   # a polymer-model property

.. contents:: Capabilities
   :local:
   :depth: 1

Creating a protein
------------------

.. code-block:: python

   from sparrow import Protein

   p = Protein("MGSQSSRSSSQQQQQQQ")
   p = Protein("MGSQXU--", validate=True)   # convert non-standard residues

   len(p)            # sequence length
   p.sequence        # the (upper-cased) sequence string

Sequence & composition
----------------------

.. autosummary::
   :nosignatures:

   sparrow.protein.Protein.sequence
   sparrow.protein.Protein.molecular_weight
   sparrow.protein.Protein.amino_acid_fractions
   sparrow.protein.Protein.compute_residue_fractions
   sparrow.protein.Protein.fraction_aromatic
   sparrow.protein.Protein.fraction_aliphatic
   sparrow.protein.Protein.fraction_polar
   sparrow.protein.Protein.fraction_proline

.. code-block:: python

   p.amino_acid_fractions["Q"]
   p.fraction_aromatic
   p.compute_residue_fractions(["S", "T"])

Charge & charge patterning
--------------------------

.. autosummary::
   :nosignatures:

   sparrow.protein.Protein.FCR
   sparrow.protein.Protein.NCPR
   sparrow.protein.Protein.fraction_positive
   sparrow.protein.Protein.fraction_negative
   sparrow.protein.Protein.kappa
   sparrow.protein.Protein.compute_kappa_x
   sparrow.protein.Protein.SCD
   sparrow.protein.Protein.compute_SCD_x

.. code-block:: python

   p.FCR, p.NCPR
   p.kappa                                   # charge segregation (0-1, or -1)
   p.compute_kappa_x("ED", "KR", window_size=6)
   p.SCD                                     # sequence charge decoration

Hydrophobicity
--------------

.. autosummary::
   :nosignatures:

   sparrow.protein.Protein.hydrophobicity
   sparrow.protein.Protein.SHD
   sparrow.protein.Protein.compute_SHD_custom

Complexity, clustering & patches
--------------------------------

.. autosummary::
   :nosignatures:

   sparrow.protein.Protein.complexity
   sparrow.protein.Protein.compute_iwd
   sparrow.protein.Protein.compute_iwd_charged_weighted
   sparrow.protein.Protein.compute_bivariate_iwd_charged_weighted
   sparrow.protein.Protein.compute_patch_fraction
   sparrow.protein.Protein.compute_rg_patch_fraction

.. code-block:: python

   p.compute_iwd(["D", "E"])                 # clustering of acidic residues
   p.compute_patch_fraction("Q")            # fraction in Q-rich patches

Linear (per-residue) profiles
-----------------------------

Windowed tracks with one value per residue -- ideal for plotting (see
:doc:`../examples`).

.. autosummary::
   :nosignatures:

   sparrow.protein.Protein.linear_sequence_profile
   sparrow.protein.Protein.linear_composition_profile
   sparrow.protein.Protein.linear_property_profile

.. code-block:: python

   p.linear_sequence_profile(mode="NCPR", window_size=7)
   p.linear_composition_profile(["Q", "N"], window_size=8)
   p.linear_property_profile("hydropathy-kyte-1982", window_size=9)  # AAindex

The 500+ AAindex scales usable with ``linear_property_profile`` are catalogued in
:doc:`properties`.

Domains, motifs & isoforms
--------------------------

.. autosummary::
   :nosignatures:

   sparrow.protein.Protein.low_complexity_domains
   sparrow.protein.Protein.plaac_prion_like_domains
   sparrow.protein.Protein.elms
   sparrow.protein.Protein.generate_phosphoisoforms

Predictions -- disorder & secondary structure
---------------------------------------------

Reached via ``protein.predictor``; networks load lazily and results are cached.

.. autosummary::
   :nosignatures:

   sparrow.predictors.Predictor.disorder
   sparrow.predictors.Predictor.disorder_domains
   sparrow.predictors.Predictor.binary_disorder
   sparrow.predictors.Predictor.pLDDT
   sparrow.predictors.Predictor.dssp_helicity
   sparrow.predictors.Predictor.dssp_extended
   sparrow.predictors.Predictor.dssp_coil

.. code-block:: python

   p.predictor.disorder()
   p.predictor.dssp_helicity()

Predictions -- polymer dimensions
---------------------------------

Single-value predictions via ``protein.predictor``; richer polymer-model
properties and distance distributions via ``protein.polymeric``.

.. autosummary::
   :nosignatures:

   sparrow.predictors.Predictor.radius_of_gyration
   sparrow.predictors.Predictor.end_to_end_distance
   sparrow.predictors.Predictor.scaling_exponent
   sparrow.predictors.Predictor.asphericity
   sparrow.predictors.Predictor.prefactor
   sparrow.polymer.Polymeric.predicted_nu
   sparrow.polymer.Polymeric.predicted_rg
   sparrow.polymer.Polymeric.predicted_re
   sparrow.polymer.Polymeric.predicted_asphericity
   sparrow.polymer.Polymeric.predicted_prefactor
   sparrow.polymer.Polymeric.get_afrc_end_to_end_distribution
   sparrow.polymer.Polymeric.get_afrc_radius_of_gyration_distribution

.. code-block:: python

   p.predictor.radius_of_gyration()
   p.polymeric.predicted_nu
   re_distance, probability = p.polymeric.get_afrc_end_to_end_distribution()

(The full polymer-model surface is documented in the ``Polymeric`` reference at
the bottom of this page.)

Predictions -- modification, localization & phase behavior
----------------------------------------------------------

.. autosummary::
   :nosignatures:

   sparrow.predictors.Predictor.serine_phosphorylation
   sparrow.predictors.Predictor.threonine_phosphorylation
   sparrow.predictors.Predictor.tyrosine_phosphorylation
   sparrow.predictors.Predictor.nuclear_import_signal
   sparrow.predictors.Predictor.nuclear_export_signal
   sparrow.predictors.Predictor.transactivation_domains
   sparrow.predictors.Predictor.transmembrane_regions
   sparrow.predictors.Predictor.mitochondrial_targeting_sequence
   sparrow.predictors.Predictor.pscore

.. code-block:: python

   p.predictor.serine_phosphorylation()
   p.predictor.transmembrane_regions()
   p.predictor.pscore()

Machine-learning feature vectors
--------------------------------

.. autosummary::
   :nosignatures:

   sparrow.protein.Protein.extract_feature_vector

.. code-block:: python

   vec = p.extract_feature_vector(num_scrambles=256, seed=1)

Visualization
-------------

.. autosummary::
   :nosignatures:

   sparrow.protein.Protein.show_sequence

.. code-block:: python

   p.show_sequence(bold_residues=["E", "D"])   # coloured HTML (e.g. in a notebook)

Good to know
------------

* ``kappa`` / ``compute_kappa_x`` return ``-1`` when undefined (sequence shorter
  than the window, or missing a required residue group).
* ``hydrophobicity`` returns the **mean** value; for a per-residue track use
  ``linear_sequence_profile(mode='hydrophobicity')``.
* Accessor objects (``predictor``, ``polymeric``, ``plugin``) are created on
  first access and reused; predictor networks load on first use.

----

Full reference
==============

``Protein``
-----------

.. autoclass:: sparrow.protein.Protein
   :members:
   :undoc-members:

The ``predictor`` accessor
--------------------------

Reached as ``protein.predictor``. See also the developer guide for
:doc:`adding a predictor <../predictors>`.

.. autoclass:: sparrow.predictors.Predictor
   :members:
   :undoc-members:

The ``polymeric`` accessor
--------------------------

Reached as ``protein.polymeric``.

.. autoclass:: sparrow.polymer.Polymeric
   :members:
   :undoc-members:
