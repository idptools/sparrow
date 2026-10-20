Polymer Models (``protein.polymeric``)
======================================

``protein.polymeric`` describes the sequence as a polymer: the ALBATROSS
predictions of its dimensions, and the dimensions and distance distributions of
three polymer models and an empirical relationship. Each value is computed the
first time you ask for it and cached. Most of them only mean something for a
sequence that behaves as an intrinsically disordered or unfolded chain.

.. code-block:: python

   from sparrow import Protein

   p = Protein("MEEEKKKKSSSTTTDDDQQQQNNNNGGGGSSSSAAAAPPPP")

   p.polymeric.predicted_nu                        # ALBATROSS scaling exponent
   p.polymeric.predicted_rg                        # ALBATROSS Rg, in Angstroms

   re_distance, probability = p.polymeric.afrc_end_to_end_distribution()
   p.polymeric.mean_afrc_radius_of_gyration()      # Analytical Flory Random Coil

   p.polymeric.mean_nudep_saw_radius_of_gyration(nu=0.55, prefactor=5.5)
   p.polymeric.empirical_nu()                      # Zheng et al. 2020

The models
----------

* **ALBATROSS predictions** - ``predicted_rg``, ``predicted_re``,
  ``predicted_nu``, ``predicted_asphericity`` and ``predicted_prefactor`` are
  the predictions of :doc:`predictor`, read from ``protein.predictor``, and
  ``predicted_nu_dep_end_to_end_distribution`` is the nu-dependent
  self-avoiding walk with the predicted scaling exponent and prefactor.
* **The Analytical Flory Random Coil (AFRC)** (Alston *et al.*, 2023) - the
  dimensions a sequence would have as an ideal chain with no attractive or
  repulsive interactions, a reference to compare a real or predicted ensemble
  against. It comes from the `afrc <https://github.com/idptools/afrc>`_
  package.
* **The self-avoiding walk (SAW)** - a chain in a good solvent, in the des
  Cloizeaux form of O'Brien *et al.* (2009), with the scaling exponent fixed
  at afrc's good-solvent value, so its size depends only on its length and a
  prefactor.
* **The nu-dependent SAW** - the same, with a scaling exponent you choose, for
  chains more compact or more expanded than a good-solvent chain. Both walks
  are computed by the afrc package too.
* **Empirical relationships** - the scaling exponent and radius of gyration
  estimated from the sequence's hydropathy and charge decoration by Zheng *et
  al.* (2020).

A distribution is returned as two arrays, the distances in Angstroms and the
probability of each, which sum to 1. Every mean distance is in Angstroms.

Each model is also a module of plain functions of a sequence, which
``protein.polymeric`` calls and caches; see :doc:`../power_user/polymer_models`.

Old names
---------

The ``Polymeric`` methods were called ``get_afrc_end_to_end_distribution`` and
so on before they lost their ``get_`` prefix. The old names still work, with a
``DeprecationWarning``, and will be removed in a future release.

Reference
---------

.. autoclass:: sparrow.polymer.Polymeric
   :no-members:
   :no-inherited-members:
   :no-undoc-members:

.. autoproperty:: sparrow.protein.Protein.polymeric

ALBATROSS predictions
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   sparrow.polymer.Polymeric.predicted_rg
   sparrow.polymer.Polymeric.predicted_re
   sparrow.polymer.Polymeric.predicted_nu
   sparrow.polymer.Polymeric.predicted_asphericity
   sparrow.polymer.Polymeric.predicted_prefactor
   sparrow.polymer.Polymeric.predicted_nu_dep_end_to_end_distribution

.. autoproperty:: sparrow.polymer.Polymeric.predicted_rg

.. autoproperty:: sparrow.polymer.Polymeric.predicted_re

.. autoproperty:: sparrow.polymer.Polymeric.predicted_nu

.. autoproperty:: sparrow.polymer.Polymeric.predicted_asphericity

.. autoproperty:: sparrow.polymer.Polymeric.predicted_prefactor

.. automethod:: sparrow.polymer.Polymeric.predicted_nu_dep_end_to_end_distribution

The Analytical Flory Random Coil
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   sparrow.polymer.Polymeric.afrc_end_to_end_distribution
   sparrow.polymer.Polymeric.afrc_radius_of_gyration_distribution
   sparrow.polymer.Polymeric.mean_afrc_end_to_end_distance
   sparrow.polymer.Polymeric.mean_afrc_radius_of_gyration
   sparrow.polymer.Polymeric.afrc_internal_scaling

.. automethod:: sparrow.polymer.Polymeric.afrc_end_to_end_distribution

.. automethod:: sparrow.polymer.Polymeric.afrc_radius_of_gyration_distribution

.. automethod:: sparrow.polymer.Polymeric.mean_afrc_end_to_end_distance

.. automethod:: sparrow.polymer.Polymeric.mean_afrc_radius_of_gyration

.. automethod:: sparrow.polymer.Polymeric.afrc_internal_scaling

Self-avoiding walks
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   sparrow.polymer.Polymeric.saw_end_to_end_distribution
   sparrow.polymer.Polymeric.mean_saw_end_to_end_distance
   sparrow.polymer.Polymeric.mean_saw_radius_of_gyration
   sparrow.polymer.Polymeric.nudep_saw_end_to_end_distribution
   sparrow.polymer.Polymeric.mean_nudep_saw_end_to_end_distance
   sparrow.polymer.Polymeric.mean_nudep_saw_radius_of_gyration

.. automethod:: sparrow.polymer.Polymeric.saw_end_to_end_distribution

.. automethod:: sparrow.polymer.Polymeric.mean_saw_end_to_end_distance

.. automethod:: sparrow.polymer.Polymeric.mean_saw_radius_of_gyration

.. automethod:: sparrow.polymer.Polymeric.nudep_saw_end_to_end_distribution

.. automethod:: sparrow.polymer.Polymeric.mean_nudep_saw_end_to_end_distance

.. automethod:: sparrow.polymer.Polymeric.mean_nudep_saw_radius_of_gyration

Empirical relationships
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   sparrow.polymer.Polymeric.empirical_nu
   sparrow.polymer.Polymeric.empirical_radius_of_gyration

.. automethod:: sparrow.polymer.Polymeric.empirical_nu

.. automethod:: sparrow.polymer.Polymeric.empirical_radius_of_gyration
