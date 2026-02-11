Polymeric
=========

.. note::
   This page is a focused guide for polymeric-model analyses through
   ``Protein.polymeric``. For the broader API map, see :doc:`api`.

The ``polymeric`` accessor exposes sequence-level polymer model utilities
implemented by :class:`sparrow.polymer.Polymeric`.

Quick Start
-----------

.. code-block:: python

   from sparrow import Protein

   p = Protein("MEEEKKKKSSSTTTDDD")
   poly = p.polymeric

   nu = poly.predicted_nu
   rg = poly.predicted_rg
   re = poly.predicted_re
   asphericity = poly.predicted_asphericity
   prefactor = poly.predicted_prefactor

Distribution Workflows
----------------------

.. code-block:: python

   from sparrow import Protein

   p = Protein("MEEEKKKKSSSTTTDDD")
   poly = p.polymeric

   # AFRC-based distributions
   re_dist = poly.get_afrc_end_to_end_distribution()
   rg_dist = poly.get_afrc_radius_of_gyration_distribution()

   # SAW-style references
   saw_dist = poly.get_saw_end_to_end_distribution(prefactor=5.5)
   nudep_dist = poly.get_nudep_saw_end_to_end_distribution(nu=0.52, prefactor=5.5)

Caveats
-------

* Many polymeric models assume disordered/unfolded behavior.
* Some methods rely on optional ``afrc`` dependencies at runtime.
* Several calls are memoized internally; use ``recompute=True`` where provided
  if you need fresh recomputation.

Full Class Reference
--------------------

Primary accessor:

* :attr:`sparrow.protein.Protein.polymeric`

Key ``Polymeric`` APIs include:

* predicted properties: ``predicted_nu``, ``predicted_rg``, ``predicted_re``,
  ``predicted_asphericity``, ``predicted_prefactor``
* AFRC workflows: ``get_afrc_end_to_end_distribution``,
  ``get_afrc_radius_of_gyration_distribution``,
  ``get_mean_afrc_end_to_end_distance``,
  ``get_mean_afrc_radius_of_gyration``, ``get_afrc_internal_scaling``
* SAW workflows: ``get_saw_end_to_end_distribution``,
  ``get_mean_saw_end_to_end_distance``, ``get_mean_saw_radius_of_gyration``,
  ``get_nudep_saw_end_to_end_distribution``,
  ``get_mean_nudep_saw_end_to_end_distance``,
  ``get_mean_nudep_saw_radius_of_gyration``
* empirical estimates: ``empirical_nu``, ``empirical_radius_of_gyration``

.. autoclass:: sparrow.polymer.Polymeric
   :no-index:

.. automodule:: sparrow.polymer
   :no-index:
