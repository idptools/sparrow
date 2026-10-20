Polymer Models as Functions
===========================

Each polymer model behind ``protein.polymeric`` (:doc:`../protein/polymeric`) is
a module of plain functions that take a sequence, which ``Polymeric`` calls and
caches. Call them directly to use a model without a ``Protein``, or with
parameters a ``Polymeric`` method fixes.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Module
     - Model
   * - :mod:`sparrow.polymer.analytical_frc`
     - The Analytical Flory Random Coil (AFRC; Alston *et al.*, 2023): the
       dimensions of the sequence as an ideal chain, from the afrc package.
   * - :mod:`sparrow.polymer.saw`
     - A self-avoiding walk (the des Cloizeaux form of O'Brien *et al.*, 2009),
       with the good-solvent scaling exponent, so only the length and a
       prefactor set its size.
   * - :mod:`sparrow.polymer.nudep_saw`
     - The same walk with a free scaling exponent, from a collapsed globule
       (nu near 1/3) through the ideal chain (0.5) to a good-solvent coil
       (near 0.588).
   * - :mod:`sparrow.polymer.zheng2020`
     - The empirical scaling exponent and radius of gyration of Zheng *et al.*
       (2020), from the sequence's hydropathy and charge decoration (SHD and
       SCD).

.. code-block:: python

   from sparrow import Protein
   from sparrow.polymer import analytical_frc, nudep_saw, saw, zheng2020

   sequence = "MEEEKKKKSSSTTTDDDQQQQNNNNGGGGSSSSAAAAPPPP"
   p = Protein(sequence)

   analytical_frc.mean_radius_of_gyration(sequence)       # = p.polymeric.mean_afrc_radius_of_gyration()
   nudep_saw.mean_radius_of_gyration(sequence, nu=0.55)   # = p.polymeric.mean_nudep_saw_radius_of_gyration(nu=0.55)
   saw.end_to_end_distribution(sequence)                  # = p.polymeric.saw_end_to_end_distribution()
   zheng2020.scaling_exponent(sequence)                   # = p.polymeric.empirical_nu()

Every distance is in Angstroms, and a distribution is two arrays, the distances
and the probability of each.

Old names
---------

The Zheng *et al.* estimates used to be ``compute_nu_zheng2020`` and
``compute_rg_zheng2020``, in ``sparrow.polymer.scaling_parameters``. They are
now :func:`sparrow.polymer.zheng2020.scaling_exponent` and
:func:`sparrow.polymer.zheng2020.radius_of_gyration`; the old names still work,
with a ``DeprecationWarning``, and will be removed in a future release.

Reference
---------

The four modules are documented in full in :doc:`../api/polymer`.
