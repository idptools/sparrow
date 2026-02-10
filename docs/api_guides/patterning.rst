Patterning Metrics (Cython-Backed)
==================================

These APIs provide direct access to high-performance patterning metrics outside
``Protein`` methods. They are implemented in Cython modules under
``sparrow.patterning`` and are suitable for functional workflows.

Public Functions Covered
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   sparrow.patterning.kappa.calculate_sigma
   sparrow.patterning.kappa.calculate_delta
   sparrow.patterning.kappa.kappa_x
   sparrow.patterning.iwd.calculate_average_inverse_distance_from_sequence
   sparrow.patterning.iwd.calculate_average_inverse_distance_charge
   sparrow.patterning.iwd.calculate_average_bivariate_inverse_distance_charge
   sparrow.patterning.patterning.patterning_percentile

Examples
--------

Binary/ternary patterning with ``kappa_x``:

.. code-block:: python

   from sparrow.patterning.kappa import kappa_x

   seq = "MEEEKKKKSSSTTTDDD"

   charge_kappa = kappa_x(seq, ["E", "D"], ["K", "R"], window_size=6, flatten=1)
   binary_kappa = kappa_x(seq, ["Q", "N"], [], window_size=6, flatten=1)

Inverse-weighted distance examples:

.. code-block:: python

   import numpy as np
   from sparrow.patterning.iwd import (
       calculate_average_inverse_distance_from_sequence,
       calculate_average_inverse_distance_charge,
       calculate_average_bivariate_inverse_distance_charge,
   )

   seq = "MEEEKKKKSSSTTTDDD"
   linear_ncpr = np.linspace(-0.5, 0.5, len(seq))

   acidic_iwd = calculate_average_inverse_distance_from_sequence(seq, ["D", "E"])
   neg_weighted = calculate_average_inverse_distance_charge(linear_ncpr, seq, "-")
   bi_weighted = calculate_average_bivariate_inverse_distance_charge(linear_ncpr, seq)

Patterning percentile from binary grouping:

.. code-block:: python

   from sparrow.patterning.patterning import patterning_percentile

   seq = "QQQQAAAQQQQAAAQQQQ"
   percentile = patterning_percentile(seq, ["Q"], window_size=6, count=200, seed=1)

Interpretation and Edge Cases
-----------------------------

* ``kappa_x`` returns ``-1`` for non-computable inputs (for example, sequence too
  short for window size, or missing required residues).
* ``window_size`` must be at least 2 for ``kappa_x`` and ``patterning_percentile``.
* For ``kappa_x``, use ``flatten=1`` if you want values capped at 1.

Performance and Compilation Notes
---------------------------------

* These functions are implemented in Cython for speed on large sequence sets.
* Doc builds on Read the Docs may mock unavailable compiled modules when needed.
* For local development, ensure the package extensions are built before benchmarking.

Reference: ``sparrow.patterning.kappa``
---------------------------------------

.. automodule:: sparrow.patterning.kappa
   :no-index:

Reference: ``sparrow.patterning.iwd``
-------------------------------------

.. automodule:: sparrow.patterning.iwd
   :no-index:

Reference: ``sparrow.patterning.patterning``
--------------------------------------------

.. automodule:: sparrow.patterning.patterning
   :no-index:
