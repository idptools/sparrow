Patterning Metrics
==================

These APIs provide direct access to sparrow's patterning metrics outside
``Protein`` methods, and are suitable for functional workflows.

They live under ``sparrow.patterning`` and are **pure Python** -- vectorized
NumPy, with no compiled extensions and no compiler needed. Earlier releases
implemented them in Cython; the module names and signatures are unchanged, so
existing code keeps working. See :ref:`patterning-vectorized` for what changed
and by how much.

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

.. _patterning-vectorized:

Where the implementations live
------------------------------

All of the above are implemented in :mod:`sparrow.patterning.vectorized`. The
``kappa``, ``iwd``, ``scd`` and ``patterning`` modules are thin re-exports of it,
so both import paths give you exactly the same functions:

.. code-block:: python

   from sparrow.patterning import vectorized as vec

   vec.kappa_x("MEEEKKKKSSSTTTDDD", ["E", "D"], ["K", "R"], 5, 1)
   vec.calculate_average_inverse_distance_from_sequence("QQQAAAQQQ", ["Q"])
   vec.compute_scd_x("MEEEKKKKSSSTTTDDD")
   vec.compute_shd("MEEEKKKKSSSTTTDDD")
   vec.patterning_percentile("QQQQAAAAQQQQ", ["Q"], 5, 1000, seed=1)

It also adds **batch** entry points with no Cython equivalent, which score a
whole stack of sequences in one pass. These are what make scramble-based null
distributions cheap:

.. code-block:: python

   vec.kappa_x_batch(list_of_scrambles, ["E", "D"], ["K", "R"], 5, 1)
   vec.iwd_batch(stack_of_binary_masks)

Why it is often faster
^^^^^^^^^^^^^^^^^^^^^^

Partly vectorization, but mostly better algorithms. Three of these statistics
have structure the loop-based versions do not exploit:

* **IWD** sums ``1 / |i - j|`` over occupied pairs. Grouping by separation makes
  the inner sum the *autocorrelation* of the occupancy mask, computable with one
  FFT instead of an explicit pairwise loop.
* **SCD** likewise reduces to an autocorrelation of the charge sequence weighted
  by ``sqrt(d)`` -- ``O(n log n)`` rather than ``O(n^2)``.
* **SHD** has a *sum* rather than a product in its numerator, so grouping by
  separation leaves only prefix sums: ``O(n)`` rather than ``O(n^2)``.

Measured against the compiled extensions:

.. code-block:: text

   single sequence           cython      numpy
     kappa_x   (L=300)       142.7 us    48.6 us       3x
     iwd       (L=300)        27.8 us    29.1 us       1x
     compute_scd_x (L=5000)    7.75 ms    0.148 ms    53x
     compute_shd   (L=5000)  392.81 ms    0.221 ms  1779x

   2000 scrambles, batch     cython      numpy
     kappa     (L=1000)        1.01 s     0.04 s      28x
     iwd       (L=1000)        0.47 s     0.03 s      18x

Numerical agreement
^^^^^^^^^^^^^^^^^^^

The IWD, SCD and SHD routines are double precision on both sides and agree to
~1e-13 or better. The kappa routines carry their intermediates in C ``float`` in
the Cython code and ``float64`` here, so they agree to ~1e-6 -- and where they
differ, **the NumPy value is the more accurate one**.
``sparrow/tests/test_patterning_vectorized.py`` pins all of this, including a
test that replays the Cython arithmetic in single precision to demonstrate the
residual gap is the compiled code's precision rather than an error in the
reimplementation.

.. warning::
   ``patterning_percentile`` is the one function whose *seed* behaviour changes.
   The compiled version draws its shuffles from C's ``rand()``; the C standard
   leaves that generator implementation-defined, so its seed-to-result mapping
   was never portable across platforms or libc versions in the first place. The
   NumPy version uses NumPy's generator, so the same ``seed`` produces a
   different - statistically equivalent - set of shuffles.

   The *statistic* is unchanged (it agrees to ~1e-6, i.e. the compiled code's
   precision), and percentiles from the two agree to within Monte Carlo sampling
   error. Do not compare a stored percentile from one implementation against the
   other; recompute instead.

What ``Protein`` uses
---------------------

As of this release :class:`sparrow.protein.Protein` and
:mod:`sparrow.polymer.scaling_parameters` call the NumPy implementations, not the
compiled extensions. The effect on user-facing values:

.. code-block:: text

   Protein.kappa / compute_kappa_x     shifted by <= 6e-06   (single -> double precision)
   Protein.SCD / SHD                   shifted by <= 5e-14
   compute_iwd / the charge-weighted   shifted by <= 7e-16
   IWD variants

   Protein.SHD    (L=3000)   138.85 ms -> 0.14 ms    967x
   Protein.SCD    (L=3000)     2.82 ms -> 0.08 ms     34x
   Protein.kappa  (L=3000)     3.33 ms -> 0.23 ms     15x

The kappa shift is the compiled code's single-precision arithmetic being
replaced by double, so the new values are the more accurate ones. It is far below
any tolerance that matters scientifically, but if you have stored kappa values
from an earlier release expect agreement to about six significant figures rather
than exactly.

Installation notes
------------------

* There is **no build step**. sparrow is pure Python, so there are no extensions
  to compile and no compiler to install.
* If you are updating a checkout from before the extensions were removed, delete
  the stale artifacts first::

      rm -f sparrow/patterning/*.so sparrow/patterning/*.c

  Python resolves extension modules before source modules, so a leftover
  ``kappa.cpython-*.so`` would shadow ``kappa.py`` and silently keep running the
  old compiled code.

Reference: ``sparrow.patterning.vectorized``
--------------------------------------------

.. automodule:: sparrow.patterning.vectorized
   :members:
   :no-index:

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
