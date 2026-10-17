Patterning Metrics
==================

These APIs give direct access to sparrow's patterning metrics outside
``Protein`` methods. They are **pure Python** -- vectorized NumPy, with no
compiled extensions and no compiler needed.

The windowed-asymmetry statistics -- kappa, delta, sigma and the patterning
percentile -- live in :mod:`sparrow.properties.patterning`, which is also where
NARDINI's omega and pair features are computed: they are all Das & Pappu's
delta, from one function. Every function there takes a batch of sequences and
returns one value per sequence. The inverse-weighted distance, SCD and SHD are
still in ``sparrow.patterning`` for now.

Public Functions Covered
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   sparrow.properties.patterning.sigma
   sparrow.properties.patterning.delta
   sparrow.properties.patterning.kappa_x
   sparrow.properties.patterning.kappa
   sparrow.properties.patterning.patterning_percentile
   sparrow.patterning.iwd.calculate_average_inverse_distance_from_sequence
   sparrow.patterning.iwd.calculate_average_inverse_distance_charge
   sparrow.patterning.iwd.calculate_average_bivariate_inverse_distance_charge

Examples
--------

Binary/ternary patterning with ``kappa_x``, for as many sequences as you like:

.. code-block:: python

   from sparrow.properties.patterning import kappa, kappa_x

   sequences = ["MEEEKKKKSSSTTTDDD", "EKEKEKEKEKEKEKEKEK"]

   charge_kappa = kappa(sequences)                                  # as Protein.kappa
   custom = kappa_x(sequences, ["E", "D"], ["K", "R"], window_size=6, flatten=True)
   binary = kappa_x(sequences, ["Q", "N"], [], window_size=6, flatten=True)

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

   from sparrow.properties.patterning import patterning_percentile

   percentiles = patterning_percentile(["QQQQAAAQQQQAAAQQQQ"], ["Q"], window_size=6, count=200, seed=1)

Interpretation and Edge Cases
-----------------------------

* ``kappa_x`` returns ``-1`` for non-computable inputs (for example, sequence too
  short for window size, or missing required residues).
* ``delta`` returns NaN for a sequence shorter than the window, which has no
  windows to average.
* ``window_size`` must be at least 2 for ``kappa_x`` and ``patterning_percentile``,
  and ``patterning_percentile`` rejects any sequence shorter than the window.
* For ``kappa_x``, use ``flatten=True`` if you want values capped at 1.
* A sequence gets the same value -- to the last bit -- whether it is scored
  alone or in a batch. Sequences of the same length are scored together as one
  array, and each patterning percentile reseeds its shuffles from ``seed``.

.. _patterning-vectorized:

Where the implementations live
------------------------------

The kappa family is implemented in :mod:`sparrow.properties.patterning`; IWD,
SCD and SHD in :mod:`sparrow.patterning.vectorized`, with ``iwd`` and ``scd``
as thin re-exports of it. Both score whole stacks at once, which is what makes
scramble-based null distributions cheap:

.. code-block:: python

   from sparrow.patterning import vectorized as vec
   from sparrow.properties import encoding, patterning

   patterning.kappa_from_ternary(encoding.ternarize_winged_batch(codes, ["E", "D"], ["K", "R"], 5), 5)
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

Reference: ``sparrow.properties.patterning``
--------------------------------------------

.. automodule:: sparrow.properties.patterning
   :members:
   :no-index:

Reference: ``sparrow.properties.windows``
-----------------------------------------

.. automodule:: sparrow.properties.windows
   :members:
   :no-index:

Reference: ``sparrow.patterning.vectorized``
--------------------------------------------

.. automodule:: sparrow.patterning.vectorized
   :members:
   :no-index:

Reference: ``sparrow.patterning.iwd``
-------------------------------------

.. automodule:: sparrow.patterning.iwd
   :no-index:
