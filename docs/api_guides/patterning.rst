Patterning Metrics
==================

These APIs give direct access to sparrow's patterning metrics -- how a
sequence's residues are *arranged*, as opposed to what they are -- outside
``Protein`` methods. They all live in :mod:`sparrow.properties.patterning`, are
**pure Python** (vectorized NumPy, no compiled extensions, no compiler needed),
and are the same functions behind the corresponding ``Protein`` properties and
NARDINI's patterning features.

Every function takes a batch of sequences and returns one value per sequence;
a single string is a batch of one.

Public Functions Covered
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   sparrow.properties.patterning.kappa
   sparrow.properties.patterning.kappa_x
   sparrow.properties.patterning.sigma
   sparrow.properties.patterning.delta
   sparrow.properties.patterning.patterning_percentile
   sparrow.properties.patterning.iwd
   sparrow.properties.patterning.iwd_charged_weighted
   sparrow.properties.patterning.bivariate_iwd_charged_weighted
   sparrow.properties.patterning.scd
   sparrow.properties.patterning.shd

Examples
--------

Charge patterning and generalized kappa:

.. code-block:: python

   from sparrow.properties.patterning import kappa, kappa_x

   sequences = ["MEEEKKKKSSSTTTDDD", "EKEKEKEKEKEKEKEKEK"]

   charge_kappa = kappa(sequences)                                  # as Protein.kappa
   custom = kappa_x(sequences, ["E", "D"], ["K", "R"], window_size=6, flatten=True)
   binary = kappa_x(sequences, ["Q", "N"], [], window_size=6, flatten=True)

Inverse-weighted distance -- plain, and weighted by the local net charge:

.. code-block:: python

   from sparrow.properties.patterning import (
       bivariate_iwd_charged_weighted,
       iwd,
       iwd_charged_weighted,
   )

   sequences = ["MEEEKKKKSSSTTTDDD", "MEKEKEKEKSSSTTTDDD"]

   acidic = iwd(sequences, ["D", "E"])
   negative = iwd_charged_weighted(sequences, "-")
   mixing = bivariate_iwd_charged_weighted(sequences)

Charge and hydropathy decoration:

.. code-block:: python

   from sparrow.properties.patterning import scd, shd

   charge_decoration = scd(sequences)                 # E/D against R/K
   hydropathy_decoration = shd(sequences)             # HPS hydropathy scale

Patterning percentile against random shuffles:

.. code-block:: python

   from sparrow.properties.patterning import patterning_percentile

   percentiles = patterning_percentile(["QQQQAAAQQQQAAAQQQQ"], ["Q"], window_size=6, count=200, seed=1)

One statistic underneath
------------------------

kappa, delta, the patterning percentile and NARDINI's omega and pair features
are all built on the windowed compositional asymmetry of Das & Pappu (2013),
computed by one function, :func:`~sparrow.properties.patterning.window_delta`:

* :func:`~sparrow.properties.patterning.delta` is the asymmetry between two
  residue groups over the windows of the sequence -- exactly what NARDINI
  z-scores for a pair of residue types;
* the patterning percentile's statistic is one group against everything else --
  exactly NARDINI's omega;
* :func:`~sparrow.properties.patterning.kappa_x` is delta over the sequence padded
  with neutral wings, divided by the delta of the most segregated arrangement of
  the same composition.

``sparrow/tests/test_properties_patterning.py`` asserts the NARDINI equalities to
the last bit.

Batches
-------

Sequences of the same length are scored together as one array, so a whole
proteome costs a few array passes per distinct length rather than one call per
sequence. They are grouped by exact length rather than padded to a common one: a
padded row is summed in a different order, and would differ in the last bit. As
a result a sequence gets the same value -- to the last bit -- alone or in any
batch, and each patterning percentile reseeds its shuffles from ``seed``, so it
does not depend on the batch either.

For scrambles of one sequence, which all share a length and a composition, two
lower-level functions take arrays directly:
:func:`~sparrow.properties.patterning.kappa_from_ternary` (computing
``delta_max`` once for the whole stack) and
:func:`~sparrow.properties.patterning.iwd_from_masks`. NARDINI's ``'kappa'`` and
``'iwd'`` backends use them.

Interpretation and Edge Cases
-----------------------------

* ``kappa_x`` returns ``-1`` for non-computable inputs (for example, sequence too
  short for window size, or missing required residues); ``kappa`` likewise for
  a sequence shorter than 6 residues or lacking either charge.
* ``delta`` returns NaN for a sequence shorter than the window, which has no
  windows to average.
* ``window_size`` must be at least 2 for ``kappa_x`` and ``patterning_percentile``,
  and ``patterning_percentile`` rejects any sequence shorter than the window.
* For ``kappa_x``, use ``flatten=True`` if you want values capped at 1.
* The charge-weighted IWDs need sequences at least as long as their NCPR window
  (8 by default).
* ``shd`` reads its scale character by character, so it works for any alphabet
  the scale covers; the SHD paper's own checks use binary ``'0'``/``'1'``
  sequences.

Why it is often faster
----------------------

Partly vectorization, but mostly better algorithms. Three of these statistics
have structure the loop-based versions do not exploit:

* **IWD** sums ``1 / |i - j|`` over occupied pairs. Grouping by separation makes
  the inner sum the *autocorrelation* of the occupancy mask, computable with one
  FFT instead of an explicit pairwise loop.
* **SCD** likewise reduces to an autocorrelation of the charge sequence weighted
  by ``sqrt(d)`` -- ``O(n log n)`` rather than ``O(n^2)``.
* **SHD** has a *sum* rather than a product in its numerator, so grouping by
  separation leaves only prefix sums: ``O(n)`` rather than ``O(n^2)``.

Measured against the Cython extensions sparrow used to ship, when they were
replaced:

.. code-block:: text

   single sequence           cython      numpy
     kappa_x   (L=300)       142.7 us    48.6 us       3x
     iwd       (L=300)        27.8 us    29.1 us       1x
     scd       (L=5000)        7.75 ms    0.148 ms    53x
     shd       (L=5000)      392.81 ms    0.221 ms  1779x

   2000 scrambles, batch     cython      numpy
     kappa     (L=1000)        1.01 s     0.04 s      28x
     iwd       (L=1000)        0.47 s     0.03 s      18x

Numerical history
-----------------

These metrics were compiled Cython extensions, under ``sparrow.patterning``,
until they were reimplemented in NumPy; that module has since been removed in
favour of :mod:`sparrow.properties.patterning`.

The IWD, SCD and SHD routines were double precision on both sides and agree with
the compiled ones to ~1e-13 or better. The kappa routines carried their
intermediates in C ``float`` in the Cython code and ``float64`` here, so they
agree to ~1e-6 -- and where they differ, **the NumPy value is the more accurate
one**. ``sparrow/tests/test_properties_patterning.py`` pins all of this against
values captured from the extensions before they were removed, including a test
that replays the Cython arithmetic in single precision to show the residual gap
is the compiled code's precision rather than an error here.

.. warning::
   ``patterning_percentile`` is the one function whose *seed* behaviour changed.
   The compiled version drew its shuffles from C's ``rand()``, whose algorithm
   the C standard leaves implementation-defined, so its seed-to-result mapping
   was never portable across platforms in the first place. The NumPy version
   uses NumPy's generator, so the same ``seed`` produces a different -
   statistically equivalent - set of shuffles. Do not compare a stored
   percentile from the compiled version against one computed now; recompute.

Installation notes
------------------

* There is **no build step**. sparrow is pure Python, so there are no extensions
  to compile and no compiler to install.
* If you are updating a checkout from before ``sparrow.patterning`` was removed,
  delete that directory - ``rm -r sparrow/patterning`` from the repository root.
  Python would otherwise still find any compiled ``kappa.cpython-*.so`` left in
  it, and ``import sparrow.patterning.kappa`` would silently run the old compiled
  code instead of failing.

Reference: ``sparrow.properties.patterning``
--------------------------------------------

The functions listed at the top of this page each have a page of their own;
everything else in the module is documented here.

.. automodule:: sparrow.properties.patterning
   :members:
   :exclude-members: kappa, kappa_x, sigma, delta, patterning_percentile, iwd,
                     iwd_charged_weighted, bivariate_iwd_charged_weighted, scd, shd

Reference: ``sparrow.properties.windows``
-----------------------------------------

.. automodule:: sparrow.properties.windows
   :members:
