"""General-purpose binary patterning significance testing.

:func:`patterning_percentile` binarizes a sequence into a residue group versus
everything else, measures how much the local windowed composition deviates from
the whole-sequence composition, and reports where that sits in the distribution
obtained from random shuffles of the same sequence. 0 means far more evenly
patterned than chance, 1 means far blockier.

.. note::
   This module was a compiled Cython extension until the implementation moved to
   pure NumPy in :mod:`sparrow.patterning.vectorized`. The names and signatures
   here are unchanged, so existing code keeps working; sparrow no longer needs a
   C compiler to install.

.. warning::
   **The** ``seed`` **argument now selects a different stream.** The compiled
   version drew its shuffles from C's ``rand()``, whose algorithm the C standard
   leaves implementation-defined - so its seed-to-result mapping was never
   portable across platforms or libc versions anyway. This implementation uses
   NumPy's generator, so the same ``seed`` produces a different, statistically
   equivalent set of shuffles.

   The underlying statistic is unchanged (it agrees with the compiled version to
   ~1e-6, i.e. that code's single-precision resolution), and percentiles from the
   two agree to within Monte Carlo sampling error. Do not compare a percentile
   stored from an older release against one computed now - recompute it.
"""

from sparrow.patterning.vectorized import binary_asymmetry, patterning_percentile

__all__ = [
    "patterning_percentile",
    "binary_asymmetry",
]
