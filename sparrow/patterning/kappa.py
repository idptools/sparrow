"""Kappa-based sequence patterning metrics.

Kappa quantifies how segregated two groups of residues are along a sequence:
values near 0 mean well-mixed, values near 1 mean blocky. It is the ratio of
``delta`` - how much the local windowed compositional asymmetry deviates from
the whole-sequence asymmetry - to ``delta_max``, the same quantity for the
maximally segregated sequence of identical composition.

.. note::
   This module was a compiled Cython extension until the implementation moved to
   pure NumPy in :mod:`sparrow.patterning.vectorized`. The names and signatures
   here are unchanged, so existing code keeps working; sparrow no longer needs a
   C compiler to install.

   The NumPy implementation carries its intermediates in ``float64`` where the
   Cython one used C ``float``, so values differ from pre-move releases by about
   1e-6 - and the current values are the more accurate ones.

See :mod:`sparrow.patterning.vectorized` for the batch entry points
(:func:`~sparrow.patterning.vectorized.kappa_x_batch` and friends), which score a
whole stack of sequences per call.
"""

from sparrow.patterning.vectorized import (
    calculate_delta,
    calculate_sigma,
    kappa_from_ternary_batch,
    kappa_x,
    kappa_x_batch,
    ternarize_winged,
    ternarize_winged_batch,
)

__all__ = [
    "calculate_sigma",
    "calculate_delta",
    "kappa_x",
    "kappa_x_batch",
    "kappa_from_ternary_batch",
    "ternarize_winged",
    "ternarize_winged_batch",
]
