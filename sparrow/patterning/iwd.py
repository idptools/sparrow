"""Inverse-weighted distance (IWD) clustering metrics.

IWD quantifies how clustered a set of residues is along a sequence: for every
occupied position, sum ``1 / distance`` to every other occupied position, then
average. Charge-weighted variants weight each contribution by the local net
charge per residue, and the bivariate variant measures how intermixed
oppositely-charged residues are.

.. note::
   This module was a compiled Cython extension until the implementation moved to
   pure NumPy in :mod:`sparrow.patterning.vectorized`. The names and signatures
   here are unchanged, so existing code keeps working; sparrow no longer needs a
   C compiler to install.

   Both implementations work in double precision, so values agree to ~1e-15.

See :mod:`sparrow.patterning.vectorized` for
:func:`~sparrow.patterning.vectorized.iwd_batch`, which scores a whole stack of
occupancy masks in one FFT-based pass.
"""

from sparrow.patterning.vectorized import (
    calculate_average_bivariate_inverse_distance_charge,
    calculate_average_inverse_distance_charge,
    calculate_average_inverse_distance_from_sequence,
    iwd_batch,
    iwd_from_binary_mask,
)

__all__ = [
    "calculate_average_inverse_distance_from_sequence",
    "calculate_average_inverse_distance_charge",
    "calculate_average_bivariate_inverse_distance_charge",
    "iwd_from_binary_mask",
    "iwd_batch",
]
