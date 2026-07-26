"""Sequence charge decoration (SCD) and sequence hydropathy decoration (SHD).

Two pairwise patterning parameters from the literature:

* **SCD** (Sawle & Ghosh 2015) sums the product of two residues' group charges
  weighted by ``sqrt`` of their separation, over all pairs.
* **SHD** (Zheng et al. 2020) sums the two residues' hydropathies divided by
  their separation, over all pairs.

.. note::
   This module was a compiled Cython extension until the implementation moved to
   pure NumPy in :mod:`sparrow.patterning.vectorized`. The names and signatures
   here are unchanged, so existing code keeps working; sparrow no longer needs a
   C compiler to install.

   Both implementations work in double precision, so values agree to ~1e-13. The
   NumPy versions are also considerably faster: exploiting the structure of the
   two sums reduces SCD from ``O(n^2)`` to ``O(n log n)`` and SHD from ``O(n^2)``
   to ``O(n)``, which at length 5000 is a 53x and 1779x speed-up respectively.
"""

from sparrow.patterning.vectorized import (
    DEFAULT_HYDRO_DICT,
    compute_scd_x,
    compute_shd,
)

__all__ = [
    "compute_scd_x",
    "compute_shd",
    "DEFAULT_HYDRO_DICT",
]
