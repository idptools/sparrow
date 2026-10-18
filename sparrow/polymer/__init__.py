"""Polymer models of a sequence's dimensions.

Each model is a module of plain functions that take a sequence:

* :mod:`sparrow.polymer.analytical_frc` - the Analytical Flory Random Coil (AFRC)
* :mod:`sparrow.polymer.saw` - a self-avoiding walk, whose size depends only on
  the length
* :mod:`sparrow.polymer.nudep_saw` - a self-avoiding walk with a free scaling
  exponent
* :mod:`sparrow.polymer.zheng2020` - the empirical estimates of Zheng et al.
  (2020), from sequence hydropathy and charge decoration

:class:`~sparrow.polymer.polymeric.Polymeric`, reached as ``Protein.polymeric``,
calls them for one sequence and caches the results, alongside the ALBATROSS
predictions of the sequence's dimensions.
"""

from sparrow.polymer.polymeric import Polymeric

__all__ = ["Polymeric"]
