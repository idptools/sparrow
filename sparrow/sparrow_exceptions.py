"""Exception types raised by sparrow.

All sparrow-specific exceptions derive from :class:`SparrowException`, so
``except SparrowException`` catches anything sparrow raises deliberately:

.. code-block:: python

    from sparrow import Protein
    from sparrow.sparrow_exceptions import SparrowException

    try:
        Protein("MEEEKKK").compute_iwd_charged_weighted(charge="?")
    except SparrowException as e:
        print(e)

The more specific subclasses let you narrow that down when you care which layer
raised.
"""

__all__ = [
    "SparrowException",
    "ProteinException",
    "PatterningException",
    "CalculationException",
]


class SparrowException(Exception):
    """Base class for every exception raised by sparrow."""


class ProteinException(SparrowException):
    """Raised for invalid input to :class:`sparrow.protein.Protein` methods."""


class PatterningException(SparrowException):
    """Raised by the patterning (kappa / IWD / SCD) routines."""


class CalculationException(SparrowException):
    """Raised when a sequence parameter calculation cannot be completed."""
