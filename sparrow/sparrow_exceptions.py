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
    "SparrowWarning",
    "BackgroundQualityWarning",
    "ClusteringQualityWarning",
    "NullModelWarning",
]


class SparrowException(Exception):
    """Base class for every exception raised by sparrow."""


class ProteinException(SparrowException):
    """Raised for invalid input to :class:`sparrow.protein.Protein` methods."""


class PatterningException(SparrowException):
    """Raised by the patterning (kappa / IWD / SCD) routines."""


class CalculationException(SparrowException):
    """Raised when a sequence parameter calculation cannot be completed."""


class SparrowWarning(UserWarning):
    """Base class for advisory warnings sparrow raises.

    These flag a result that is computable but probably not what you want. Filter
    them as a group with::

        import warnings
        from sparrow.sparrow_exceptions import SparrowWarning

        warnings.filterwarnings("ignore", category=SparrowWarning)
    """


class BackgroundQualityWarning(SparrowWarning):
    """Raised by :func:`sparrow.nardini.background.build_background` for a weak background.

    A NARDINI+ compositional background is a per-feature mean and standard
    deviation over a reference set of IDRs. Built from too few sequences, or from
    sequences that never exhibit some feature, it still produces z-scores - they
    are simply not trustworthy, and for a feature that never varied they are
    silently zero. See :data:`sparrow.nardini.background.MIN_RECOMMENDED_SEQUENCES`
    and :data:`sparrow.nardini.background.MAX_DEGENERATE_FRACTION`.
    """


class ClusteringQualityWarning(SparrowWarning):
    """Raised by :func:`sparrow.nardini.clustering.build_gin_reference` for a weak clustering.

    k-means always returns k centroids, however little structure the data holds.
    Too few sequences per cluster and the centroids describe noise - they will
    still assign every query sequence to something, and the labels will still look
    plausible. See
    :data:`sparrow.nardini.clustering.MIN_SEQUENCES_PER_CLUSTER`.
    """


class NullModelWarning(SparrowWarning):
    """Raised by the null models of :mod:`sparrow.null` when they cannot do quite what was asked.

    A background model warns when it is fit to a background with residues it
    cannot count, or too few k-mers for their frequencies to mean much; when
    it cannot weight the background to a sequence's composition; and when it
    makes a null with some blocks or residues drawn from shorter k-mers than
    asked for, because the background had none that fit. The nulls are still
    exact in length and composition. See
    :data:`sparrow.null.kmers.MIN_MEAN_KMER_COUNT` and
    :data:`sparrow.null.tilt.TILT_TOLERANCE`.
    """
