"""Shared base class for the serine/threonine/tyrosine phosphorylation predictors.

The three phosphorylation predictors are identical apart from their weights file
and the target residue (``S``/``T``/``Y``). They all turn the network output into
per-residue positive-class probabilities and then optionally mask those to
phosphosites via :func:`phospho_predictor_utils.return_hits`.
"""

import numpy as np

from sparrow.predictors import outputs
from sparrow.predictors.base import BaseNetworkPredictor

from . import phospho_predictor_utils

__all__ = ["PhosphoPredictor"]


class PhosphoPredictor(BaseNetworkPredictor):
    """Base class for residue-specific phosphorylation predictors.

    Subclasses set :attr:`NETWORK_PATH`, :attr:`DEFAULT_VERSION`, and
    :attr:`TARGET_RESIDUE`, and expose a residue-named ``predict_*`` method that
    delegates to :meth:`_predict_phospho`.
    """

    ARCHITECTURE = "MtM"
    TARGET_RESIDUE: str = None

    def _predict_phospho(
        self, seq, raw_values=False, return_sites_only=False, windowsize=4, threshold=0.6
    ):
        """Predict phosphosites for :attr:`TARGET_RESIDUE` in a sequence.

        Parameters
        ----------
        seq : str
            Valid amino acid sequence.
        raw_values : bool, optional
            If True, return the raw per-residue probabilities (0-1) as an array.
        return_sites_only : bool, optional
            If True, return only the list of predicted phosphosite positions.
        windowsize : int, optional
            +/- window over which a local phosphosite probability is extended.
        threshold : float, optional
            Probability threshold used for masking. Default 0.6.

        Returns
        -------
        numpy.ndarray or list
            Binary mask array by default; raw probability array if
            ``raw_values``; a list of site positions if ``return_sites_only``.
        """
        seq = seq.upper()
        probability = outputs.per_residue_class1_probability(
            self._raw_predict(seq), self.number_of_classes
        )

        if raw_values:
            return np.array(probability)

        if return_sites_only:
            return phospho_predictor_utils.return_hits(
                seq,
                probability,
                self.TARGET_RESIDUE,
                windowsize=windowsize,
                threshold=threshold,
                return_sites_only=True,
            )

        return np.array(
            phospho_predictor_utils.return_hits(
                seq,
                probability,
                self.TARGET_RESIDUE,
                windowsize=windowsize,
                threshold=threshold,
            )
        )
