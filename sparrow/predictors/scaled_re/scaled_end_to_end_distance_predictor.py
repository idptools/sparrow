"""Scaled end-to-end distance predictor (ALBATROSS).

Thin wrapper over :class:`sparrow.predictors.base.BaseNetworkPredictor`; all
network loading is handled by the shared loader. The length-rescaling
(``* sqrt(N)``) is applied by the caller in :mod:`sparrow.predictors`.
"""

from sparrow.predictors import outputs
from sparrow.predictors.base import BaseNetworkPredictor

DEFAULT_VERSION = "2"


class ScaledRePredictor(BaseNetworkPredictor):
    """Loads a network so that :meth:`predict_scaled_re` can predict scaled Re."""

    NETWORK_PATH = "networks/scaled_re/scaled_re_network_v{version}.pt"
    DEFAULT_VERSION = DEFAULT_VERSION
    ARCHITECTURE = "MtO"

    def predict_scaled_re(self, seq):
        """Predict the length-normalized end-to-end distance for a sequence.

        Parameters
        ----------
        seq : str
            Valid amino acid sequence.

        Returns
        -------
        float
            Predicted scaled end-to-end distance.
        """
        return outputs.scalar_regression(self._raw_predict(seq))
