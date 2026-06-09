"""Scaled radius-of-gyration predictor (ALBATROSS).

Thin wrapper over :class:`sparrow.predictors.base.BaseNetworkPredictor`; all
network loading is handled by the shared loader. The length-rescaling
(``* sqrt(N)``) is applied by the caller in :mod:`sparrow.predictors`.
"""

from sparrow.predictors import outputs
from sparrow.predictors.base import BaseNetworkPredictor

DEFAULT_VERSION = "2"


class ScaledRgPredictor(BaseNetworkPredictor):
    """Loads a network so that :meth:`predict_scaled_rg` can predict scaled Rg."""

    NETWORK_PATH = "networks/scaled_rg/scaled_rg_network_v{version}.pt"
    DEFAULT_VERSION = DEFAULT_VERSION
    ARCHITECTURE = "MtO"

    def predict_scaled_rg(self, seq):
        """Predict the length-normalized radius of gyration for a sequence.

        Parameters
        ----------
        seq : str
            Valid amino acid sequence.

        Returns
        -------
        float
            Predicted scaled radius of gyration.
        """
        return outputs.scalar_regression(self._raw_predict(seq))
