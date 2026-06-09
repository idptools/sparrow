"""Radius-of-gyration predictor (ALBATROSS).

Thin wrapper over :class:`sparrow.predictors.base.BaseNetworkPredictor`; all
network loading is handled by the shared loader.
"""

from sparrow.predictors import outputs
from sparrow.predictors.base import BaseNetworkPredictor

DEFAULT_VERSION = "2"


class RgPredictor(BaseNetworkPredictor):
    """Loads a network so that :meth:`predict_rg` can predict radius of gyration."""

    NETWORK_PATH = "networks/rg/rg_network_v{version}.pt"
    DEFAULT_VERSION = DEFAULT_VERSION
    ARCHITECTURE = "MtO"

    def predict_rg(self, seq):
        """Predict the radius of gyration for a sequence.

        Parameters
        ----------
        seq : str
            Valid amino acid sequence.

        Returns
        -------
        float
            Predicted radius of gyration.
        """
        return outputs.scalar_regression(self._raw_predict(seq))
