"""End-to-end distance predictor (ALBATROSS).

Thin wrapper over :class:`sparrow.predictors.base.BaseNetworkPredictor`; all
network loading is handled by the shared loader.
"""

from sparrow.predictors import outputs
from sparrow.predictors.base import BaseNetworkPredictor

DEFAULT_VERSION = "2"


class RePredictor(BaseNetworkPredictor):
    """Loads a network so that :meth:`predict_re` can predict end-to-end distance."""

    NETWORK_PATH = "networks/re/re_network_v{version}.pt"
    DEFAULT_VERSION = DEFAULT_VERSION
    ARCHITECTURE = "MtO"

    def predict_re(self, seq):
        """Predict the end-to-end distance for a sequence.

        Parameters
        ----------
        seq : str
            Valid amino acid sequence.

        Returns
        -------
        float
            Predicted end-to-end distance.
        """
        return outputs.scalar_regression(self._raw_predict(seq))
