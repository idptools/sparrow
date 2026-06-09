"""Asphericity predictor (ALBATROSS).

Thin wrapper over :class:`sparrow.predictors.base.BaseNetworkPredictor`; all
network loading is handled by the shared loader.
"""

from sparrow.predictors import outputs
from sparrow.predictors.base import BaseNetworkPredictor

DEFAULT_VERSION = "2"


class AsphericityPredictor(BaseNetworkPredictor):
    """Loads a network so that :meth:`predict_asphericity` can predict asphericity."""

    NETWORK_PATH = "networks/asphericity/asphericity_network_v{version}.pt"
    DEFAULT_VERSION = DEFAULT_VERSION
    ARCHITECTURE = "MtO"

    def predict_asphericity(self, seq):
        """Predict the asphericity for a sequence.

        Parameters
        ----------
        seq : str
            Valid amino acid sequence.

        Returns
        -------
        float
            Predicted asphericity.
        """
        return outputs.scalar_regression(self._raw_predict(seq))
