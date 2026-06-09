"""Prefactor predictor (ALBATROSS).

Thin wrapper over :class:`sparrow.predictors.base.BaseNetworkPredictor`; all
network loading is handled by the shared loader.
"""

from sparrow.predictors import outputs
from sparrow.predictors.base import BaseNetworkPredictor

DEFAULT_VERSION = "2"


class PrefactorPredictor(BaseNetworkPredictor):
    """Loads a network so that :meth:`predict_prefactor` can predict the prefactor."""

    NETWORK_PATH = "networks/prefactor/prefactor_network_v{version}.pt"
    DEFAULT_VERSION = DEFAULT_VERSION
    ARCHITECTURE = "MtO"

    def predict_prefactor(self, seq):
        """Predict the polymer-model prefactor for a sequence.

        Parameters
        ----------
        seq : str
            Valid amino acid sequence.

        Returns
        -------
        float
            Predicted prefactor.
        """
        return outputs.scalar_regression(self._raw_predict(seq))
