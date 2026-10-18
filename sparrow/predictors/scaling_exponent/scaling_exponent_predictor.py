"""Scaling-exponent (nu) predictor (ALBATROSS).

Thin wrapper over :class:`sparrow.predictors.base.BaseNetworkPredictor`; all
network loading is handled by the shared loader.
"""

from sparrow.predictors import outputs
from sparrow.predictors.base import BaseNetworkPredictor

DEFAULT_VERSION = "2"


class ScalingExponentPredictor(BaseNetworkPredictor):
    """Loads a network so :meth:`predict_scaling_exponent` can predict nu."""

    NETWORK_PATH = "networks/scaling_exponent/scaling_exponent_network_v{version}.pt"
    DEFAULT_VERSION = DEFAULT_VERSION
    ARCHITECTURE = "MtO"

    def predict_scaling_exponent(self, seq):
        """Predict the Flory scaling exponent (nu) for a sequence.

        Parameters
        ----------
        seq : str
            Valid amino acid sequence.

        Returns
        -------
        float
            Predicted scaling exponent.
        """
        return self.predict(seq)

    def _from_raw(self, seq, raw):
        """The prediction for a sequence, from the network's raw output for it.

        See :meth:`~sparrow.predictors.base.BaseNetworkPredictor._from_raw`.

        Returns
        -------
        float
            The predicted scaling exponent.
        """
        return outputs.scalar_regression(raw)
