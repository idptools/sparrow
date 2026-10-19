"""Scaled end-to-end distance predictor (ALBATROSS).

Thin wrapper over :class:`sparrow.predictors.base.BaseNetworkPredictor`; all
network loading is handled by the shared loader. The network predicts the
end-to-end distance divided by sqrt(N), and ``_from_raw`` multiplies it back by sqrt(N),
so ``predict`` and ``predict_many`` give the end-to-end distance itself.
"""

import numpy as np

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

    def _from_raw(self, seq, raw):
        """The end-to-end distance of a sequence, in Angstroms, from the network's raw output.

        The network predicts the end-to-end distance divided by sqrt(N), which
        :meth:`predict_scaled_re` returns; the prediction is that times sqrt(N), as
        ``Protein.predictor`` and ``batch_predict`` give it. See
        :meth:`~sparrow.predictors.base.BaseNetworkPredictor._from_raw`.

        Returns
        -------
        float
            The predicted end-to-end distance, in Angstroms.
        """
        return outputs.scalar_regression(raw) * np.sqrt(len(seq))
