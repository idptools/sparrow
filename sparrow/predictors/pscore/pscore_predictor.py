"""PScore predictor (phase-separation propensity).

Thin wrapper over :class:`sparrow.predictors.base.BaseNetworkPredictor`; all
network loading is handled by the shared loader.
"""

from sparrow.predictors.base import BaseNetworkPredictor

DEFAULT_VERSION = "4"


class PScorePredictor(BaseNetworkPredictor):
    """Loads a network so that :meth:`predict_pscore` can predict the PScore."""

    NETWORK_PATH = "networks/pscore/pscore_predictor_network_v{version}.pt"
    DEFAULT_VERSION = DEFAULT_VERSION
    ARCHITECTURE = "MtM"

    def predict_pscore(self, seq):
        """Predict the raw per-residue PScore output for a sequence.

        Parameters
        ----------
        seq : str
            Valid amino acid sequence.

        Returns
        -------
        numpy.ndarray
            Flattened network output.
        """
        return self.predict(seq)

    def _from_raw(self, seq, raw):
        """The prediction for a sequence, from the network's raw output for it.

        See :meth:`~sparrow.predictors.base.BaseNetworkPredictor._from_raw`.

        Returns
        -------
        numpy.ndarray
            The PScore of each residue, on its native 0-6 scale.
        """
        return raw.flatten()
