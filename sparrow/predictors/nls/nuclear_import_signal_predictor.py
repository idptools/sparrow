"""Nuclear import signal (NLS) predictor.

Thin wrapper over :class:`sparrow.predictors.base.BaseNetworkPredictor`; all
network loading is handled by the shared loader.
"""

from sparrow.predictors import outputs
from sparrow.predictors.base import BaseNetworkPredictor

DEFAULT_VERSION = "1"


class NLSPredictor(BaseNetworkPredictor):
    """Loads a network so :meth:`predict_nuclear_import_signal` can predict NLSs."""

    NETWORK_PATH = "networks/nuclear_import_signal/nls_predictor_network_v{version}.pt"
    DEFAULT_VERSION = DEFAULT_VERSION
    ARCHITECTURE = "MtM"

    def predict_nuclear_import_signal(self, seq):
        """Predict per-residue nuclear-import-signal probability.

        Parameters
        ----------
        seq : str
            Valid amino acid sequence.

        Returns
        -------
        list of float
            Per-residue probability of being in an NLS.
        """
        return self.predict(seq)

    def _from_raw(self, seq, raw):
        """The prediction for a sequence, from the network's raw output for it.

        See :meth:`~sparrow.predictors.base.BaseNetworkPredictor._from_raw`.

        Returns
        -------
        list of float
            The probability that each residue is in a nuclear import signal.
        """
        return outputs.per_residue_class1_probability(raw, self.number_of_classes)
