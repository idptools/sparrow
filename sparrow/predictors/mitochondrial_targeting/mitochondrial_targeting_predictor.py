"""Mitochondrial targeting sequence predictor.

Thin wrapper over :class:`sparrow.predictors.base.BaseNetworkPredictor`; all
network loading is handled by the shared loader. Only the N-terminal 168
residues are scored (mitochondrial targeting sequences are N-terminal); the
remaining positions are reported as 0.
"""

from sparrow.predictors import outputs
from sparrow.predictors.base import BaseNetworkPredictor

DEFAULT_VERSION = "1"

# mitochondrial targeting sequences are N-terminal; only this many leading
# residues are scored by the network
_MTS_WINDOW = 168


class MitochondrialTargetingPredictor(BaseNetworkPredictor):
    """Loads a network so :meth:`predict_mitochondrial_targeting` can classify MTSs."""

    NETWORK_PATH = "networks/mitochondrial_targeting/mitochondrial_targeting_predictor_network_v{version}.pt"
    DEFAULT_VERSION = DEFAULT_VERSION
    ARCHITECTURE = "MtM"

    def predict_mitochondrial_targeting(self, seq):
        """Predict per-residue mitochondrial-targeting class.

        Parameters
        ----------
        seq : str
            Valid amino acid sequence.

        Returns
        -------
        list
            Per-residue binary classification (0/1); positions beyond the first
            ``168`` residues are 0.
        """
        return self.predict(seq)

    def _network_input(self, seq):
        """The first 168 residues, upper-cased: mitochondrial targeting sequences are N-terminal.

        See :meth:`~sparrow.predictors.base.BaseNetworkPredictor._network_input`.

        Returns
        -------
        str
            What the network is run on.
        """
        return seq.upper()[0:_MTS_WINDOW]

    def _from_raw(self, seq, raw):
        """The prediction for a sequence, from the network's raw output for its first residues.

        See :meth:`~sparrow.predictors.base.BaseNetworkPredictor._from_raw`.

        Returns
        -------
        list
            The predicted class (0/1) of each residue; residues beyond the
            first 168 are 0.
        """
        prediction = outputs.argmax_per_residue(raw)
        prediction.extend([0] * (len(seq) - len(prediction)))
        return prediction
