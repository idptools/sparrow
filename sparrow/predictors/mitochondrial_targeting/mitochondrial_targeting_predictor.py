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

    NETWORK_PATH = (
        "networks/mitochondrial_targeting/"
        "mitochondrial_targeting_predictor_network_v{version}.pt"
    )
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
        seq = seq.upper()
        sub_seq = seq[0:_MTS_WINDOW]
        prediction = outputs.argmax_per_residue(self._raw_predict(sub_seq))
        prediction.extend([0] * (len(seq) - len(sub_seq)))
        return prediction
