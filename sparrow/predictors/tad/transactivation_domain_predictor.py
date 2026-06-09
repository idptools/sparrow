"""Transactivation domain (TAD) predictor.

Thin wrapper over :class:`sparrow.predictors.base.BaseNetworkPredictor`; all
network loading is handled by the shared loader.
"""

from sparrow.predictors import outputs
from sparrow.predictors.base import BaseNetworkPredictor

DEFAULT_VERSION = "1"


class TADPredictor(BaseNetworkPredictor):
    """Loads a network so :meth:`predict_transactivation_domains` can predict TADs."""

    NETWORK_PATH = "networks/transactivation_domains/tad_predictor_network_v{version}.pt"
    DEFAULT_VERSION = DEFAULT_VERSION
    ARCHITECTURE = "MtM"

    def predict_transactivation_domains(self, seq):
        """Predict per-residue transactivation-domain probability.

        Parameters
        ----------
        seq : str
            Valid amino acid sequence.

        Returns
        -------
        list of float
            Per-residue probability of being in a transactivation domain.
        """
        return outputs.per_residue_class1_probability(
            self._raw_predict(seq), self.number_of_classes
        )
