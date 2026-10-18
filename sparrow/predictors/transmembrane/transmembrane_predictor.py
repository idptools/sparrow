"""Transmembrane region predictor.

Thin wrapper over :class:`sparrow.predictors.base.BaseNetworkPredictor`; all
network loading is handled by the shared loader.
"""

from sparrow.predictors import outputs
from sparrow.predictors.base import BaseNetworkPredictor

DEFAULT_VERSION = "4"


class TransmembranePredictor(BaseNetworkPredictor):
    """Loads a network so :meth:`predict_transmembrane_regions` can classify TM regions."""

    NETWORK_PATH = "networks/transmembrane/transmembrane_predictor_network_v{version}.pt"
    DEFAULT_VERSION = DEFAULT_VERSION
    ARCHITECTURE = "MtM"

    def predict_transmembrane_regions(self, seq):
        """Predict per-residue transmembrane class.

        Parameters
        ----------
        seq : str
            Valid amino acid sequence.

        Returns
        -------
        list
            Per-residue predicted class (arg-max over network output).
        """
        return self.predict(seq)

    def _from_raw(self, seq, raw):
        """The prediction for a sequence, from the network's raw output for it.

        See :meth:`~sparrow.predictors.base.BaseNetworkPredictor._from_raw`.

        Returns
        -------
        list
            The predicted class of each residue (arg-max over the network output).
        """
        return outputs.argmax_per_residue(raw)

    # backwards-compatible alias for the previously-misspelled method name
    predict_transmebrane_regions = predict_transmembrane_regions
