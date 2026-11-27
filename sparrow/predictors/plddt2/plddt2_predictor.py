"""pLDDT predictor, version 2: sparrow's own prediction of AlphaFold2's per-residue confidence.

Thin wrapper over :class:`sparrow.predictors.base.BaseNetworkPredictor`; all
network loading is handled by the shared loader. The network was trained with
PARROT 2, so it records its own architecture and encoding, and loading it needs
PARROT 2 installed (see
:data:`sparrow.predictors.network_loader.PARROT2_INSTALL_COMMAND`).
sparrow/data/networks/plddt2/README.md describes the network and how it was
trained.
"""

import numpy as np

from sparrow.predictors.base import BaseNetworkPredictor

DEFAULT_VERSION = "1"

# pLDDT runs from 0 to 100 (Jumper et al. 2021, Nature 596:583). The network was
# trained on pLDDT / 100, so its output is multiplied by PLDDT_SCALE, and a
# prediction is held to pLDDT's range: on the 26.2 million residues of the
# held-out test set, 16,988 (0.06%) were predicted above 100 and none below 0.
PLDDT_SCALE = 100.0
PLDDT_MINIMUM = 0.0
PLDDT_MAXIMUM = 100.0


class PLDDT2Predictor(BaseNetworkPredictor):
    """Loads a network so :meth:`predict_plddt2` can predict the pLDDT of every residue."""

    NETWORK_PATH = "networks/plddt2/plddt2_network_v{version}.pt"
    DEFAULT_VERSION = DEFAULT_VERSION

    def predict_plddt2(self, seq):
        """Predict the pLDDT AlphaFold2 would give each residue of a sequence.

        Parameters
        ----------
        seq : str
            Valid amino acid sequence.

        Returns
        -------
        numpy.ndarray
            The predicted pLDDT of each residue (0 to 100), as
            :meth:`_from_raw` makes it.
        """
        return self.predict(seq)

    def _from_raw(self, seq, raw):
        """The prediction for a sequence, from PARROT's prediction for it.

        See :meth:`~sparrow.predictors.base.BaseNetworkPredictor._from_raw`.

        Parameters
        ----------
        seq : str
            The sequence.
        raw : numpy.ndarray
            PARROT's prediction: pLDDT / 100 for each residue, a float32 array
            of length ``len(seq)``.

        Returns
        -------
        numpy.ndarray
            The predicted pLDDT of each residue, float32, on AlphaFold's 0-100
            scale (the network's output times 100), clipped to 0-100. The
            network is a regression, so a few residues are predicted just
            outside that range (0.06% above 100 on the test set).
        """
        return np.clip(raw * PLDDT_SCALE, PLDDT_MINIMUM, PLDDT_MAXIMUM)
