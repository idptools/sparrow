"""Threonine phosphorylation predictor.

Thin wrapper over :class:`sparrow.predictors.phosphorylation.base.PhosphoPredictor`.
"""

from .base import PhosphoPredictor

DEFAULT_VERSION = "1"


class ThrPhosphorylationPredictor(PhosphoPredictor):
    """Predicts phosphothreonine sites via :meth:`predict_thr_phosphorylation`."""

    NETWORK_PATH = (
        "networks/phosphorylation/thr_phosphorylation_predictor_network_v{version}.pt"
    )
    DEFAULT_VERSION = DEFAULT_VERSION
    TARGET_RESIDUE = "T"

    def predict_thr_phosphorylation(
        self, seq, raw_values=False, return_sites_only=False, windowsize=4, threshold=0.6
    ):
        """Predict phosphothreonine sites for a sequence.

        See :meth:`PhosphoPredictor._predict_phospho` for parameter and return
        semantics.
        """
        return self._predict_phospho(
            seq,
            raw_values=raw_values,
            return_sites_only=return_sites_only,
            windowsize=windowsize,
            threshold=threshold,
        )
