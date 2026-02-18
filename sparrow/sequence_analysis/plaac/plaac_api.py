"""
Public API for PLAAC (Prion-Like Amino Acid Composition) scoring.

This module provides the user-facing functions for scoring protein
sequences for prion-like composition. Import from here rather than
from ``plaac.plaac`` directly.

Example
-------
>>> from sparrow.sequence_analysis.plaac.plaac_api import score_sequence, score_sequences
>>> result = score_sequence("MFKSGNQNN" * 20)
>>> result.core_score
...
"""

from typing import Dict, Optional, Sequence

from sparrow.sequence_analysis.plaac.plaac import (
    PLAACResult,
    PRDRegion,
    DisorderReport,
    score_sequence,
    score_sequences,
)

__all__ = [
    "score_sequence",
    "score_sequences",
    "PLAACResult",
    "PRDRegion",
    "DisorderReport",
]
