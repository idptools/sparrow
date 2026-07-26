"""Tests for grammar feature-vector extraction.

``sparrow.sequence_analysis.grammar`` is deprecated in favour of
:mod:`sparrow.nardini`, but these tests keep the deprecated behaviour pinned for
as long as it ships. The deprecation warning itself is asserted in
``test_regressions.py``; it is silenced here so the suite output stays readable.
"""

import numpy as np
import pytest

from sparrow import Protein
from sparrow.sequence_analysis import grammar

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

SEQ = "MEEEKKKKSSSTTTDDDQQQQNNNNGGGGSSSS"


def test_feature_vector_is_float32_array():
    vec = grammar.compute_feature_vector(SEQ, num_scrambles=64, seed=1)
    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.ndim == 1 and len(vec) > 0


def test_feature_vector_deterministic_with_seed():
    a = grammar.compute_feature_vector(SEQ, num_scrambles=64, seed=7)
    b = grammar.compute_feature_vector(SEQ, num_scrambles=64, seed=7)
    assert np.array_equal(a, b)


def test_feature_names_match_vector_length():
    vec, names = grammar.compute_feature_vector(
        SEQ, num_scrambles=64, seed=1, return_feature_names=True
    )
    assert len(names) == len(vec)
    assert all(isinstance(n, str) for n in names)


def test_accepts_sequence_or_protein():
    from_seq = grammar.compute_feature_vector(SEQ, num_scrambles=64, seed=1)
    from_protein = grammar.compute_feature_vector(Protein(SEQ), num_scrambles=64, seed=1)
    assert np.allclose(from_seq, from_protein)


def test_protein_extract_feature_vector_matches_function():
    p = Protein(SEQ)
    via_method = p.extract_feature_vector(num_scrambles=64, seed=1)
    via_function = grammar.compute_feature_vector(SEQ, num_scrambles=64, seed=1)
    assert np.allclose(via_method, via_function)


def test_include_raw_adds_features():
    base = grammar.compute_feature_vector(SEQ, num_scrambles=64, seed=1)
    with_raw = grammar.compute_feature_vector(
        SEQ, num_scrambles=64, seed=1, include_raw=True
    )
    assert len(with_raw) > len(base)
