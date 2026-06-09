"""Tests for sparrow.sequence_analysis.compression."""

import pytest

from sparrow.sequence_analysis.compression import KMerCompressor

SEQUENCES = [
    "A",
    "AAAAAA",
    "ACACACACQQQQ",
    "MEEEKKKKSSSTTTDDDQQQQNNNN",
    "ACDEFGHIKLMNPQRSTVWY" * 3,
]


@pytest.fixture
def compressor():
    return KMerCompressor()


@pytest.mark.parametrize("seq", SEQUENCES)
def test_compress_decompress_roundtrip(compressor, seq):
    tokens = compressor.compress(seq)
    assert compressor.decompress(tokens) == seq


def test_empty_string_roundtrip(compressor):
    tokens = compressor.compress("")
    assert compressor.decompress(tokens) == ""


def test_compress_rejects_non_alphabet(compressor):
    with pytest.raises(ValueError):
        compressor.compress("ACB")  # B is not in the 20-letter alphabet


def test_compression_ratio_is_finite(compressor):
    seq = "ACACACACACACACAC"
    tokens = compressor.compress(seq)
    ratio = KMerCompressor.compression_ratio(seq, tokens)
    assert ratio > 0.0


def test_repetitive_sequence_compresses_better_than_random(compressor):
    repetitive = "AC" * 40
    diverse = ("ACDEFGHIKLMNPQRSTVWY" * 4)
    r_rep = KMerCompressor.compression_ratio(repetitive, compressor.compress(repetitive))
    r_div = KMerCompressor.compression_ratio(diverse, compressor.compress(diverse))
    assert r_rep < r_div


def test_custom_alphabet_duplicates_rejected():
    with pytest.raises(ValueError):
        KMerCompressor(alphabet="AAB")
