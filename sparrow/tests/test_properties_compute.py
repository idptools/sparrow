"""Tests for sparrow.properties.compute, the Protein properties of a batch by name."""

import numpy as np
import pytest

import sparrow.properties as properties
from sparrow import Protein
from sparrow.sparrow_exceptions import SparrowException

SEQUENCES = [
    "MEEEKKKKSSSTTTDDDQQQQNNNNGGG",
    "",
    "A",
    "gsgsRGRGyy",
    "ACDEFGHIKLMNPQRSTVWY" * 4,
    "MSDNGPQSNQRSAPRITFGGPTDSTDNNQNGGRNGARPKQRRPQGLPNNTASWFTALTQHGK",
]

# Protein properties that are not one number per sequence, so compute() leaves
# them out
NOT_SCALAR = {"amino_acid_fractions", "sequence", "plugin", "nardini", "predictor", "polymeric", "elms"}


@pytest.mark.parametrize("name", properties.PROPERTY_NAMES)
def test_each_property_is_the_protein_property(name):
    computed = properties.compute(SEQUENCES, [name])[name]
    expected = [getattr(Protein(sequence), name) for sequence in SEQUENCES]
    # assert_array_equal treats NaN as equal to NaN, which the empty sequence needs
    np.testing.assert_array_equal(computed, expected)


def test_every_scalar_protein_property_has_a_name():
    protein_properties = {name for name, value in vars(Protein).items() if isinstance(value, property)}
    assert set(properties.PROPERTY_NAMES) == protein_properties - NOT_SCALAR


def test_no_names_computes_everything():
    table = properties.compute(SEQUENCES)
    assert tuple(table) == properties.PROPERTY_NAMES
    for values in table.values():
        assert values.dtype == np.float64
        assert values.shape == (len(SEQUENCES),)


def test_names_come_back_in_the_order_asked_for_and_once():
    table = properties.compute(SEQUENCES, ["SCD", "FCR", "SCD", "kappa"])
    assert list(table) == ["SCD", "FCR", "kappa"]


def test_one_name_on_its_own():
    np.testing.assert_array_equal(
        properties.compute(SEQUENCES, "kappa")["kappa"],
        properties.compute(SEQUENCES, ["kappa"])["kappa"],
    )


def test_a_bare_string_is_a_batch_of_one():
    table = properties.compute(SEQUENCES[0], ["FCR", "SHD"])
    assert table["FCR"].tolist() == [Protein(SEQUENCES[0]).FCR]
    assert table["SHD"].tolist() == [Protein(SEQUENCES[0]).SHD]


def test_an_unknown_name_raises_before_anything_is_computed():
    with pytest.raises(SparrowException, match="not_a_property"):
        properties.compute(["MEEK", "XXXX"], ["FCR", "not_a_property"])


def test_a_non_standard_residue_raises_naming_the_sequence():
    with pytest.raises(SparrowException, match="Sequence 1 of 2"):
        properties.compute(["MEEK", "MEXK"], ["FCR"])
