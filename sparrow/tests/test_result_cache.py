"""Every result a Protein computes is cached under its name and arguments.

sparrow.batch relies on this: it computes results for many proteins and stores
each one where the method will look for it. Each test here checks that a second
call returns the cached object, that the cached value is the one a fresh Protein
computes, and that different arguments are not given the cached result.
"""

import numpy as np
import pytest

from sparrow.data.amino_acids import AA_hydro_KD
from sparrow.predictors.dssp.dssp_predictor import DSSPPredictor
from sparrow.properties import PROPERTY_NAMES
from sparrow.protein import Protein
from sparrow.tools.utilities import cache_key

# charged, Q-rich and S-rich stretches, so that each method's two argument sets
# below give different values
SEQUENCE = "MEEEKKKKSSSTTTDDDQQQQQQQQNNNNYYGRGRGSSSSSSSSAEDKRQQQQNQQQQY"

# every hydropathy value doubled: a second, different scale for compute_SHD_custom
DOUBLED_KD = {residue: 2 * value for residue, value in AA_hydro_KD.items()}

# (method, arguments, different arguments), for the Protein methods that take arguments
METHOD_CALLS = [
    ("charge_at_pH", {"pH": 6.0}, {"pH": 8.0}),
    ("compute_residue_fractions", {"residue_selector": ["Q", "N"]}, {"residue_selector": ["S"]}),
    ("compute_iwd", {"target_residues": "Q"}, {"target_residues": "S"}),
    ("compute_patch_fraction", {"residue_selector": "Q"}, {"residue_selector": "S"}),
    ("compute_SCD_x", {"group1": ["E", "D"], "group2": ["R", "K"]}, {"group1": ["Q"], "group2": ["S"]}),
    ("compute_SHD_custom", {"hydro_dict": AA_hydro_KD}, {"hydro_dict": DOUBLED_KD}),
    ("compute_iwd_charged_weighted", {"charge": "-"}, {"charge": "+"}),
    ("compute_kappa_x", {"group1": ["E", "D"], "group2": ["R", "K"]}, {"group1": ["Q"], "group2": ["S"]}),
    ("generate_phosphoisoforms", {"mode": "custom", "phosphosites": [8]}, {"mode": "custom", "phosphosites": [9]}),
    (
        "low_complexity_domains",
        {"residue_selector": "Q", "minimum_length": 5},
        {"residue_selector": "S", "minimum_length": 5},
    ),
    ("linear_sequence_profile", {"mode": "FCR"}, {"mode": "NCPR"}),
    ("linear_composition_profile", {"composition_list": ["Q"]}, {"composition_list": ["S"]}),
    ("linear_property_profile", {"mode": "hydropathy-kyte-1982"}, {"mode": "hydropathy-kyte-1982", "window_size": 5}),
]


def same_value(first, second):
    """Whether two results are equal, comparing arrays element by element."""
    if isinstance(first, np.ndarray):
        return np.array_equal(first, second)
    return first == second


def test_equal_arguments_give_equal_keys():
    assert cache_key("f", ["E", "D"]) == cache_key("f", ("E", "D"))
    assert cache_key("f", {"A": 1.0, "G": 2.0}) == cache_key("f", {"G": 2.0, "A": 1.0})
    assert cache_key("f", {"E", "D"}) == cache_key("f", {"D", "E"})
    assert cache_key("f", np.array([1.0, 2.0])) == cache_key("f", np.array([1.0, 2.0]))


def test_different_arguments_give_different_keys():
    assert cache_key("f", ["E", "D"]) != cache_key("f", ["D", "E"])
    # a dict and a list of its items are different arguments
    assert cache_key("f", {"A": 1.0}) != cache_key("f", [("A", 1.0)])
    # the same numbers in a different dtype are a different array
    assert cache_key("f", np.array([1, 2])) != cache_key("f", np.array([1.0, 2.0]))
    assert cache_key("f", np.array([1.0, 2.0])) != cache_key("f", np.array([[1.0, 2.0]]))


def test_a_key_is_hashable_even_with_nested_arguments():
    hash(cache_key("f", {"kwargs": {"bg_freqs": {"A": 0.1}, "sites": [1, 2]}}))


def test_an_unhashable_argument_fails_loudly():
    with pytest.raises(TypeError):
        hash(cache_key("f", bytearray(b"Q")))


@pytest.mark.parametrize("name", PROPERTY_NAMES + ("amino_acid_fractions", "elms"))
def test_each_property_is_cached(name):
    protein = Protein(SEQUENCE)
    assert getattr(protein, name) is getattr(protein, name)


def test_a_method_without_arguments_is_cached():
    protein = Protein(SEQUENCE)
    first = protein.compute_bivariate_iwd_charged_weighted()
    assert protein.compute_bivariate_iwd_charged_weighted() is first
    assert first == Protein(SEQUENCE).compute_bivariate_iwd_charged_weighted()


@pytest.mark.parametrize(("method", "arguments", "other_arguments"), METHOD_CALLS)
def test_each_method_is_cached_by_its_arguments(method, arguments, other_arguments):
    protein = Protein(SEQUENCE)
    first = getattr(protein, method)(**arguments)

    # the second call returns the cached result, which is what a fresh protein computes
    assert getattr(protein, method)(**arguments) is first
    assert same_value(first, getattr(Protein(SEQUENCE), method)(**arguments))

    # different arguments are computed for themselves, not given the cached result
    other = getattr(protein, method)(**other_arguments)
    assert not same_value(other, first)
    assert same_value(other, getattr(Protein(SEQUENCE), method)(**other_arguments))


def test_plaac_runs_once_for_the_simple_and_full_forms(monkeypatch):
    import sparrow.protein

    calls = []
    real_score_sequence = sparrow.protein.plaac_score_sequence

    def counting_score_sequence(sequence, **kwargs):
        calls.append(kwargs)
        return real_score_sequence(sequence, **kwargs)

    monkeypatch.setattr(sparrow.protein, "plaac_score_sequence", counting_score_sequence)

    protein = Protein(SEQUENCE)
    full = protein.plaac_prion_like_domains(simple=False, core_length=30)
    protein.plaac_prion_like_domains(simple=True, core_length=30)
    assert protein.plaac_prion_like_domains(simple=False, core_length=30) is full
    assert len(calls) == 1

    # a different option is a different run
    protein.plaac_prion_like_domains(simple=False, core_length=40)
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("method", "length_argument"),
    [
        ("dssp_helicity", "minimum_helical_length"),
        ("dssp_coil", "minimum_coil_length"),
        ("dssp_extended", "minimum_extended_length"),
    ],
)
def test_each_dssp_method_runs_the_network_once_for_its_arguments(monkeypatch, method, length_argument):
    calls = []
    real_probabilities = DSSPPredictor.predict_dssp_probabilities

    def counting_probabilities(self, sequence):
        calls.append(sequence)
        return real_probabilities(self, sequence)

    monkeypatch.setattr(DSSPPredictor, "predict_dssp_probabilities", counting_probabilities)

    protein = Protein(SEQUENCE)
    predict = getattr(protein.predictor, method)
    first = predict(mode="both", threshold=0.4, **{length_argument: 2})
    assert predict(mode="both", threshold=0.4, **{length_argument: 2}) is first
    assert len(calls) == 1

    fresh = getattr(Protein(SEQUENCE).predictor, method)(mode="both", threshold=0.4, **{length_argument: 2})
    assert np.array_equal(first[0], fresh[0])
    assert np.array_equal(first[1], fresh[1])

    # a different threshold is computed again rather than given the cached classes
    predict(mode="class", threshold=0.6, **{length_argument: 2})
    assert len(calls) == 3


def test_nardini_composition_is_cached_by_background_and_raw():
    protein = Protein(SEQUENCE)
    zscores = protein.nardini.composition()
    assert protein.nardini.composition() is zscores

    raw = protein.nardini.composition(raw=True)
    assert raw is not zscores
    assert not np.array_equal(raw.values, zscores.values)
    assert np.array_equal(raw.values, Protein(SEQUENCE).nardini.composition(raw=True).values)
    assert np.array_equal(zscores.values, Protein(SEQUENCE).nardini.composition().values)
