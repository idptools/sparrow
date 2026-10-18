"""Every result a Protein computes is cached under its name and arguments, and
every call hands back a copy of it.

sparrow.batch relies on the cache: it computes results for many proteins and
stores each one where the method will look for it. Each test here checks that a
second call does not compute again, that the cached value is the one a fresh
Protein computes, and that different arguments are not given the cached result.
The copies are checked by changing what a call returned, in place, and calling
again.
"""

import dataclasses
import math

import numpy as np
import pytest

import sparrow.nardini.composition
import sparrow.properties.charge
import sparrow.properties.composition
import sparrow.properties.patches
import sparrow.properties.patterning
import sparrow.properties.profiles
import sparrow.sequence_analysis.elm
import sparrow.sequence_analysis.phospho_isoforms
import sparrow.sequence_analysis.sequence_complexity
from sparrow.data.amino_acids import AA_hydro_KD
from sparrow.nardini.table import FeatureVector
from sparrow.predictors.dssp.dssp_predictor import DSSPPredictor
from sparrow.properties import PROPERTY_NAMES
from sparrow.protein import Protein
from sparrow.sequence_analysis.plugins import BasePlugin
from sparrow.tools.utilities import cache_key, copy_of

# charged, Q-rich and S-rich stretches, so that each method's two argument sets
# below give different values
SEQUENCE = "MEEEKKKKSSSTTTDDDQQQQQQQQNNNNYYGRGRGSSSSSSSSAEDKRQQQQNQQQQY"

# every hydropathy value doubled: a second, different scale for compute_SHD_custom
DOUBLED_KD = {residue: 2 * value for residue, value in AA_hydro_KD.items()}

# few scrambles, so the NARDINI tests are quick
NARDINI_ARGUMENTS = {"seed": 1, "num_scrambles": 20}

# (method, the module and function it computes with, arguments, different arguments),
# for the Protein methods that take arguments
METHOD_CALLS = [
    ("charge_at_pH", sparrow.properties.charge, "charge_at_pH", {"pH": 6.0}, {"pH": 8.0}),
    (
        "compute_residue_fractions",
        sparrow.properties.composition,
        "fraction_in_set",
        {"residue_selector": ["Q", "N"]},
        {"residue_selector": ["S"]},
    ),
    ("compute_iwd", sparrow.properties.patterning, "iwd", {"target_residues": "Q"}, {"target_residues": "S"}),
    (
        "compute_patch_fraction",
        sparrow.properties.patches,
        "patch_fraction",
        {"residue_selector": "Q"},
        {"residue_selector": "S"},
    ),
    (
        "compute_SCD_x",
        sparrow.properties.patterning,
        "scd",
        {"group1": ["E", "D"], "group2": ["R", "K"]},
        {"group1": ["Q"], "group2": ["S"]},
    ),
    (
        "compute_SHD_custom",
        sparrow.properties.patterning,
        "shd",
        {"hydro_dict": AA_hydro_KD},
        {"hydro_dict": DOUBLED_KD},
    ),
    (
        "compute_iwd_charged_weighted",
        sparrow.properties.patterning,
        "iwd_charged_weighted",
        {"charge": "-"},
        {"charge": "+"},
    ),
    (
        "compute_kappa_x",
        sparrow.properties.patterning,
        "kappa_x",
        {"group1": ["E", "D"], "group2": ["R", "K"]},
        {"group1": ["Q"], "group2": ["S"]},
    ),
    (
        "generate_phosphoisoforms",
        sparrow.sequence_analysis.phospho_isoforms,
        "phosphoisoforms",
        {"mode": "custom", "phosphosites": [8]},
        {"mode": "custom", "phosphosites": [9]},
    ),
    (
        "low_complexity_domains",
        sparrow.sequence_analysis.sequence_complexity,
        "low_complexity_domains_holt",
        {"residue_selector": "Q", "minimum_length": 5},
        {"residue_selector": "S", "minimum_length": 5},
    ),
    ("linear_sequence_profile", sparrow.properties.profiles, "window_profile", {"mode": "FCR"}, {"mode": "NCPR"}),
    (
        "linear_composition_profile",
        sparrow.properties.profiles,
        "fraction_profile",
        {"composition_list": ["Q"]},
        {"composition_list": ["S"]},
    ),
    (
        "linear_property_profile",
        sparrow.properties.profiles,
        "scale_profile",
        {"mode": "hydropathy-kyte-1982"},
        {"mode": "hydropathy-kyte-1982", "window_size": 5},
    ),
]

# results that can be changed in place, and how to read each from a protein
CHANGEABLE_RESULTS = {
    "amino_acid_fractions": lambda p: p.amino_acid_fractions,
    "elms": lambda p: p.elms,
    "generate_phosphoisoforms": lambda p: p.generate_phosphoisoforms(),
    "low_complexity_domains": lambda p: p.low_complexity_domains(residue_selector="Q", minimum_length=5),
    "linear_sequence_profile": lambda p: p.linear_sequence_profile("FCR"),
    "linear_composition_profile": lambda p: p.linear_composition_profile(["Q"]),
    "linear_property_profile": lambda p: p.linear_property_profile("hydropathy-kyte-1982"),
    "plaac_prion_like_domains": lambda p: p.plaac_prion_like_domains(simple=False, core_length=30),
    "predictor.disorder": lambda p: p.predictor.disorder(),
    "predictor.disorder_domains": lambda p: p.predictor.disorder_domains(return_boundaries=True),
    "predictor.pLDDT": lambda p: p.predictor.pLDDT(),
    "predictor.dssp_helicity": lambda p: p.predictor.dssp_helicity(mode="both"),
    "predictor.serine_phosphorylation": lambda p: p.predictor.serine_phosphorylation(),
    "polymeric.afrc_end_to_end_distribution": lambda p: p.polymeric.afrc_end_to_end_distribution(),
    "nardini.vector": lambda p: p.nardini.vector(**NARDINI_ARGUMENTS),
    "nardini.patterning": lambda p: p.nardini.patterning(**NARDINI_ARGUMENTS),
    "nardini.composition": lambda p: p.nardini.composition(),
}


def same_value(first, second):
    """Whether two results are equal: arrays element by element, the fields of a
    dataclass one by one, and NaN equal to NaN, as PLAAC's smoothed tracks are
    at their ends."""
    if isinstance(first, FeatureVector):
        return np.array_equal(first.values, second.values)
    if dataclasses.is_dataclass(first):
        return same_value(dataclasses.asdict(first), dataclasses.asdict(second))
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(same_value(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(same_value(one, other) for one, other in zip(first, second))
    if isinstance(first, np.ndarray):
        return np.array_equal(first, second)
    if isinstance(first, float) and isinstance(second, float) and math.isnan(first) and math.isnan(second):
        return True
    return first == second


def change_in_place(value):
    """Change a result in place, as a caller might, whatever kind of container it is."""
    if isinstance(value, FeatureVector):
        value.values[0] = 99.0
    elif isinstance(value, np.ndarray):
        value[0] = 99
    elif isinstance(value, tuple):
        change_in_place(value[0])
    elif isinstance(value, list):
        value.append("changed")
    elif isinstance(value, dict):
        value["changed"] = True
    elif isinstance(value, set):
        value.clear()
    else:
        # a dataclass, such as the full PLAAC result
        value.sequence = "changed"


def count_calls(monkeypatch, module, function_name):
    """Count the calls to a module's function, which still does its work."""
    calls = []
    real_function = getattr(module, function_name)

    def counting_function(*args, **kwargs):
        calls.append(args)
        return real_function(*args, **kwargs)

    monkeypatch.setattr(module, function_name, counting_function)
    return calls


# .................................................................
# the keys
#
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


# .................................................................
# the cache
#
@pytest.mark.parametrize("name", PROPERTY_NAMES)
def test_each_scalar_property_is_cached(name):
    # a number is handed back as it is, so a cached one is the same object
    protein = Protein(SEQUENCE)
    assert getattr(protein, name) is getattr(protein, name)


@pytest.mark.parametrize(
    ("name", "module", "function_name"),
    [
        ("amino_acid_fractions", sparrow.properties.composition, "amino_acid_fractions"),
        ("elms", sparrow.sequence_analysis.elm, "find_all_elms"),
    ],
)
def test_each_container_property_is_computed_once(monkeypatch, name, module, function_name):
    calls = count_calls(monkeypatch, module, function_name)
    protein = Protein(SEQUENCE)
    first = getattr(protein, name)
    assert getattr(protein, name) == first
    assert len(calls) == 1


def test_a_method_without_arguments_is_cached():
    protein = Protein(SEQUENCE)
    first = protein.compute_bivariate_iwd_charged_weighted()
    assert protein.compute_bivariate_iwd_charged_weighted() is first
    assert first == Protein(SEQUENCE).compute_bivariate_iwd_charged_weighted()


@pytest.mark.parametrize(("method", "module", "function_name", "arguments", "other_arguments"), METHOD_CALLS)
def test_each_method_is_cached_by_its_arguments(
    monkeypatch, method, module, function_name, arguments, other_arguments
):
    protein = Protein(SEQUENCE)
    expected = getattr(Protein(SEQUENCE), method)(**arguments)
    expected_other = getattr(Protein(SEQUENCE), method)(**other_arguments)
    calls = count_calls(monkeypatch, module, function_name)

    # the second call returns the cached result, which is what a fresh protein computes
    first = getattr(protein, method)(**arguments)
    computed = len(calls)
    assert computed > 0
    assert same_value(getattr(protein, method)(**arguments), first)
    assert len(calls) == computed
    assert same_value(first, expected)

    # different arguments are computed for themselves, not given the cached result
    other = getattr(protein, method)(**other_arguments)
    assert len(calls) > computed
    assert not same_value(other, first)
    assert same_value(other, expected_other)


def test_plaac_runs_once_for_the_simple_and_full_forms(monkeypatch):
    import sparrow.protein

    calls = count_calls(monkeypatch, sparrow.protein, "plaac_score_sequence")
    protein = Protein(SEQUENCE)
    protein.plaac_prion_like_domains(simple=False, core_length=30)
    protein.plaac_prion_like_domains(simple=True, core_length=30)
    protein.plaac_prion_like_domains(simple=False, core_length=30)
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
    calls = count_calls(monkeypatch, DSSPPredictor, "predict_dssp_probabilities")
    protein = Protein(SEQUENCE)
    predict = getattr(protein.predictor, method)
    first = predict(mode="both", threshold=0.4, **{length_argument: 2})
    assert same_value(predict(mode="both", threshold=0.4, **{length_argument: 2}), first)
    assert len(calls) == 1

    fresh = getattr(Protein(SEQUENCE).predictor, method)(mode="both", threshold=0.4, **{length_argument: 2})
    assert same_value(first, fresh)

    # a different threshold is computed again rather than given the cached classes
    predict(mode="class", threshold=0.6, **{length_argument: 2})
    assert len(calls) == 3


def test_nardini_composition_is_cached_by_background_and_raw(monkeypatch):
    expected_zscores = Protein(SEQUENCE).nardini.composition()
    expected_raw = Protein(SEQUENCE).nardini.composition(raw=True)
    calls = count_calls(monkeypatch, sparrow.nardini.composition, "composition_values")

    protein = Protein(SEQUENCE)
    zscores = protein.nardini.composition()
    assert same_value(protein.nardini.composition(), zscores)
    assert len(calls) == 1

    raw = protein.nardini.composition(raw=True)
    assert len(calls) == 2
    assert not same_value(raw, zscores)
    assert same_value(raw, expected_raw)
    assert same_value(zscores, expected_zscores)


# .................................................................
# the copies
#
def test_a_copy_of_a_number_or_string_is_itself():
    for value in (1, 2.5, np.float32(1.5), "QQQ", None, True):
        assert copy_of(value) is value


def test_a_copy_of_a_container_is_a_copy_all_the_way_down():
    value = [np.array([1.0, 2.0]), {"key": [1, 2]}]
    copied = copy_of(value)
    assert copied is not value
    assert copied[0] is not value[0]
    assert copied[1]["key"] is not value[1]["key"]
    assert np.array_equal(copied[0], value[0])
    assert copied[1] == value[1]


@pytest.mark.parametrize("name", CHANGEABLE_RESULTS)
def test_changing_a_result_in_place_does_not_change_the_next_one(name):
    read = CHANGEABLE_RESULTS[name]
    protein = Protein(SEQUENCE)
    first = read(protein)
    expected = read(Protein(SEQUENCE))

    change_in_place(first)
    assert same_value(read(protein), expected), name


def test_changing_a_plugin_result_in_place_does_not_change_the_next_one():
    class ResidueList(BasePlugin):
        def calculate(self):
            return list(self.protein.sequence)

    protein = Protein(SEQUENCE)
    protein.plugin._PluginManager__plugins["ResidueList"] = ResidueList(protein)
    first = protein.plugin.ResidueList()
    first.append("changed")
    assert protein.plugin.ResidueList() == list(SEQUENCE)
