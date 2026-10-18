"""sparrow.batch: many sequences in, Protein objects out, with the named values computed."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import sparrow
import sparrow.batching
from sparrow import Protein
from sparrow.batching import _available_names
from sparrow.data.amino_acids import AA_hydro_KD
from sparrow.nardini.table import FeatureVector
from sparrow.properties import PROPERTY_NAMES
from sparrow.sparrow_exceptions import SparrowException

# lengths either side of MIN_LENGTH_ALBATROSS_RE_RG (35), so both the scaled and
# the unscaled ALBATROSS networks are used, with charged, Q-rich and S-rich stretches
SEQUENCES = {
    "short": "MEEEKKKKSSSTTTDDDQQQQ",
    "medium": "MEEEKKKKSSSTTTDDDQQQQQQQQNNNNYYGRGRGSSSSSSSSAEDKRQQQQNQQQQY",
    "long": "MKYLAAYLLLNAAGNTPDATKIKAILESVGIEIEDEKVSSVLSALEGKSVDELITEGNEKLAAVPAAGPASAGGAAAASGDAAAEEEKEEEAAEESDDDMGFGLFD",
}

# a prediction made in a batch differs from one made alone by the precision of
# the networks' 32-bit arithmetic
ALBATROSS_RELATIVE_TOLERANCE = 1e-6
ALBATROSS_NAMES = [
    "predictor.radius_of_gyration",
    "predictor.end_to_end_distance",
    "predictor.asphericity",
    "predictor.prefactor",
    "predictor.scaling_exponent",
]

# few scrambles, so the NARDINI tests are quick; a seed, so they are reproducible
NARDINI_ARGUMENTS = {"seed": 1, "num_scrambles": 50}

# arguments for each name whose method needs some
ARGUMENTS = {
    "compute_SCD_x": {"group1": ["E", "D"], "group2": ["R", "K"]},
    "compute_SHD_custom": {"hydro_dict": AA_hydro_KD},
    "compute_iwd": {"target_residues": "Q"},
    "compute_iwd_charged_weighted": {"charge": "-"},
    "compute_kappa_x": {"group1": ["E", "D"], "group2": ["R", "K"]},
    "compute_patch_fraction": {"residue_selector": "Q"},
    "compute_residue_fractions": {"residue_selector": ["Q", "N"]},
    "linear_composition_profile": {"composition_list": ["Q"]},
    "linear_property_profile": {"mode": "hydropathy-kyte-1982"},
    "linear_sequence_profile": {"mode": "FCR"},
    "low_complexity_domains": {"residue_selector": "Q", "minimum_length": 5},
    "predictor.low_complexity_domains": {"residue_selector": "Q", "minimum_length": 5},
    "nardini.vector": NARDINI_ARGUMENTS,
    "nardini.patterning": NARDINI_ARGUMENTS,
    "nardini.gin_cluster": NARDINI_ARGUMENTS,
    "nardini.gin_report": NARDINI_ARGUMENTS,
}


def read_back(protein, name, arguments):
    """The value a name refers to, read from a protein as a user would."""
    accessor, _, attribute = name.rpartition(".")
    owner = getattr(protein, accessor) if accessor else protein
    value = getattr(owner, attribute)
    if sparrow.batching._request(name, arguments).is_property:
        return value
    return value(**arguments)


def assert_same(first, second, name):
    """Assert two values are equal: anything built from an ALBATROSS prediction to
    the networks' precision, everything else exactly."""
    from_albatross = name in ALBATROSS_NAMES or name.startswith("polymeric.predicted")
    if isinstance(first, FeatureVector):
        assert np.array_equal(first.values, second.values), name
    elif dataclasses.is_dataclass(first):
        assert_same(dataclasses.asdict(first), dataclasses.asdict(second), name)
    elif isinstance(first, dict):
        assert first.keys() == second.keys(), name
        for key in first:
            assert_same(first[key], second[key], name)
    elif isinstance(first, (tuple, list)) and not isinstance(first, str):
        assert len(first) == len(second), name
        for one, other in zip(first, second):
            assert_same(one, other, name)
    elif from_albatross:
        # a distribution is compared relative to its peak, where its precision is set
        scale = np.max(np.abs(second))
        assert np.allclose(
            first, second, rtol=ALBATROSS_RELATIVE_TOLERANCE, atol=ALBATROSS_RELATIVE_TOLERANCE * scale
        ), name
    elif isinstance(first, np.ndarray):
        assert np.array_equal(first, second), name
    else:
        assert first == second, name


# .................................................................
# what goes in, and what comes out
#
def test_sparrow_batch_is_the_batch_function():
    assert sparrow.batch is sparrow.batching.batch


def test_a_list_gives_a_list_in_the_same_order():
    proteins = sparrow.batch(list(SEQUENCES.values()), ["FCR"])
    assert isinstance(proteins, list)
    assert [protein.sequence for protein in proteins] == list(SEQUENCES.values())


def test_a_dict_gives_a_dict_with_the_same_keys_in_the_same_order():
    proteins = sparrow.batch(SEQUENCES, ["FCR"])
    assert list(proteins) == list(SEQUENCES)
    assert all(proteins[key].sequence == sequence for key, sequence in SEQUENCES.items())


def test_a_protein_given_is_filled_in_and_returned():
    protein = Protein(SEQUENCES["medium"])
    proteins = sparrow.batch([protein, SEQUENCES["long"]], ["FCR"])
    assert proteins[0] is protein
    assert isinstance(proteins[1], Protein)


@pytest.mark.parametrize("as_path", [str, Path])
def test_a_fasta_file_gives_a_dict_keyed_by_header(tmp_path, as_path):
    fasta = tmp_path / "sequences.fasta"
    fasta.write_text("".join(f">{key}\n{sequence}\n" for key, sequence in SEQUENCES.items()))
    proteins = sparrow.batch(as_path(fasta), ["FCR"])
    assert list(proteins) == list(SEQUENCES)
    assert proteins["long"].FCR == Protein(SEQUENCES["long"]).FCR


def test_one_sequence_is_refused_with_a_pointer_to_protein():
    with pytest.raises(SparrowException, match=r"is not a file.*Protein\(sequence\)"):
        sparrow.batch(SEQUENCES["medium"], ["FCR"])


@pytest.mark.parametrize("empty", [[], {}])
def test_no_sequences_is_refused(empty):
    with pytest.raises(SparrowException, match="No sequences"):
        sparrow.batch(empty, ["FCR"])


def test_an_item_that_is_not_a_sequence_is_refused_by_its_key():
    with pytest.raises(SparrowException, match="Item 'bad' is a int"):
        sparrow.batch({"good": SEQUENCES["medium"], "bad": 7}, ["FCR"])


# .................................................................
# what is asked for
#
@pytest.mark.parametrize(
    "names", ["FCR", ["FCR"], ("FCR", "FCR"), {"FCR": {}}, {"FCR": None}, [{"FCR": None}], ["kappa", {"FCR": {}}]]
)
def test_a_name_a_list_and_a_dict_are_all_accepted(names):
    proteins = sparrow.batch(SEQUENCES, names)
    assert proteins["medium"].FCR == Protein(SEQUENCES["medium"]).FCR


@pytest.mark.parametrize(
    ("name", "suggestion"),
    [("FRC", "'FCR'"), ("predictor.disorde", "'predictor.disorder'"), ("nardini.vectors", "'nardini.vector'")],
)
def test_an_unknown_name_is_refused_with_a_suggestion(name, suggestion):
    with pytest.raises(SparrowException, match=f"is not something a Protein computes. Did you mean {suggestion}"):
        sparrow.batch(SEQUENCES, [name])


@pytest.mark.parametrize(
    "name",
    [
        "sequence",
        "show_sequence",
        "plugin",
        "predictor",
        "_store",
        "predictor._network",
        "polymeric.EMPIRICAL_MODES",
        "nardini.vector.values",
        "elsewhere.FCR",
        # an old name, which works on a Polymeric with a warning, is not offered here
        "polymeric.get_mean_afrc_radius_of_gyration",
    ],
)
def test_something_that_is_not_a_value_is_refused(name):
    with pytest.raises(SparrowException, match="is not something a Protein computes"):
        sparrow.batch(SEQUENCES, [name])


def test_a_property_given_arguments_is_refused():
    with pytest.raises(SparrowException, match="FCR is a property, so it takes no arguments"):
        sparrow.batch(SEQUENCES, {"FCR": {"pH": 7.0}})


def test_recompute_is_refused():
    with pytest.raises(SparrowException, match="cannot take 'recompute'"):
        sparrow.batch(SEQUENCES, {"predictor.disorder": {"recompute": True}})


def test_an_argument_the_method_does_not_take_is_refused():
    with pytest.raises(SparrowException, match="charge_at_pH cannot be called with"):
        sparrow.batch(SEQUENCES, {"charge_at_pH": {"ph": 6.0}})


def test_a_missing_argument_is_refused_with_how_to_give_it():
    with pytest.raises(SparrowException, match=r"missing a required argument.*in a dict"):
        sparrow.batch(SEQUENCES, ["compute_iwd"])


@pytest.mark.parametrize("names", [{"charge_at_pH": 6.0}, [{"charge_at_pH": 6.0}]])
def test_arguments_that_are_not_a_dict_or_none_are_refused(names):
    with pytest.raises(SparrowException, match="must be a dict, such as .*, or None"):
        sparrow.batch(SEQUENCES, names)


def test_an_entry_that_is_neither_a_name_nor_a_dict_is_refused():
    with pytest.raises(SparrowException, match="Each entry of names must be a name"):
        sparrow.batch(SEQUENCES, ["FCR", 7])


def test_a_mixed_list_and_a_dict_ask_for_the_same_values():
    mixed = ["FCR", "kappa", {"charge_at_pH": {"pH": 6.0}}]
    as_dict = {"FCR": None, "kappa": None, "charge_at_pH": {"pH": 6.0}}
    assert sparrow.batching._requests(mixed) == sparrow.batching._requests(as_dict)


def test_the_same_value_written_several_ways_is_asked_for_once():
    # pH 7.4 is charge_at_pH's default, so all three of those are one value
    names = ["FCR", {"FCR": None}, "charge_at_pH", {"charge_at_pH": {}}, {"charge_at_pH": {"pH": 7.4}}]
    assert [request.name for request in sparrow.batching._requests(names)] == ["FCR", "charge_at_pH"]


def test_a_name_can_be_asked_for_with_several_sets_of_arguments(monkeypatch):
    import sparrow.protein

    names = [{"charge_at_pH": {"pH": 6.0}}, {"charge_at_pH": {"pH": 7.0}}]
    proteins = sparrow.batch(SEQUENCES, names)
    expected = [Protein(SEQUENCES["medium"]).charge_at_pH(pH=pH) for pH in (6.0, 7.0)]

    def fail(*args, **kwargs):
        raise AssertionError("computed again")

    monkeypatch.setattr(sparrow.protein.charge, "charge_at_pH", fail)
    assert [proteins["medium"].charge_at_pH(pH=pH) for pH in (6.0, 7.0)] == expected


@pytest.mark.parametrize("names", [[], {}, [{}]])
def test_no_names_is_refused(names):
    with pytest.raises(SparrowException, match="No names"):
        sparrow.batch(SEQUENCES, names)


def test_every_name_is_checked_before_anything_is_computed(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("computed before every name was checked")

    monkeypatch.setattr("sparrow.properties.compute", fail)
    with pytest.raises(SparrowException, match="is not something a Protein computes"):
        sparrow.batch(SEQUENCES, ["FCR", "not_a_value"])


# .................................................................
# the values
#
def test_the_scalar_properties_are_each_proteins_own_to_the_last_bit():
    proteins = sparrow.batch(SEQUENCES, PROPERTY_NAMES)
    for key, sequence in SEQUENCES.items():
        alone = Protein(sequence)
        for name in PROPERTY_NAMES:
            assert getattr(proteins[key], name) == getattr(alone, name), name


def test_the_scalar_properties_are_computed_once_for_the_whole_batch(monkeypatch):
    import sparrow.properties.composition as composition

    calls = []
    real_fraction_charged = composition.fraction_charged

    def counting_fraction_charged(counts):
        calls.append(counts)
        return real_fraction_charged(counts)

    monkeypatch.setattr(composition, "fraction_charged", counting_fraction_charged)
    proteins = sparrow.batch(SEQUENCES, ["FCR"])
    assert [protein.FCR for protein in proteins.values()]
    assert len(calls) == 1


@pytest.mark.parametrize("name", ALBATROSS_NAMES)
def test_the_albatross_predictions_are_each_proteins_own(name):
    proteins = sparrow.batch(SEQUENCES, [name])
    for key, sequence in SEQUENCES.items():
        batched = read_back(proteins[key], name, {})
        alone = read_back(Protein(sequence), name, {})
        assert type(batched) is type(alone)
        assert_same(batched, alone, name)


@pytest.mark.parametrize("arguments", [{"use_scaled": False}, {"use_scaled": False, "safe": False}])
def test_each_sequence_uses_the_network_its_method_would(arguments):
    # a short sequence uses the scaled network unless safe is False, as the method does
    proteins = sparrow.batch(SEQUENCES, {"predictor.radius_of_gyration": arguments})
    for key, sequence in SEQUENCES.items():
        batched = proteins[key].predictor.radius_of_gyration(**arguments)
        alone = Protein(sequence).predictor.radius_of_gyration(**arguments)
        assert np.isclose(batched, alone, rtol=ALBATROSS_RELATIVE_TOLERANCE, atol=0), key


def test_reading_a_batch_prediction_loads_no_network(monkeypatch):
    proteins = sparrow.batch(SEQUENCES, ALBATROSS_NAMES)

    def fail(*args, **kwargs):
        raise AssertionError("a network was loaded to read a prediction already made")

    monkeypatch.setattr("sparrow.predictors.base.load_parrot_network", fail)
    for protein in proteins.values():
        for name in ALBATROSS_NAMES:
            read_back(protein, name, {})


def test_a_prediction_already_made_is_kept():
    protein = Protein(SEQUENCES["long"])
    alone = protein.predictor.radius_of_gyration()
    sparrow.batch([protein], ["predictor.radius_of_gyration"])
    assert protein.predictor.radius_of_gyration() is alone


def test_the_polymeric_predictions_read_the_batch_predictions(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("a protein ran its own network")

    monkeypatch.setattr("sparrow.predictors.base.load_parrot_network", fail)
    names = ["polymeric.predicted_rg", "polymeric.predicted_nu", "polymeric.predicted_nu_dep_end_to_end_distribution"]
    proteins = sparrow.batch(SEQUENCES, names)
    monkeypatch.undo()

    for key, sequence in SEQUENCES.items():
        assert np.isclose(
            proteins[key].polymeric.predicted_rg,
            Protein(sequence).polymeric.predicted_rg,
            rtol=ALBATROSS_RELATIVE_TOLERANCE,
            atol=0,
        )


def test_the_nardini_vectors_are_each_proteins_own_to_the_last_bit():
    proteins = sparrow.batch(SEQUENCES, {"nardini.vector": NARDINI_ARGUMENTS}, n_processes=1)
    for key, sequence in SEQUENCES.items():
        alone = Protein(sequence).nardini.vector(**NARDINI_ARGUMENTS)
        assert np.array_equal(proteins[key].nardini.vector(**NARDINI_ARGUMENTS).values, alone.values)


def test_the_nardini_methods_read_the_batch_scrambles(monkeypatch):
    import sparrow.nardini.batch
    import sparrow.nardini.vectors

    option_sets = []
    real_nardini_proteins = sparrow.nardini.batch.nardini_proteins

    def counting_nardini_proteins(proteins, **options):
        option_sets.append(options)
        return real_nardini_proteins(proteins, **options)

    monkeypatch.setattr(sparrow.nardini.batch, "nardini_proteins", counting_nardini_proteins)
    names = {name: NARDINI_ARGUMENTS for name in ["nardini.vector", "nardini.gin_cluster", "nardini.patterning"]}
    proteins = sparrow.batch(SEQUENCES, names, n_processes=1)

    # GIN uses the default vector, so it shares the vector's scrambles; patterning
    # needs no background, so it has its own
    assert len(option_sets) == 2

    def fail(*args, **kwargs):
        raise AssertionError("a protein scrambled its own sequence")

    monkeypatch.setattr(sparrow.nardini.vectors, "nardini_vectors", fail)
    for protein in proteins.values():
        protein.nardini.vector(**NARDINI_ARGUMENTS)
        protein.nardini.gin_cluster(**NARDINI_ARGUMENTS)
        protein.nardini.patterning(**NARDINI_ARGUMENTS)


def test_a_value_is_free_to_read_with_the_arguments_given_and_computed_with_others(monkeypatch):
    import sparrow.protein

    proteins = sparrow.batch(SEQUENCES, {"charge_at_pH": {"pH": 6.0}})
    expected = Protein(SEQUENCES["medium"]).charge_at_pH(pH=6.0)

    def fail(*args, **kwargs):
        raise AssertionError("computed again")

    monkeypatch.setattr(sparrow.protein.charge, "charge_at_pH", fail)
    assert proteins["medium"].charge_at_pH(pH=6.0) == expected
    with pytest.raises(AssertionError, match="computed again"):
        proteins["medium"].charge_at_pH(pH=7.0)


@pytest.mark.parametrize("name", _available_names())
def test_every_name_batch_accepts_gives_what_a_protein_computes(name):
    arguments = ARGUMENTS.get(name, {})
    proteins = sparrow.batch(SEQUENCES, {name: arguments}, n_processes=1)
    for key, sequence in SEQUENCES.items():
        assert_same(read_back(proteins[key], name, arguments), read_back(Protein(sequence), name, arguments), name)


def test_the_arguments_table_names_only_names_batch_accepts():
    assert set(ARGUMENTS) <= set(_available_names())
