"""Regression tests for previously-fixed bugs.

Each test here pins behaviour that was once wrong. Grouped by the module the bug
lived in.
"""

import numpy as np
import pytest

from sparrow import Protein
from sparrow.sequence_analysis import phospho_isoforms
from sparrow.sequence_analysis.physical_properties import calculate_molecular_weight
from sparrow.sparrow_exceptions import (
    CalculationException,
    PatterningException,
    ProteinException,
    SparrowException,
)
from sparrow.tools.utilities import validate_keyword_option

# a sequence long enough for the polymer models and with mixed charge/composition
TEST_SEQ = "MEEEKKKKSSSTTTDDDQQQQNNNNGGGSSSEEEKKKWWW"


# .................................................................
# sparrow.tools.utilities


def test_validate_keyword_option_raises_with_custom_message():
    """A custom error_message must not disable the validation itself."""
    with pytest.raises(SparrowException) as exc:
        validate_keyword_option("bogus", ["a", "b"], "mode", error_message="custom msg")

    assert "custom msg" in str(exc.value)


def test_validate_keyword_option_default_message():
    with pytest.raises(SparrowException):
        validate_keyword_option("bogus", ["a", "b"], "mode")

    # valid values must pass silently
    assert validate_keyword_option("a", ["a", "b"], "mode") is None


# .................................................................
# sparrow.sparrow_exceptions


@pytest.mark.parametrize(
    "exception_cls", [ProteinException, PatterningException, CalculationException]
)
def test_all_exceptions_derive_from_sparrow_exception(exception_cls):
    """`except SparrowException` must catch everything sparrow raises."""
    assert issubclass(exception_cls, SparrowException)


# .................................................................
# sparrow.sequence_analysis.physical_properties


def test_molecular_weight_empty_sequence_is_zero():
    """An empty sequence has no residues, so it must not pick up a water."""
    assert calculate_molecular_weight("") == 0.0
    assert Protein("").molecular_weight == 0.0


def test_molecular_weight_single_residue_and_peptide():
    # one residue = free amino acid weight, no peptide-bond correction
    assert calculate_molecular_weight("A") == pytest.approx(89.1)
    # two residues = one peptide bond, so one water removed
    assert calculate_molecular_weight("AA") == pytest.approx(89.1 * 2 - 18)


# .................................................................
# sparrow.polymer -- memoization must key on the arguments


def test_nudep_saw_rg_varies_with_nu():
    """Cached polymer values must not leak across different nu values."""
    p = Protein(TEST_SEQ)
    low = p.polymeric.get_mean_nudep_saw_radius_of_gyration(nu=0.40)
    high = p.polymeric.get_mean_nudep_saw_radius_of_gyration(nu=0.60)

    assert low != pytest.approx(high)

    # and the first value is still retrievable (i.e. the cache is per-nu)
    assert p.polymeric.get_mean_nudep_saw_radius_of_gyration(nu=0.40) == pytest.approx(low)

    # cross-check against a protein that only ever saw nu=0.6
    fresh = Protein(TEST_SEQ)
    assert fresh.polymeric.get_mean_nudep_saw_radius_of_gyration(nu=0.60) == pytest.approx(high)


def test_nudep_saw_re_varies_with_nu_and_prefactor():
    p = Protein(TEST_SEQ)
    a = p.polymeric.get_mean_nudep_saw_end_to_end_distance(nu=0.45, prefactor=5.5)
    b = p.polymeric.get_mean_nudep_saw_end_to_end_distance(nu=0.55, prefactor=5.5)
    c = p.polymeric.get_mean_nudep_saw_end_to_end_distance(nu=0.45, prefactor=8.0)

    assert a != pytest.approx(b)
    assert a != pytest.approx(c)


def test_saw_values_vary_with_prefactor():
    p = Protein(TEST_SEQ)
    assert p.polymeric.get_mean_saw_radius_of_gyration(
        prefactor=5.5
    ) != pytest.approx(p.polymeric.get_mean_saw_radius_of_gyration(prefactor=8.0))

    p2 = Protein(TEST_SEQ)
    assert p2.polymeric.get_mean_saw_end_to_end_distance(
        prefactor=5.5
    ) != pytest.approx(p2.polymeric.get_mean_saw_end_to_end_distance(prefactor=8.0))


def test_afrc_mean_rg_varies_with_mode():
    p = Protein(TEST_SEQ)
    dist = p.polymeric.get_mean_afrc_radius_of_gyration(mode="distribution")
    law = p.polymeric.get_mean_afrc_radius_of_gyration(mode="scaling law")

    assert dist != pytest.approx(law)


def test_afrc_mean_re_varies_with_mode():
    p = Protein(TEST_SEQ)
    law = p.polymeric.get_mean_afrc_end_to_end_distance(mode="scaling law")
    dist = p.polymeric.get_mean_afrc_end_to_end_distance(mode="distribution")

    assert law != pytest.approx(dist)


# .................................................................
# sparrow.polymer -- the empirical mode selector must not be ignored


@pytest.mark.parametrize(
    "method_name", ["empirical_nu", "empirical_radius_of_gyration"]
)
def test_empirical_methods_reject_unknown_mode(method_name):
    p = Protein(TEST_SEQ)
    method = getattr(p.polymeric, method_name)

    with pytest.raises(SparrowException):
        method(mode="not-a-real-parameterization")

    # the supported mode still works, both explicitly and by default
    assert method() == pytest.approx(method(mode="zheng2020"))


# .................................................................
# sparrow.visualize.sequence_visuals


def test_show_sequence_newline_minus_one_is_one_line():
    """newline=-1 is documented as 'use the sequence length', i.e. no breaks."""
    p = Protein("A" * 120)
    assert p.show_sequence(newline=-1, return_raw_string=True).count("<br>") == 0


def test_show_sequence_blocksize_minus_one_has_no_spacing():
    p = Protein("A" * 120)
    html = p.show_sequence(blocksize=-1, return_raw_string=True)
    assert html.count("<br>") == 0
    # no block separators between the residue spans
    assert "</span> <span" not in html


def test_show_sequence_default_still_breaks_lines():
    p = Protein("A" * 120)
    html = p.show_sequence(return_raw_string=True)
    # 120 residues at newline=50 -> breaks after residues 50 and 100
    assert html.count("<br>") == 2


def test_show_sequence_defaults_are_not_shared_between_calls():
    """Mutable default arguments must not accumulate state across calls."""
    p = Protein("ACDEFGHIKL")

    first = p.show_sequence(return_raw_string=True)
    # a call that supplies its own lists must not contaminate the defaults
    p.show_sequence(bold_positions=[1, 2, 3], bold_residues=["A"], return_raw_string=True)
    second = p.show_sequence(return_raw_string=True)

    assert first == second


# .................................................................
# sparrow.sequence_analysis.phospho_isoforms


def test_phospho_rate_out_of_range_raises():
    p = Protein("MGSQSSRSSSTTTYYY")

    with pytest.raises(SparrowException):
        p.generate_phosphoisoforms(phospho_rate=1.5)

    with pytest.raises(SparrowException):
        p.generate_phosphoisoforms(phospho_rate=-0.1)


def test_phospho_isoform_bad_mode_raises_sparrow_exception():
    p = Protein("MGSQSSRSSSTTTYYY")

    with pytest.raises(SparrowException):
        p.generate_phosphoisoforms(mode="nonsense")

    # custom mode without sites is also a sparrow error
    with pytest.raises(SparrowException):
        p.generate_phosphoisoforms(mode="custom")


def test_phospho_isoform_out_of_range_site_raises():
    with pytest.raises(SparrowException):
        phospho_isoforms._build_phosphoSeqome("MGSQ", [99])


def test_phospho_isoforms_all_mode_count():
    # "MGSTYA" holds 3 candidate sites (S, T, Y) -> 2^3 isoforms at phospho_rate=1
    p = Protein("MGSTYA")
    isoforms = p.generate_phosphoisoforms(mode="all")
    assert len(isoforms) == 2**3
    # the unphosphorylated sequence is one of the isoforms
    assert p.sequence in isoforms


# .................................................................
# sparrow.protein


def test_compute_kappa_x_rejects_empty_group1():
    p = Protein(TEST_SEQ)
    with pytest.raises(ProteinException):
        p.compute_kappa_x("")


def test_compute_kappa_x_group_order_is_irrelevant():
    """Group ordering must not produce distinct cache entries / values."""
    p = Protein(TEST_SEQ)
    assert p.compute_kappa_x("ED", "KR") == pytest.approx(
        p.compute_kappa_x("DE", "RK")
    )


def test_elms_returns_a_set_of_floats():
    p = Protein("MGSQSSRSSSQQQTPPKKRKSAEDLLSPQ")
    elms = p.elms

    assert isinstance(elms, set)
    # probability is annotated as float, so it must actually be one
    assert all(isinstance(e.probability, float) for e in elms)


def test_elm_equality_with_unrelated_type():
    p = Protein("MGSQSSRSSSQQQTPPKKRKSAEDLLSPQ")
    elm = next(iter(p.elms))

    # must not raise AttributeError
    assert (elm == 5) is False
    assert (elm != 5) is True


# .................................................................
# sparrow.sequence_analysis.grammar -- the PATCH_FEATURES schema


def test_patch_features_exclude_tryptophan_only():
    """W Patch is deliberately excluded to match NARDINI+; every other residue is present."""
    from sparrow.data.amino_acids import VALID_AMINO_ACIDS
    from sparrow.sequence_analysis import grammar

    for residue in VALID_AMINO_ACIDS:
        if residue == "W":
            continue
        assert f"{residue} Patch" in grammar.PATCH_FEATURES

    assert "W Patch" not in grammar.PATCH_FEATURES
    assert "RG Frac" in grammar.PATCH_FEATURES
    assert len(grammar.PATCH_FEATURES) == len(VALID_AMINO_ACIDS)


def test_patch_features_match_bundled_background():
    """The feature schema must line up with the shipped composition background."""
    from sparrow.sequence_analysis import grammar

    stats = grammar.load_default_composition_stats()
    expected = list(grammar.COMPOSITION_FEATURES) + list(grammar.PATCH_FEATURES)
    assert list(stats.feature_names) == expected


def test_zscore_feature_vector_length():
    from sparrow.sequence_analysis import grammar

    vec = grammar.compute_feature_vector(TEST_SEQ, num_scrambles=10, seed=1)
    assert vec.shape == (86,)


# .................................................................
# sparrow.predictors -- binary_disorder must key its cache on its parameters


def test_binary_disorder_respects_changed_threshold():
    p = Protein(
        "MKYLAAYLLLNAAGNTPDATKIKAILESVGIEIEDEKVSSVLSALEGKSVDELITEGNEKLAA"
        "VPAAGPASAGGAAAASGDAAAEEEKEEEAAEESDDDMGFGLFD"
    )

    permissive = p.predictor.binary_disorder(disorder_threshold=0.1)
    strict = p.predictor.binary_disorder(disorder_threshold=0.95)

    # both are per-residue masks of the right length
    assert len(permissive) == len(p)
    assert len(strict) == len(p)

    # a much stricter threshold cannot call MORE residues disordered
    assert sum(strict) <= sum(permissive)

    # the first result is still retrievable unchanged
    assert p.predictor.binary_disorder(disorder_threshold=0.1) == permissive


def test_binary_disorder_deprecated_option_warns():
    p = Protein("MEEEKKKKSSSTTTDDDQQQQNNNNGGGSSSEEEKKK" * 2)

    with pytest.warns(DeprecationWarning):
        p.predictor.binary_disorder(override_folded_domain_minsize=True)


# .................................................................
# linear profile return types


def test_linear_profiles_return_arrays():
    p = Protein(TEST_SEQ)

    assert isinstance(p.linear_sequence_profile("NCPR"), np.ndarray)
    assert isinstance(p.linear_composition_profile(["Q", "N"]), np.ndarray)
    assert isinstance(
        p.linear_property_profile("hydropathy-kyte-1982"), np.ndarray
    )
