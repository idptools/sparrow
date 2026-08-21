"""Regression tests for previously-fixed bugs.

Each test here pins behaviour that was once wrong. Grouped by the module the bug
lived in.
"""

import math

import numpy as np
import pytest

from sparrow import Protein
from sparrow.data.amino_acids import VALID_AMINO_ACIDS
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


@pytest.mark.parametrize("exception_cls", [ProteinException, PatterningException, CalculationException])
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
    assert p.polymeric.get_mean_saw_radius_of_gyration(prefactor=5.5) != pytest.approx(
        p.polymeric.get_mean_saw_radius_of_gyration(prefactor=8.0)
    )

    p2 = Protein(TEST_SEQ)
    assert p2.polymeric.get_mean_saw_end_to_end_distance(prefactor=5.5) != pytest.approx(
        p2.polymeric.get_mean_saw_end_to_end_distance(prefactor=8.0)
    )


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


@pytest.mark.parametrize("method_name", ["empirical_nu", "empirical_radius_of_gyration"])
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
    assert p.compute_kappa_x("ED", "KR") == pytest.approx(p.compute_kappa_x("DE", "RK"))


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
# sparrow.sequence_analysis.grammar


def test_grammar_module_is_deprecated_in_favour_of_nardini():
    """grammar.py is superseded by sparrow.nardini but must still work."""
    from sparrow.sequence_analysis import grammar

    with pytest.warns(DeprecationWarning, match="sparrow.nardini"):
        vector = grammar.compute_feature_vector(TEST_SEQ, num_scrambles=10, seed=1)
    assert vector.shape == (86,)

    with pytest.warns(DeprecationWarning, match="sparrow.nardini"):
        grammar.compute_composition_background_stats([TEST_SEQ, "QQQQNNNNSSSS"])


def test_extract_feature_vector_still_works_while_deprecated():
    p = Protein(TEST_SEQ)
    with pytest.warns(DeprecationWarning):
        assert p.extract_feature_vector(num_scrambles=10, seed=1).shape == (86,)


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


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_zscore_feature_vector_length():
    # the deprecation itself is asserted separately, above
    from sparrow.sequence_analysis import grammar

    vec = grammar.compute_feature_vector(TEST_SEQ, num_scrambles=10, seed=1)
    assert vec.shape == (86,)


# .................................................................
# sparrow.predictors -- binary_disorder must key its cache on its parameters


def test_binary_disorder_respects_changed_threshold():
    p = Protein(
        "MKYLAAYLLLNAAGNTPDATKIKAILESVGIEIEDEKVSSVLSALEGKSVDELITEGNEKLAAVPAAGPASAGGAAAASGDAAAEEEKEEEAAEESDDDMGFGLFD"
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
    assert isinstance(p.linear_property_profile("hydropathy-kyte-1982"), np.ndarray)


# .................................................................
# physical properties shared between Protein and sparrow.nardini


#: Allowed drift between the Protein property and the NARDINI+ feature, in units
#: of the last bit of the expected value. Measured across this test's corpus,
#: 12 of the 13 pairs agree bit-for-bit and "Frac Aromatic" differs by exactly
#: 1 ULP; 2 leaves headroom without admitting anything meaningful.
MAX_ULP_DRIFT = 2

PHYSICAL_PROPERTY_PAIRS = {
    "Frac Chain Expanding": "fraction_chain_expanding",
    "Disorder Promoting": "fraction_disorder_promoting",
    "Iso point": "isoelectric_point",
    "PPII": "PPII_propensity",
    "FCR": "FCR",
    "NCPR": "NCPR",
    "Hydrophobicity": "hydrophobicity",
    "Frac K+R": "fraction_positive",
    "Frac D+E": "fraction_negative",
    "Frac Aromatic": "fraction_aromatic",
    "Frac Aliphatic": "fraction_aliphatic",
    "R/K Ratio": "RK_ratio",
    "E/D Ratio": "ED_ratio",
}


@pytest.mark.parametrize("feature,attribute", sorted(PHYSICAL_PROPERTY_PAIRS.items()))
def test_protein_property_matches_nardini_feature(feature, attribute):
    """These are general sequence properties, computed in exactly one place.

    They used to be implemented inside sparrow.nardini.composition. They now live
    in sequence_analysis.physical_properties, are exposed on Protein, and NARDINI
    reads them from there. The two must not be allowed to drift: the bundled
    per-species backgrounds and GIN centroids are defined against these values, so
    a real change would shift every compositional z-score and cluster distance.

    Agreement is asserted to within MAX_ULP_DRIFT units in the last place rather
    than bit-for-bit. ``physical_properties.fraction_in_set`` deliberately sums
    per-residue fractions (``a/n + b/n``, not ``(a + b)/n``) to preserve the exact
    floats the bundled data was built against, and floating-point addition is not
    associative -- so the *order* of a residue set changes its last bit.
    ``amino_acids.ARO`` is ``['Y', 'W', 'F']`` where NARDINI+ uses ``tuple("FWY")``,
    which puts "Frac Aromatic" 1 ULP apart. That is a summation-order artefact, not
    a drift in definition; a redefined residue set moves the value far more than a
    couple of ULP and still fails here.
    """
    import random

    from sparrow.nardini import composition_values

    rng = random.Random(99)
    sequences = ["".join(rng.choice(VALID_AMINO_ACIDS) for _ in range(rng.randint(12, 200))) for _ in range(25)]
    sequences += ["Q" * 40, "E" * 40, "K" * 40, "GRGRGGYGG" * 5, "MEEEKKKSSSTTT"]

    for sequence in sequences:
        expected = composition_values(sequence)[feature]
        actual = getattr(Protein(sequence), attribute)
        # scale by the expected value's own last bit; fall back to ulp(1.0) so an
        # expected 0.0 gets a sane absolute floor rather than demanding exactness
        tolerance = MAX_ULP_DRIFT * math.ulp(expected if expected else 1.0)
        assert abs(actual - expected) <= tolerance, (
            f"{feature} / {attribute} on {sequence[:30]}: {actual!r} vs {expected!r} "
            f"(off by {abs(actual - expected) / math.ulp(expected if expected else 1.0):.2f} ULP)"
        )


def test_no_duplicate_implementation_remains_in_nardini():
    """nardini must delegate, not carry its own copy.

    Guards against the obvious regression: someone re-inlines one of these into
    the composition module and the two drift apart.
    """
    import inspect

    from sparrow.nardini import composition
    from sparrow.sequence_analysis import physical_properties

    assert composition.charge_at_pH is physical_properties.charge_at_pH
    assert composition.isoelectric_point is physical_properties.isoelectric_point

    source = inspect.getsource(composition)
    for helper in (
        "fraction_chain_expanding",
        "fraction_disorder_promoting",
        "polyproline_II_propensity",
    ):
        assert f"physical_properties.{helper}" in source, (
            f"composition_values should call physical_properties.{helper}"
        )


def test_charge_at_ph_and_isoelectric_point_are_consistent():
    """The pI is where the titratable charge crosses zero, by construction."""
    for sequence in ["MEEEKKKSSSTTT", "EEEEDDDD", "KKKKRRRR", "GSGSGSGS"]:
        protein = Protein(sequence)
        assert abs(protein.charge_at_pH(protein.isoelectric_point)) <= 0.02


def test_charge_at_ph_is_monotonic_in_ph():
    """More acidic conditions cannot make a sequence more negative."""
    protein = Protein("MEEEKKKSSSTTTDDDRRRHHHYYYCCC")
    charges = [protein.charge_at_pH(pH) for pH in (2.0, 5.0, 7.4, 9.0, 12.0)]

    assert charges == sorted(charges, reverse=True)


def test_charge_at_ph_normalize_flag():
    protein = Protein("EEEEKKKK")

    summed = protein.charge_at_pH(normalize=False)
    mean = protein.charge_at_pH(normalize=True)

    assert mean == pytest.approx(summed / 8)


def test_physical_properties_handle_a_sequence_with_no_titratable_residues():
    protein = Protein("GSGSGSGSAAA")

    assert protein.charge_at_pH() == 0.0
    assert protein.isoelectric_point == 7.0


def test_new_physical_properties_are_cached():
    protein = Protein(TEST_SEQ)

    for attribute in (
        "fraction_chain_expanding",
        "fraction_disorder_promoting",
        "isoelectric_point",
        "PPII_propensity",
    ):
        first = getattr(protein, attribute)
        assert getattr(protein, attribute) is first or getattr(protein, attribute) == first


def test_amino_acid_fractions_match_nardini():
    """The 20 single-residue fractions come from one implementation."""
    from sparrow.nardini import composition_values

    for sequence in [TEST_SEQ, "QQQQNNNN" * 4, "GRGRGGYGG" * 4, "MEEEKKKCCC"]:
        features = composition_values(sequence)
        fractions = Protein(sequence).amino_acid_fractions
        for residue in VALID_AMINO_ACIDS:
            assert fractions[residue] == features[f"Frac {residue}"], f"{residue} in {sequence[:20]}"


def test_patch_features_match_nardini():
    """Patch fractions come from sequence_analysis.patching, used by both."""
    from sparrow.nardini.composition import (
        PATCH_INTERRUPTION,
        PATCH_MIN_TARGET_COUNT,
        PATCH_RESIDUES,
        RG_MIN_ADJACENT_PAIRS,
        composition_values,
    )

    for sequence in ["QQQQQQAAAQQQQQQ" * 3, "GRGRGGYGGRGGYGG" * 3, TEST_SEQ]:
        features = composition_values(sequence)
        protein = Protein(sequence)

        for residue in PATCH_RESIDUES:
            assert (
                protein.compute_patch_fraction(
                    residue_selector=residue,
                    interruption=PATCH_INTERRUPTION,
                    min_target_count=PATCH_MIN_TARGET_COUNT,
                )
                == features[f"{residue} Patch"]
            )

        assert (
            protein.compute_rg_patch_fraction(
                interruption=PATCH_INTERRUPTION,
                min_adjacent_rg_pairs=RG_MIN_ADJACENT_PAIRS,
            )
            == features["RG Frac"]
        )


def test_polar_definitions_differ_only_by_cysteine():
    """The one deliberate disagreement, pinned so it stays deliberate.

    NARDINI+ counts cysteine as polar ('Frac Polar' = QNSTGCH); sparrow's own
    POLAR set does not (GHNQST). Both conventions are used in the literature, and
    the bundled backgrounds and GIN centroids are defined against the NARDINI one,
    so neither can be quietly changed to match the other. The gap must be exactly
    the cysteine content - anything else means one of them has drifted.
    """
    import random

    from sparrow.nardini import composition_values

    rng = random.Random(17)
    sequences = ["".join(rng.choice(VALID_AMINO_ACIDS) for _ in range(rng.randint(30, 200))) for _ in range(20)]
    sequences += ["CCCCSSSSTTTT", "SSSSTTTTQQQQ"]

    saw_cysteine = False
    for sequence in sequences:
        features = composition_values(sequence)
        protein = Protein(sequence)
        gap = features["Frac Polar"] - protein.fraction_polar

        assert gap == pytest.approx(features["Frac C"], abs=1e-12), sequence[:30]
        saw_cysteine = saw_cysteine or features["Frac C"] > 0

    assert saw_cysteine, "this test proves nothing without a cysteine-containing case"


def test_nardini_composition_has_no_arithmetic_of_its_own():
    """Every compositional feature must be delegated, not computed in place.

    The failure this guards against is someone re-inlining a formula for speed or
    convenience, after which the NARDINI feature and its Protein property drift
    apart silently - and the bundled reference data quietly stops matching.
    """
    import inspect

    from sparrow.nardini import composition

    source = inspect.getsource(composition.composition_values)

    for helper in (
        "calculate_aa_fractions",
        "fraction_in_set",
        "log_residue_ratio",
        "fraction_chain_expanding",
        "fraction_charged",
        "net_charge_per_residue",
        "mean_residue_scale",
        "fraction_disorder_promoting",
        "isoelectric_point",
        "polyproline_II_propensity",
    ):
        assert helper in source, f"composition_values should delegate to {helper}"

    # no residue-set arithmetic left inline
    assert "sum(fractions[" not in source
    assert "np.log10" not in source
