"""Unit and regression tests for ``sparrow.protein``."""

import sys

import numpy as np
import pytest

import sparrow
from sparrow.protein import Protein
from sparrow.sequence_analysis import patching
from sparrow.sequence_analysis.elm import (
    compute_gained_elms,
    compute_lost_elms,
    compute_retained_elms,
)

CANONICAL_SEQUENCE = "MKASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQ"
BROKEN_SEQUENCE = "MKASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSXYGQSSYSSYXQ"
PATCH_SEQUENCE = "AAAAQQRGRGTTTAAAQQ"
ELM_RICH_SEQUENCE = "KRRARKRRARKRRARKRRAR"


@pytest.fixture
def canonical_protein():
    return Protein(BROKEN_SEQUENCE, validate=True)


@pytest.fixture
def patch_protein():
    return Protein(PATCH_SEQUENCE)


@pytest.fixture
def elm_rich_protein():
    return Protein(ELM_RICH_SEQUENCE)


@pytest.fixture
def wt_protein():
    return sparrow.Protein("MKKK")


@pytest.fixture
def mut_protein():
    return sparrow.Protein("MRKK")


def test_sparrow_imported():
    assert "sparrow" in sys.modules


def test_compute_patch_fraction_wrapper_matches_functional_api(patch_protein):
    assert np.isclose(
        patch_protein.compute_patch_fraction("A"),
        patching.patch_fraction(PATCH_SEQUENCE, "A"),
    )


def test_compute_rg_patch_fraction_wrapper_matches_functional_api(patch_protein):
    assert np.isclose(
        patch_protein.compute_rg_patch_fraction(),
        patching.rg_patch_fraction(PATCH_SEQUENCE),
    )


def test_compute_rg_patch_fraction_matches_generic_patch_call(patch_protein):
    assert np.isclose(
        patch_protein.compute_patch_fraction(
            residue_selector="RG",
            min_target_count=None,
            adjacent_pair_pattern="RG",
            min_adjacent_pair_count=2,
        ),
        patch_protein.compute_rg_patch_fraction(),
    )


def test_constructor_length_default():
    assert len(Protein(CANONICAL_SEQUENCE)) == 61


def test_constructor_length_with_validation():
    assert len(Protein(CANONICAL_SEQUENCE, validate=True)) == 61


def test_constructor_validation_normalizes_broken_sequence(canonical_protein):
    assert len(canonical_protein) == 61
    assert canonical_protein.sequence == CANONICAL_SEQUENCE


def test_amino_acid_fraction_dict_has_20_entries(canonical_protein):
    assert len(canonical_protein.amino_acid_fractions) == 20


@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("FCR", 0.04918032786885246),
        ("fraction_positive", 0.01639344262295082),
        ("fraction_negative", 0.03278688524590164),
        ("NCPR", -0.01639344262295082),
        ("fraction_aromatic", 0.16393442622950818),
        ("fraction_aliphatic", 0.06557377049180328),
        ("fraction_polar", 0.6721311475409836),
        ("fraction_proline", 0.04918032786885246),
    ],
)
def test_basic_fractional_attributes(canonical_protein, attribute, expected):
    assert getattr(canonical_protein, attribute) == pytest.approx(expected)


def test_predictor_disorder_mean(canonical_protein):
    pytest.importorskip("metapredict")
    assert np.isclose(np.mean(canonical_protein.predictor.disorder()), 0.92875415)


def test_hydrophobicity(canonical_protein):
    assert canonical_protein.hydrophobicity == pytest.approx(3.052459016393442)


def test_compute_residue_fractions_for_custom_group(canonical_protein):
    assert canonical_protein.compute_residue_fractions(["P", "E", "K", "R", "D"]) == (
        pytest.approx(0.09836065573770492)
    )


def test_patch_fraction_matches_functional_api_on_canonical_sequence(canonical_protein):
    assert np.isclose(
        canonical_protein.compute_patch_fraction("Q"),
        patching.patch_fraction(canonical_protein.sequence, "Q"),
    )


def test_rg_patch_fraction_matches_functional_api_on_canonical_sequence(
    canonical_protein,
):
    assert np.isclose(
        canonical_protein.compute_rg_patch_fraction(),
        patching.rg_patch_fraction(canonical_protein.sequence),
    )


@pytest.mark.parametrize(
    ("profile_name", "expected"),
    [
        ("FCR", 0.04918032786885246),
        ("NCPR", -0.02459016393442623),
        ("aromatic", 0.1680327868852459),
        ("aliphatic", 0.05737704918032787),
        ("polar", 0.6762295081967213),
        ("proline", 0.04918032786885246),
        ("positive", 0.012295081967213115),
        ("negative", 0.036885245901639344),
        ("hydrophobicity", 3.0450819672131146),
    ],
)
def test_linear_sequence_profile_means(canonical_protein, profile_name, expected):
    observed = np.mean(canonical_protein.linear_sequence_profile(profile_name))
    assert observed == pytest.approx(expected)


def test_linear_composition_profile_mean(canonical_protein):
    observed = np.mean(canonical_protein.linear_composition_profile(["E", "K"]))
    assert observed == pytest.approx(0.012295081967213115)


def test_elm_functional_sites_match_expected_names(elm_rich_protein):
    observed = {elm.functional_site_name for elm in elm_rich_protein.elms}
    assert observed.issubset(
        {
            "di Arginine retention/retrieving signal",
            "CendR Motif Binding to Neuropilin Receptors",
            "FEM1ABC C-terminal Arg degrons",
            "NLS classical Nuclear Localization Signals",
            "N-degron",
            "NRD cleavage site",
            "PCSK cleavage site",
        }
    )


def test_elm_start_positions_match_expected(elm_rich_protein):
    starts = [elm.start for elm in elm_rich_protein.elms]
    assert sorted(starts) == [
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        3,
        4,
        4,
        5,
        5,
        6,
        6,
        9,
        9,
        10,
        10,
        11,
        11,
        11,
        13,
        14,
        14,
        15,
        15,
        16,
        16,
        16,
        16,
    ]


def test_elm_end_positions_match_expected(elm_rich_protein):
    ends = [elm.end for elm in elm_rich_protein.elms]
    assert sorted(ends) == [
        3,
        3,
        3,
        4,
        5,
        8,
        8,
        8,
        9,
        9,
        9,
        9,
        10,
        10,
        13,
        13,
        14,
        14,
        15,
        15,
        18,
        18,
        18,
        19,
        19,
        19,
        20,
        20,
        20,
        20,
        20,
    ]


def test_elm_sequences_match_expected(elm_rich_protein):
    sequences = [elm.sequence for elm in elm_rich_protein.elms]
    assert sorted(sequences) == sorted(
        [
            "RRA",
            "RRA",
            "RRA",
            "RRA",
            "RKRRA",
            "RKRRA",
            "RKRRA",
            "KRR",
            "KRR",
            "KRR",
            "KRR",
            "KRR",
            "KRR",
            "KRR",
            "KRR",
            "RRARKRR",
            "RRARKRR",
            "KRR",
            "RRAR",
            "RRAR",
            "RRAR",
            "RRAR",
            "RRAR",
            "RRAR",
            "KRRARKRRARKRRARKRRAR",
            "ARKRRA",
            "ARKRRA",
            "RKRRAR",
            "RKRRAR",
            "RKRRAR",
            "RRARKRRA",
        ]
    )


def _elm_signature_set(elms):
    return {
        (
            e.regex,
            e.identifier,
            e.functional_site_name,
            e.start,
            e.end,
            e.sequence,
        )
        for e in elms
    }


def test_wt_elm_set(wt_protein):
    assert _elm_signature_set(wt_protein.elms) == {
        (
            "^M{0,1}[RK][^P].",
            "DEG_Nend_UBRbox_1",
            "N-degron",
            0,
            4,
            "MKKK",
        )
    }


def test_mut_elm_set(mut_protein):
    assert _elm_signature_set(mut_protein.elms) == {
        (
            "(.RK)|(RR[^KR])",
            "CLV_NRD_NRD_1",
            "NRD cleavage site",
            0,
            3,
            "MRK",
        ),
        (
            "^M{0,1}[RK][^P].",
            "DEG_Nend_UBRbox_1",
            "N-degron",
            0,
            4,
            "MRKK",
        ),
    }


def test_elm_set_difference_is_empty(wt_protein, mut_protein):
    assert wt_protein.elms - mut_protein.elms == set()


def test_elm_set_intersection(wt_protein, mut_protein):
    assert _elm_signature_set(wt_protein.elms & mut_protein.elms) == {
        (
            "^M{0,1}[RK][^P].",
            "DEG_Nend_UBRbox_1",
            "N-degron",
            0,
            4,
            "MKKK",
        )
    }


def test_compute_lost_elms_for_wt(wt_protein):
    assert compute_lost_elms(wt_protein, [2, "K"]) == set()


def test_compute_retained_elms_for_wt(wt_protein):
    assert _elm_signature_set(compute_retained_elms(wt_protein, "p.K1R")) == {
        (
            "^M{0,1}[RK][^P].",
            "DEG_Nend_UBRbox_1",
            "N-degron",
            0,
            3,
            "RKK",
        )
    }


def test_compute_gained_elms_for_wt(wt_protein):
    assert _elm_signature_set(compute_gained_elms(wt_protein, "p.K2R")) == {
        (
            "(.RK)|(RR[^KR])",
            "CLV_NRD_NRD_1",
            "NRD cleavage site",
            0,
            3,
            "MRK",
        )
    }


def test_compute_retained_elms_for_mut(mut_protein):
    assert _elm_signature_set(compute_retained_elms(mut_protein, "p.M1K")) == {
        (
            "(.RK)|(RR[^KR])",
            "CLV_NRD_NRD_1",
            "NRD cleavage site",
            0,
            3,
            "MRK",
        ),
        (
            "^M{0,1}[RK][^P].",
            "DEG_Nend_UBRbox_1",
            "N-degron",
            0,
            4,
            "MRKK",
        ),
    }


def test_compute_gained_elms_for_mut(mut_protein):
    assert _elm_signature_set(compute_gained_elms(mut_protein, "p.M1K")) == {
        (
            "KR.",
            "CLV_PCSK_PC1ET2_1",
            "PCSK cleavage site",
            0,
            3,
            "KRK",
        ),
        (
            "[KR]R.",
            "CLV_PCSK_KEX2_1",
            "PCSK cleavage site",
            0,
            3,
            "KRK",
        ),
    }


def test_compute_lost_elms_for_mut(mut_protein):
    assert _elm_signature_set(compute_lost_elms(mut_protein, "p.M1G")) == {
        (
            "^M{0,1}[RK][^P].",
            "DEG_Nend_UBRbox_1",
            "N-degron",
            0,
            4,
            "MRKK",
        )
    }
