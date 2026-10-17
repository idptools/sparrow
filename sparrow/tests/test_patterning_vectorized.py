"""Tests for the IWD, SCD and SHD routines in sparrow.patterning.vectorized.

These pin the NumPy implementations against **golden reference values captured
from the Cython extensions before they were removed** (see
``test_data/cython_patterning_golden_v1.npz``, 400 randomly-generated cases).
They were computed in ``double`` on both sides, so agreement is ~1e-13 or better.
The kappa-family tests now live in ``test_properties_patterning.py``.
"""

import random

import numpy as np
import pytest

from sparrow.patterning import vectorized as vec
from sparrow.tools.track_tools import predefined_linear_track

from . import reference_data_path

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

DOUBLE_TOLERANCE = 1e-12


def _random_sequence(rng, length):
    return "".join(rng.choice(AMINO_ACIDS) for _ in range(length))


def _mask_from(sequence, residues):
    lookup = np.zeros(256, dtype=bool)
    for residue in residues:
        lookup[ord(residue)] = True
    return lookup[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]


# =====================================================================
# golden reference values captured from the Cython extensions
# =====================================================================


@pytest.fixture(scope="module")
def golden():
    """400 cases scored by the Cython extensions before they were removed."""
    data = np.load(reference_data_path("cython_patterning_golden_v1.npz"), allow_pickle=False)
    return {key: data[key] for key in data.files}


def _case(golden, index):
    """Unpack one golden case into its inputs."""
    return dict(
        sequence=str(golden["sequence"][index]),
        group1=list(str(golden["group1"][index])),
        group2=list(str(golden["group2"][index])),
        targets=list(str(golden["targets"][index])),
        scd_group1=list(str(golden["scd_group1"][index])),
        scd_group2=list(str(golden["scd_group2"][index])),
        window_size=int(golden["window_size"][index]),
    )


def test_iwd_matches_cython_golden(golden):
    for index in range(len(golden["sequence"])):
        case = _case(golden, index)
        got = vec.calculate_average_inverse_distance_from_sequence(case["sequence"], case["targets"])
        assert got == pytest.approx(float(golden["iwd"][index]), rel=DOUBLE_TOLERANCE, abs=1e-14), f"case {index}"


def test_iwd_charge_matches_cython_golden(golden):
    for index in range(len(golden["sequence"])):
        case = _case(golden, index)
        ncpr = np.asarray(
            predefined_linear_track(case["sequence"], "NCPR", 8, "extend-ends", None),
            dtype=np.float64,
        )
        for charge, key in (("-", "iwd_charge_neg"), ("+", "iwd_charge_pos")):
            got = vec.calculate_average_inverse_distance_charge(ncpr, case["sequence"], charge)
            assert got == pytest.approx(float(golden[key][index]), rel=DOUBLE_TOLERANCE, abs=1e-14), (
                f"case {index} charge {charge}"
            )

        got = vec.calculate_average_bivariate_inverse_distance_charge(ncpr, case["sequence"])
        assert got == pytest.approx(float(golden["iwd_bivariate"][index]), rel=DOUBLE_TOLERANCE, abs=1e-14), (
            f"case {index}"
        )


def test_scd_shd_match_cython_golden(golden):
    for index in range(len(golden["sequence"])):
        case = _case(golden, index)

        assert vec.compute_scd_x(case["sequence"]) == pytest.approx(
            float(golden["scd"][index]), rel=1e-11, abs=1e-11
        ), f"case {index}"

        assert vec.compute_scd_x(case["sequence"], case["scd_group1"], case["scd_group2"]) == pytest.approx(
            float(golden["scd_custom"][index]), rel=1e-11, abs=1e-11
        ), f"case {index}"

        assert vec.compute_shd(case["sequence"]) == pytest.approx(float(golden["shd"][index]), rel=1e-11, abs=1e-11), (
            f"case {index}"
        )


# =====================================================================
# IWD -- edge cases and batch path
# =====================================================================


def test_iwd_no_targets_is_zero():
    assert vec.calculate_average_inverse_distance_from_sequence("AAAA", ["Q"]) == 0.0
    assert vec.iwd_from_binary_mask([0, 0, 0, 0]) == 0.0


def test_iwd_single_target_is_zero():
    """One residue has no partner, so there is no pairwise distance to sum."""
    assert vec.calculate_average_inverse_distance_from_sequence("AAQAA", ["Q"]) == 0.0
    assert vec.iwd_batch(np.array([[0, 0, 1, 0, 0]]))[0] == 0.0


def test_iwd_empty_sequence():
    assert vec.calculate_average_inverse_distance_from_sequence("", ["Q"]) == 0.0


def test_iwd_clustered_exceeds_dispersed():
    clustered = vec.calculate_average_inverse_distance_from_sequence("AAAAAQQQQQ", ["Q"])
    dispersed = vec.calculate_average_inverse_distance_from_sequence("AQAQAQAQAQ", ["Q"])
    assert clustered > dispersed


def test_iwd_batch_matches_per_sequence():
    rng = random.Random(13)
    for _ in range(25):
        length = rng.randint(20, 300)
        base = _random_sequence(rng, length)
        scrambles = ["".join(rng.sample(base, length)) for _ in range(48)]
        targets = rng.sample(list(AMINO_ACIDS), rng.randint(1, 6))

        expected = np.array([vec.calculate_average_inverse_distance_from_sequence(s, targets) for s in scrambles])
        masks = np.stack([_mask_from(s, targets) for s in scrambles])
        got = vec.iwd_batch(masks)

        assert np.allclose(expected, got, rtol=DOUBLE_TOLERANCE, atol=1e-14)


def test_iwd_batch_dense_masks():
    """The FFT autocorrelation path must hold up when most positions are hits."""
    rng = random.Random(14)
    length = 400
    base = _random_sequence(rng, length)
    scrambles = ["".join(rng.sample(base, length)) for _ in range(16)]
    targets = list("ACDEFGHIKL")  # ~50% occupancy

    expected = np.array([vec.calculate_average_inverse_distance_from_sequence(s, targets) for s in scrambles])
    got = vec.iwd_batch(np.stack([_mask_from(s, targets) for s in scrambles]))
    assert np.allclose(expected, got, rtol=DOUBLE_TOLERANCE, atol=1e-14)


def test_iwd_batch_all_positions_occupied():
    mask = np.ones((3, 25), dtype=bool)
    expected = vec.calculate_average_inverse_distance_from_sequence("Q" * 25, ["Q"])
    assert np.allclose(vec.iwd_batch(mask), expected, rtol=DOUBLE_TOLERANCE)


def test_iwd_from_binary_mask_weights():
    mask = np.array([1, 0, 1, 0, 1])
    uniform = vec.iwd_from_binary_mask(mask)
    # weighting every hit equally must reproduce the unweighted answer
    weighted = vec.iwd_from_binary_mask(mask, weights=np.full(5, 2.0))
    assert weighted == pytest.approx(uniform)


def test_iwd_from_binary_mask_zero_weights():
    assert vec.iwd_from_binary_mask([1, 0, 1], weights=[0.0, 0.0, 0.0]) == 0.0


# =====================================================================
# SCD / SHD
# =====================================================================


def test_scd_blocky_versus_mixed():
    """SCD is more negative for well-mixed charge, more positive for blocky."""
    blocky = vec.compute_scd_x("E" * 20 + "K" * 20)
    mixed = vec.compute_scd_x("EK" * 20)
    assert blocky < mixed


def test_scd_shd_short_sequences():
    assert vec.compute_scd_x("") == 0.0
    assert vec.compute_scd_x("E") == 0.0
    assert vec.compute_shd("") == 0.0
    assert vec.compute_shd("A") == 0.0


def test_shd_rejects_unknown_residue():
    with pytest.raises(ValueError, match="hydrophobicity dictionary"):
        vec.compute_shd("ACDEF", {"A": 0.5})
