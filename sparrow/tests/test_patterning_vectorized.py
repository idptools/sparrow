"""Tests for sparrow.patterning.vectorized.

These pin the NumPy implementations against **golden reference values captured
from the Cython extensions before they were removed** (see
``test_data/cython_patterning_golden_v1.npz``, 400 randomly-generated cases). The
extensions themselves no longer exist - ``sparrow.patterning.kappa`` and friends
are now thin re-exports of this module - so comparing against them directly would
be a self-comparison and prove nothing.

Two tolerances appear throughout:

* **IWD, SCD and SHD** were computed in ``double`` on both sides, so agreement is
  ~1e-13 or better.
* **kappa** carried its intermediates in C ``float`` in the extension and
  ``float64`` here, so agreement is ~1e-6. ``test_numpy_kappa_is_more_accurate``
  demonstrates the NumPy value is the better of the two rather than assuming it.
"""

import random

import numpy as np
import pytest

from sparrow.patterning import vectorized as vec
from sparrow.sparrow_exceptions import PatterningException
from sparrow.tools.track_tools import predefined_linear_track

from . import reference_data_path

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# groups spanning binary mode, singletons, and multi-residue sets
GROUP_PAIRS = [
    (["E", "D"], ["K", "R"]),
    (["Q"], ["N"]),
    (["G"], []),
    (["S", "T"], []),
    (["A"], ["G"]),
    (list("STNQCH"), list("ILMV")),
    (["W"], ["C"]),
]

# single-precision is ~1.2e-7 relative; allow a small multiple for accumulation
KAPPA_TOLERANCE = 1e-5
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


def test_golden_fixture_is_intact(golden):
    assert len(golden["sequence"]) == 400
    # the fixture must exercise both real values and the -1 sentinel
    assert np.count_nonzero(golden["kappa_x_flat"] == -1) > 5
    assert np.count_nonzero(golden["kappa_x_flat"] != -1) > 300


def test_kappa_x_matches_cython_golden(golden):
    for index in range(len(golden["sequence"])):
        case = _case(golden, index)
        for key, flatten in (("kappa_x_flat", 1), ("kappa_x_noflat", 0)):
            expected = float(golden[key][index])
            got = vec.kappa_x(
                case["sequence"],
                case["group1"],
                case["group2"],
                case["window_size"],
                flatten,
            )
            if expected == -1:
                assert got == -1
            else:
                assert got == pytest.approx(expected, rel=KAPPA_TOLERANCE, abs=1e-9), f"case {index}"


def test_calculate_sigma_matches_cython_golden(golden):
    for index in range(len(golden["sequence"])):
        case = _case(golden, index)
        got = vec.calculate_sigma(case["sequence"], case["group1"], case["group2"] or ["K"])
        assert got == pytest.approx(float(golden["sigma"][index]), rel=DOUBLE_TOLERANCE, abs=1e-15), f"case {index}"


def test_calculate_delta_matches_cython_golden(golden):
    for index in range(len(golden["sequence"])):
        case = _case(golden, index)
        got = vec.calculate_delta(
            case["sequence"],
            case["group1"],
            case["group2"] or ["K"],
            case["window_size"],
        )
        assert got == pytest.approx(float(golden["delta"][index]), rel=DOUBLE_TOLERANCE, abs=1e-15), f"case {index}"


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


def test_patterning_percentile_statistic_matches_cython_golden(golden):
    """The deterministic half of patterning_percentile.

    Only the observed statistic can be compared - the null distribution depends
    on the RNG, and the NumPy implementation deliberately uses a different one.
    """
    for index in range(len(golden["sequence"])):
        case = _case(golden, index)
        _, observed, _ = vec.patterning_percentile(
            case["sequence"],
            case["targets"],
            case["window_size"],
            1,
            1,
            return_distribution=True,
        )
        assert observed == pytest.approx(
            float(golden["percentile_observed"][index]),
            rel=KAPPA_TOLERANCE,
            abs=1e-9,
        ), f"case {index}"


def test_numpy_kappa_is_more_accurate_than_cython(golden):
    """The residual kappa difference is the extension's single precision.

    Replaying its arithmetic in float32 lands far closer to the golden values
    than float64 does, which is only possible if the gap comes from the compiled
    code's precision rather than from an error here.
    """

    def float32_replica(sequence, group1, group2, window):
        ternary = vec.ternarize_winged(sequence, group1, group2, window).astype(int)
        winged_length = ternary.size
        f32 = np.float32

        n_a = int((ternary == 1).sum())
        n_b = int((ternary == -1).sum())
        frac_a, frac_b = f32(n_a / winged_length), f32(n_b / winged_length)
        sigma = (
            f32(0.0) if (n_a + n_b) == 0 else f32(f32(frac_a - frac_b) * f32(frac_a - frac_b) / f32(frac_a + frac_b))
        )

        def delta(arr):
            total = f32(0.0)
            n_windows = winged_length - window + 1
            for start in range(n_windows):
                chunk = arr[start : start + window]
                a, b = int((chunk == 1).sum()), int((chunk == -1).sum())
                if a + b == 0:
                    asym = f32(0.0)
                else:
                    fa, fb = f32(a / window), f32(b / window)
                    asym = f32(f32(fa - fb) * f32(fa - fb) / f32(fa + fb))
                diff = f32(asym - sigma)
                total = f32(total + f32(diff * diff))
            return f32(total / f32(n_windows))

        segregated = vec._deltamax_ternary(ternary.astype(np.int8), window).astype(int)
        return float(f32(delta(ternary) / delta(segregated)))

    float32_errors, float64_errors = [], []
    for index in range(80):
        case = _case(golden, index)
        expected = float(golden["kappa_x_noflat"][index])
        if expected == -1:
            continue

        float64_errors.append(
            abs(
                expected
                - vec.kappa_x(
                    case["sequence"],
                    case["group1"],
                    case["group2"],
                    case["window_size"],
                    0,
                )
            )
        )
        float32_errors.append(
            abs(
                expected
                - float32_replica(
                    case["sequence"],
                    case["group1"],
                    case["group2"],
                    case["window_size"],
                )
            )
        )

    assert len(float64_errors) > 40
    assert np.mean(float32_errors) < np.mean(float64_errors) / 5


# =====================================================================
# kappa -- edge cases and error handling
# =====================================================================


def test_kappa_window_larger_than_sequence_returns_sentinel():
    assert vec.kappa_x("ACDEF", ["E"], ["K"], 10, 1) == -1


def test_kappa_absent_group_returns_sentinel():
    # no E/D anywhere
    assert vec.kappa_x("KKKKRRRRKKKK", ["E", "D"], ["K", "R"], 5, 1) == -1
    # no K/R anywhere
    assert vec.kappa_x("EEEEDDDDEEEE", ["E", "D"], ["K", "R"], 5, 1) == -1


def test_kappa_binary_mode_needs_only_group1():
    """With an empty group2 the second-group sentinel does not apply.

    Every residue is in group1 here, so the group2 count is zero - which would
    trip the sentinel if group2 had been supplied. In binary mode it must not.

    The value is 1.0, which is correct rather than degenerate: a single
    homogeneous block *is* the maximally segregated arrangement of its own
    composition, so delta equals delta_max.
    """
    value = vec.kappa_x("GGGGGGGGGGGG", ["G"], [], 5, 1)

    assert value != -1
    assert value == pytest.approx(1.0)

    # a group1 that is genuinely absent still trips the first sentinel
    assert vec.kappa_x("AAAAAAAAAAAA", ["G"], [], 5, 1) == -1


def test_kappa_rejects_small_window():
    with pytest.raises(PatterningException):
        vec.kappa_x("ACDEFGHIK", ["E"], ["K"], 1, 0)


def test_kappa_rejects_empty_group1():
    with pytest.raises(PatterningException):
        vec.kappa_x("ACDEFGHIK", [], ["K"], 5, 0)


def test_kappa_flatten_caps_at_one():
    # a hugely skewed composition can push kappa above 1
    sequence = "E" + "K" * 200
    unflattened = vec.kappa_x(sequence, ["E"], ["K"], 5, 0)
    flattened = vec.kappa_x(sequence, ["E"], ["K"], 5, 1)

    assert flattened <= 1.0
    if unflattened > 1:
        assert flattened == 1.0


def test_blocky_kappa_exceeds_mixed():
    blocky = vec.kappa_x("E" * 20 + "K" * 20, ["E"], ["K"], 5, 1)
    mixed = vec.kappa_x("EK" * 20, ["E"], ["K"], 5, 1)
    assert blocky > mixed
    assert 0 <= mixed <= blocky <= 1


# =====================================================================
# kappa -- batch path
# =====================================================================


def test_kappa_batch_matches_per_sequence():
    rng = random.Random(7)
    for _ in range(25):
        length = rng.randint(12, 150)
        base = _random_sequence(rng, length)
        scrambles = ["".join(rng.sample(base, length)) for _ in range(32)]
        group1, group2 = rng.choice(GROUP_PAIRS)
        window = rng.choice([3, 5, 6])

        expected = np.array([vec.kappa_x(s, list(group1), list(group2), window, 1) for s in scrambles])
        got = vec.kappa_x_batch(scrambles, list(group1), list(group2), window, 1)

        assert np.array_equal(expected == -1, got == -1)
        defined = expected != -1
        if defined.any():
            assert np.allclose(expected[defined], got[defined], rtol=KAPPA_TOLERANCE, atol=1e-9)


def test_kappa_batch_agrees_with_single_call():
    """The batch and scalar NumPy paths must be internally identical."""
    rng = random.Random(8)
    sequences = [_random_sequence(rng, 60) for _ in range(20)]

    batch = vec.kappa_x_batch(sequences, ["E", "D"], ["K", "R"], 5, 1)
    single = np.array([vec.kappa_x(s, ["E", "D"], ["K", "R"], 5, 1) for s in sequences])
    assert np.allclose(batch, single, rtol=1e-14, atol=1e-15)


def test_kappa_batch_handles_mixed_compositions():
    """delta_max is only shareable when compositions match; check the fallback."""
    sequences = ["EEEEEKKKKK", "EEEEEEEEKK", "EKEKEKEKEK"]
    batch = vec.kappa_x_batch(sequences, ["E"], ["K"], 5, 1)
    single = np.array([vec.kappa_x(s, ["E"], ["K"], 5, 1) for s in sequences])
    assert np.allclose(batch, single, rtol=1e-14, atol=1e-15)


def test_kappa_batch_empty_and_short():
    assert vec.kappa_x_batch([], ["E"], ["K"], 5, 1).shape == (0,)
    assert np.all(vec.kappa_x_batch(["ACD", "EFG"], ["E"], ["K"], 5, 1) == -1)


def test_kappa_batch_rejects_ragged_input():
    with pytest.raises(PatterningException):
        vec.kappa_x_batch(["ACDEFG", "ACDE"], ["E"], ["K"], 5, 1)


# =====================================================================
# IWD -- agreement with the Cython extension
# =====================================================================


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
# ternarize
# =====================================================================


def test_ternarize_winged_shape_and_content():
    ternary = vec.ternarize_winged("EKA", ["E"], ["K"], 2)

    assert ternary.shape == (2 * 2 + 3,)
    assert list(ternary) == [0, 0, 1, -1, 0, 0, 0]


def test_ternarize_binary_mode():
    ternary = vec.ternarize_winged("EKA", ["E"], [], 2)
    # wings stay 0; everything that is not E becomes -1
    assert list(ternary) == [0, 0, 1, -1, -1, 0, 0]


# =====================================================================
# patterning_percentile
# =====================================================================


def test_patterning_percentile_ranks_blocky_above_mixed():
    blocky = vec.patterning_percentile("Q" * 20 + "A" * 20, ["Q"], 5, 1000, 1)
    mixed = vec.patterning_percentile("QA" * 20, ["Q"], 5, 1000, 1)

    assert blocky > mixed
    assert blocky == pytest.approx(1.0)
    assert mixed == pytest.approx(0.0, abs=0.01)


def test_patterning_percentile_is_reproducible():
    kwargs = ("MEEEKKKKSSSTTTDDDQQQQNNNN", ["E", "D"], 5, 500, 17)
    assert vec.patterning_percentile(*kwargs) == vec.patterning_percentile(*kwargs)


def test_patterning_percentile_return_distribution():
    percentile, observed, null = vec.patterning_percentile(
        "MEEEKKKKSSSTTTDDDQQQQNNNN", ["E", "D"], 5, 200, 1, return_distribution=True
    )

    assert 0.0 <= percentile <= 1.0
    assert null.shape == (200,)
    # the null is returned sorted, as in the compiled version
    assert np.all(np.diff(null) >= 0)
    # and the percentile is the fraction of the null at or below the observation
    assert percentile == pytest.approx(np.count_nonzero(null <= observed) / 200)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(window_size=100), "larger than sequence"),
        (dict(window_size=1), "2 or larger"),
        (dict(group=[]), "group is empty"),
        (dict(seed="not-an-int"), "seed"),
    ],
)
def test_patterning_percentile_validation(kwargs, match):
    call = dict(sequence="MEEEKKKKSSSTTTDDD", group=["E"], window_size=5, count=10, seed=1)
    call.update(kwargs)
    with pytest.raises(PatterningException, match=match):
        vec.patterning_percentile(**call)


def test_binary_asymmetry():
    # half and half -> perfectly balanced
    assert vec.binary_asymmetry("QQQQAAAA", ["Q"]) == pytest.approx(0.0)
    # entirely one class -> maximally asymmetric
    assert vec.binary_asymmetry("QQQQQQQQ", ["Q"]) == pytest.approx(1.0)
    assert vec.binary_asymmetry("AAAAAAAA", ["Q"]) == pytest.approx(1.0)
    assert vec.binary_asymmetry("", ["Q"]) == 0.0


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


def test_ternarize_batch_matches_single():
    rng = random.Random(15)
    sequences = [_random_sequence(rng, 40) for _ in range(8)]
    codes = np.stack([np.frombuffer(s.encode("ascii"), dtype=np.uint8) for s in sequences])

    batch = vec.ternarize_winged_batch(codes, ["E", "D"], ["K", "R"], 5)
    for row, sequence in enumerate(sequences):
        assert np.array_equal(batch[row], vec.ternarize_winged(sequence, ["E", "D"], ["K", "R"], 5))
