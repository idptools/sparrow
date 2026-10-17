"""Tests for the windowed-asymmetry statistics in sparrow.properties.patterning.

kappa, sigma, delta and the patterning percentile are pinned against **golden
reference values captured from the Cython extensions before they were removed**
(``test_data/cython_patterning_golden_v1.npz``, 400 randomly generated cases).
kappa carried its intermediates in C ``float`` there and ``float64`` here, so it
agrees to ~1e-6, and ``test_kappa_is_more_accurate_than_the_cython_code`` shows
the gap is the compiled code's precision rather than an error here.

The last section pins the reason this module exists: NARDINI's omega and pair
features, kappa, delta and the patterning percentile are one statistic, computed
by one function, and agree with each other exactly.
"""

import random

import numpy as np
import pytest

from sparrow.nardini.groups import TYPES_8, patterning_feature_names
from sparrow.nardini.patterning import patterning_values
from sparrow.properties import encoding, patterning
from sparrow.sparrow_exceptions import PatterningException

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

# single precision is ~1.2e-7 relative; allow a small multiple for accumulation
KAPPA_TOLERANCE = 1e-5
DOUBLE_TOLERANCE = 1e-12


def _random_sequence(rng, length):
    return "".join(rng.choice(AMINO_ACIDS) for _ in range(length))


# =====================================================================
# golden reference values captured from the Cython extensions
# =====================================================================


@pytest.fixture(scope="module")
def golden():
    """400 cases scored by the Cython extensions before they were removed."""
    data = np.load(reference_data_path("cython_patterning_golden_v1.npz"), allow_pickle=False)
    return {key: data[key] for key in data.files}


def _case(golden, index):
    return dict(
        sequence=str(golden["sequence"][index]),
        group1=list(str(golden["group1"][index])),
        group2=list(str(golden["group2"][index])),
        targets=list(str(golden["targets"][index])),
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
        for key, flatten in (("kappa_x_flat", True), ("kappa_x_noflat", False)):
            expected = float(golden[key][index])
            got = patterning.kappa_x(case["sequence"], case["group1"], case["group2"], case["window_size"], flatten)[0]
            if expected == -1:
                assert got == -1
            else:
                assert got == pytest.approx(expected, rel=KAPPA_TOLERANCE, abs=1e-9), f"case {index}"


def test_sigma_matches_cython_golden(golden):
    for index in range(len(golden["sequence"])):
        case = _case(golden, index)
        got = patterning.sigma(case["sequence"], case["group1"], case["group2"] or ["K"])[0]
        assert got == pytest.approx(float(golden["sigma"][index]), rel=DOUBLE_TOLERANCE, abs=1e-15), f"case {index}"


def test_delta_matches_cython_golden(golden):
    for index in range(len(golden["sequence"])):
        case = _case(golden, index)
        got = patterning.delta(case["sequence"], case["group1"], case["group2"] or ["K"], case["window_size"])[0]
        assert got == pytest.approx(float(golden["delta"][index]), rel=DOUBLE_TOLERANCE, abs=1e-15), f"case {index}"


def test_patterning_percentile_statistic_matches_cython_golden(golden):
    """The deterministic half of patterning_percentile.

    Only the observed statistic can be compared - the null distribution depends
    on the RNG, and this implementation deliberately uses a different one.
    """
    for index in range(len(golden["sequence"])):
        case = _case(golden, index)
        _, observed, _ = patterning.patterning_percentile(
            case["sequence"], case["targets"], case["window_size"], 1, 1, return_distribution=True
        )
        assert observed[0] == pytest.approx(
            float(golden["percentile_observed"][index]), rel=KAPPA_TOLERANCE, abs=1e-9
        ), f"case {index}"


def test_kappa_is_more_accurate_than_the_cython_code(golden):
    """The residual kappa difference is the extension's single precision.

    Replaying its arithmetic in float32 lands far closer to the golden values
    than float64 does, which is only possible if the gap comes from the compiled
    code's precision rather than from an error here.
    """

    def float32_replica(sequence, group1, group2, window):
        ternary = encoding.ternarize_winged(sequence, group1, group2, window).astype(int)
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

        segregated = patterning.segregated_arrangements(ternary.astype(np.int8)[None, :], window)[0].astype(int)
        return float(f32(delta(ternary) / delta(segregated)))

    float32_errors, float64_errors = [], []
    for index in range(80):
        case = _case(golden, index)
        expected = float(golden["kappa_x_noflat"][index])
        if expected == -1:
            continue
        got = patterning.kappa_x(case["sequence"], case["group1"], case["group2"], case["window_size"])[0]
        float64_errors.append(abs(expected - got))
        float32_errors.append(
            abs(expected - float32_replica(case["sequence"], case["group1"], case["group2"], case["window_size"]))
        )

    assert len(float64_errors) > 40
    assert np.mean(float32_errors) < np.mean(float64_errors) / 5


# =====================================================================
# kappa -- edge cases and error handling
# =====================================================================


def test_kappa_window_larger_than_sequence_returns_sentinel():
    assert patterning.kappa_x("ACDEF", ["E"], ["K"], 10, True)[0] == -1


def test_kappa_absent_group_returns_sentinel():
    # no E/D anywhere, then no K/R anywhere
    assert patterning.kappa_x(["KKKKRRRRKKKK", "EEEEDDDDEEEE"], ["E", "D"], ["K", "R"], 5, True).tolist() == [-1, -1]


def test_kappa_binary_mode_needs_only_group1():
    """With an empty group2 the second-group sentinel does not apply.

    Every residue is in group1 here, so the group2 count is zero - which would
    trip the sentinel if group2 had been supplied. In binary mode it must not.

    The value is 1.0, which is correct rather than degenerate: a single
    homogeneous block *is* the maximally segregated arrangement of its own
    composition, so delta equals delta_max.
    """
    value = patterning.kappa_x("GGGGGGGGGGGG", ["G"], [], 5, True)[0]
    assert value == pytest.approx(1.0)

    # a group1 that is genuinely absent still trips the first sentinel
    assert patterning.kappa_x("AAAAAAAAAAAA", ["G"], [], 5, True)[0] == -1


def test_kappa_rejects_small_window():
    with pytest.raises(PatterningException):
        patterning.kappa_x("ACDEFGHIK", ["E"], ["K"], 1)


def test_kappa_rejects_empty_group1():
    with pytest.raises(PatterningException):
        patterning.kappa_x("ACDEFGHIK", [], ["K"], 5)


def test_kappa_flatten_caps_at_one():
    # a hugely skewed composition can push kappa above 1
    sequence = "E" + "K" * 200
    unflattened = patterning.kappa_x(sequence, ["E"], ["K"], 5)[0]
    flattened = patterning.kappa_x(sequence, ["E"], ["K"], 5, flatten=True)[0]

    assert flattened <= 1.0
    if unflattened > 1:
        assert flattened == 1.0


def test_blocky_kappa_exceeds_mixed():
    blocky, mixed = patterning.kappa_x(["E" * 20 + "K" * 20, "EK" * 20], ["E"], ["K"], 5, True)
    assert 0 <= mixed < blocky <= 1


def test_charge_kappa_is_the_mean_of_windows_five_and_six():
    sequence = "MEEEKKKKSSSTTTDDDQQQQNNNNGGGGRRRRDDDD"
    k5 = patterning.kappa_x(sequence, ["R", "K"], ["E", "D"], 5, True)[0]
    k6 = patterning.kappa_x(sequence, ["R", "K"], ["E", "D"], 6, True)[0]
    assert patterning.kappa(sequence)[0] == (k5 + k6) / 2
    assert patterning.kappa(["EK", "EKEKE"]).tolist() == [-1, -1]


# =====================================================================
# batches
# =====================================================================


def test_kappa_x_of_a_ragged_batch_matches_one_at_a_time():
    """Sequences of different lengths, scored together and alone, to the last bit."""
    rng = random.Random(7)
    sequences = [_random_sequence(rng, rng.randint(3, 150)) for _ in range(120)]
    for group1, group2 in GROUP_PAIRS:
        batch = patterning.kappa_x(sequences, group1, group2, 5, True)
        alone = [patterning.kappa_x(sequence, group1, group2, 5, True)[0] for sequence in sequences]
        assert batch.tolist() == alone


def test_kappa_from_ternary_scores_scrambles_as_kappa_x_does():
    rng = random.Random(8)
    for _ in range(10):
        length = rng.randint(12, 150)
        base = _random_sequence(rng, length)
        scrambles = ["".join(rng.sample(base, length)) for _ in range(32)]
        group1, group2 = rng.choice(GROUP_PAIRS)
        window = rng.choice([3, 5, 6])

        codes = np.stack([encoding.residue_codes(s) for s in scrambles])
        ternary = encoding.ternarize_winged_batch(codes, group1, group2, window)
        got = patterning.kappa_from_ternary(ternary, window, True, group2_present=len(group2) > 0)
        expected = patterning.kappa_x(scrambles, group1, group2, window, True)
        assert got.tolist() == expected.tolist()


def test_kappa_x_of_an_empty_batch():
    assert patterning.kappa_x([], ["E"], ["K"], 5).shape == (0,)


def test_delta_of_a_sequence_shorter_than_the_window_is_nan():
    assert np.isnan(patterning.delta("EKE", ["E"], ["K"], 5)[0])


# =====================================================================
# patterning percentile
# =====================================================================


def test_patterning_percentile_ranks_blocky_above_mixed():
    blocky, mixed = patterning.patterning_percentile(["Q" * 20 + "A" * 20, "QA" * 20], ["Q"], 5, 1000, 1)
    assert blocky == pytest.approx(1.0)
    assert mixed == pytest.approx(0.0, abs=0.01)


def test_patterning_percentile_is_reproducible_and_batch_independent():
    sequence = "MEEEKKKKSSSTTTDDDQQQQNNNN"
    alone = patterning.patterning_percentile(sequence, ["E", "D"], 5, 500, 17)
    together = patterning.patterning_percentile(["GSGSGSGSGS", sequence], ["E", "D"], 5, 500, 17)
    assert alone[0] == together[1]


def test_patterning_percentile_return_distribution():
    percentile, observed, null = patterning.patterning_percentile(
        "MEEEKKKKSSSTTTDDDQQQQNNNN", ["E", "D"], 5, 200, 1, return_distribution=True
    )
    assert null.shape == (1, 200)
    # the null is returned sorted, as in the compiled version
    assert np.all(np.diff(null[0]) >= 0)
    # and the percentile is the fraction of the null at or below the observation
    assert percentile[0] == pytest.approx(np.count_nonzero(null[0] <= observed[0]) / 200)


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
    call = dict(sequences="MEEEKKKKSSSTTTDDD", group=["E"], window_size=5, count=10, seed=1)
    call.update(kwargs)
    with pytest.raises(PatterningException, match=match):
        patterning.patterning_percentile(**call)


def test_binary_asymmetry():
    values = patterning.binary_asymmetry(["QQQQAAAA", "QQQQQQQQ", "AAAAAAAA", ""], ["Q"])
    assert values.tolist() == [0.0, 1.0, 1.0, 0.0]


# =====================================================================
# one statistic: NARDINI, kappa's delta, delta and the percentile agree
# =====================================================================

NARDINI_SEQUENCES = [
    "MEEEKKKKSSSTTTDDDQQQQNNNNGGGSSSEEEKKKWWWAAPPGSGSRRDE",
    "GRGRGGYGGSSSSQQQQAAAPPPGSGSKKKEEE" * 2,
]


@pytest.mark.parametrize("sequence", NARDINI_SEQUENCES)
def test_nardini_omega_is_the_patterning_percentile_statistic(sequence):
    """One type against all the rest, without wings, is exactly the percentile's delta."""
    names = patterning_feature_names("8")
    values = patterning_values(sequence, min_fraction=0.0)
    compared = 0
    for type_name, residues in TYPES_8.items():
        if not any(residue in sequence for residue in residues):
            continue  # NARDINI does not evaluate an absent type, and reports 0
        _, observed, _ = patterning.patterning_percentile(sequence, list(residues), 5, 1, 1, return_distribution=True)
        assert values[names.index(f"{type_name}-{type_name}")] == observed[0], type_name
        compared += 1
    assert compared >= 5


@pytest.mark.parametrize("sequence", NARDINI_SEQUENCES)
def test_nardini_pair_features_are_delta(sequence):
    """Two types against each other, without wings, is exactly Das & Pappu's delta."""
    names = patterning_feature_names("8")
    values = patterning_values(sequence, min_fraction=0.0)
    present = [(name, residues) for name, residues in TYPES_8.items() if any(r in sequence for r in residues)]
    compared = 0
    for i, (name_i, residues_i) in enumerate(present):
        for name_j, residues_j in present[i + 1 :]:
            expected = patterning.delta(sequence, list(residues_i), list(residues_j), 5)[0]
            assert values[names.index(f"{name_i}-{name_j}")] == expected, (name_i, name_j)
            compared += 1
    assert compared >= 10
