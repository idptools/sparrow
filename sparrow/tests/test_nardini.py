"""Tests for the sparrow.nardini module.

The patterning tests include a deliberately naive reference implementation
transcribed from the NARDINI+ reference notebook (``_reference_*`` below). The
production code is a vectorized rewrite, so keeping the slow version here and
asserting bit-level agreement is what guarantees the rewrite did not change any
numbers.
"""

import math
import random
import warnings

import numpy as np
import pytest

from sparrow.nardini import (
    NardiniBackground,
    build_background,
    composition_feature_names,
    composition_values,
    nardini_vectors,
    patterning_feature_names,
    patterning_values,
    patterning_zscores,
)
from sparrow.nardini.composition import charge_at_pH, isoelectric_point
from sparrow.nardini.groups import TYPES_8, TYPES_9, get_typeset, type_names
from sparrow.sparrow_exceptions import (
    BackgroundQualityWarning,
    SparrowException,
)

SMARCC2_LIKE = (
    "MENDPRVRKFASDGAQWAIKWQKKGWSTLTSRQKQTARAAMGIKLSPVAQPVQKVTRLSAPVALAYREVSTQPRVSTARD"
)
RG_RICH = "GRGRGGYGGRGGYGGSRGGYGGSRGGRGGYGGSRGGRGGRGGYGGRGRGRG"
BLOCKY_EK = "EEEEEEEEEEEEEEEKKKKKKKKKKKKKKK"
MIXED_EK = "EKEKEKEKEKEKEKEKEKEKEKEKEKEKEK"

# The synthetic fixtures here are deliberately small, so build_background warns
# that they are weak. That warning is asserted directly in the dedicated tests
# below (pytest.warns overrides this filter); everywhere else it is just noise.
pytestmark = pytest.mark.filterwarnings(
    "ignore::sparrow.sparrow_exceptions.BackgroundQualityWarning"
)


# =====================================================================
# Reference implementation, transcribed from the NARDINI+ notebook
# =====================================================================

_REF_TYPES = [
    list("STNQCH"),
    list("ILMV"),
    list("RK"),
    list("ED"),
    list("FWY"),
    ["A"],
    ["P"],
    ["G"],
]


def _reference_omega(seq, type1, b=5):
    count = sum(seq.count(res) for res in type1)
    sigAll = ((count / len(seq)) - (1 - (count / len(seq)))) ** 2

    sigX = []
    for x in range(0, len(seq) - b + 1):
        subseq = seq[x : x + b]
        count = sum(subseq.count(res) for res in type1)
        sigX.append(((count / b) - (1 - (count / b))) ** 2)

    return float(np.mean([(v - sigAll) ** 2 for v in sigX]))


def _reference_kappa(seq, type1, type2, b=5):
    count1 = sum(seq.count(res) for res in type1)
    count2 = sum(seq.count(res) for res in type2)
    sigAll = ((count1 / len(seq)) - (count2 / len(seq))) ** 2 / (
        (count1 / len(seq)) + (count2 / len(seq))
    )

    sigX = []
    for x in range(0, len(seq) - b + 1):
        subseq = seq[x : x + b]
        c1 = sum(subseq.count(res) for res in type1)
        c2 = sum(subseq.count(res) for res in type2)
        if c1 + c2 == 0:
            sigX.append(0)
        else:
            sigX.append(((c1 / b) - (c2 / b)) ** 2 / ((c1 / b) + (c2 / b)))

    return float(np.mean([(v - sigAll) ** 2 for v in sigX]))


def _reference_patterning_values(seq, min_fraction=0.10, b=5):
    """Upper-triangle raw asymmetries, in canonical feature order."""
    fracs = [sum(seq.count(r) for r in group) / len(seq) for group in _REF_TYPES]

    values = []
    for i in range(len(_REF_TYPES)):
        for j in range(i, len(_REF_TYPES)):
            if i == j:
                if fracs[i] > min_fraction:
                    values.append(_reference_omega(seq, _REF_TYPES[i], b))
                else:
                    values.append(0.0)
            else:
                if fracs[i] > min_fraction and fracs[j] > min_fraction:
                    values.append(
                        _reference_kappa(seq, _REF_TYPES[i], _REF_TYPES[j], b)
                    )
                else:
                    values.append(0.0)
    return np.array(values)


# =====================================================================
# groups
# =====================================================================


def test_typesets_are_exact_partitions():
    from sparrow.data.amino_acids import VALID_AMINO_ACIDS

    for typeset in (TYPES_8, TYPES_9):
        residues = [r for group in typeset.values() for r in group]
        assert sorted(residues) == sorted(VALID_AMINO_ACIDS)
        assert len(residues) == len(set(residues))


def test_type_names_and_feature_counts():
    assert type_names("8") == ("pol", "hyd", "pos", "neg", "aro", "ala", "pro", "gly")
    assert len(patterning_feature_names("8")) == 36
    # 9 types -> 9*10/2 = 45 upper-triangle features
    assert len(patterning_feature_names("9")) == 45


def test_feature_names_match_notebook_order():
    names = patterning_feature_names("8")
    assert names[0] == "pol-pol"
    assert names[1] == "pol-hyd"
    assert names[-1] == "gly-gly"

    expected = []
    labels = ["pol", "hyd", "pos", "neg", "aro", "ala", "pro", "gly"]
    for i in range(len(labels)):
        for j in range(i, len(labels)):
            expected.append(f"{labels[i]}-{labels[j]}")
    assert list(names) == expected


def test_nine_type_splits_histidine_out():
    assert "H" in TYPES_8["pol"]
    assert "H" not in TYPES_9["pol"]
    assert TYPES_9["his"] == ("H",)


def test_unknown_typeset_raises():
    with pytest.raises(SparrowException):
        get_typeset("17")


def test_custom_typeset_must_be_a_partition():
    with pytest.raises(ValueError):
        get_typeset({"a": tuple("AC"), "b": tuple("CD")})


# =====================================================================
# patterning -- agreement with the reference implementation
# =====================================================================


@pytest.mark.parametrize(
    "sequence",
    [SMARCC2_LIKE, RG_RICH, BLOCKY_EK, MIXED_EK, "QQQQNNNNSSSSTTTTGGGGPPPP"],
)
def test_patterning_values_match_reference(sequence):
    assert np.allclose(
        patterning_values(sequence),
        _reference_patterning_values(sequence),
        rtol=1e-12,
        atol=1e-15,
    )


def test_patterning_values_match_reference_on_random_sequences():
    rng = random.Random(11)
    for _ in range(15):
        length = rng.randint(20, 200)
        sequence = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(length))
        assert np.allclose(
            patterning_values(sequence),
            _reference_patterning_values(sequence),
            rtol=1e-12,
            atol=1e-15,
        )


def test_scramble_scoring_matches_reference():
    """The vectorized batch path must score scrambles exactly as the reference does."""
    from sparrow.nardini import patterning as P

    rng = random.Random(3)
    scrambles = [
        "".join(rng.sample(SMARCC2_LIKE, len(SMARCC2_LIKE))) for _ in range(25)
    ]

    labels = np.stack([P.encode_types(s) for s in scrambles])
    fractions = P.type_fractions(P.encode_types(SMARCC2_LIKE), 8)
    pairs = P._pair_index(8)
    counts = P._window_counts(labels, 8, 5)
    got = P._asymmetries_from_counts(counts, fractions, pairs, 5)
    got[:, ~P._evaluated_mask(fractions, pairs, 0.10)] = 0.0

    for row, scramble in enumerate(scrambles):
        assert np.allclose(
            got[row], _reference_patterning_values(scramble), rtol=1e-12, atol=1e-15
        )


# =====================================================================
# patterning -- behaviour
# =====================================================================


def test_patterning_zscore_length_and_order():
    z = patterning_zscores(SMARCC2_LIKE, num_scrambles=200, seed=1)
    assert z.shape == (36,)


def test_blocky_is_more_segregated_than_mixed():
    """The core NARDINI signal: positive z = blockier than its own scrambles."""
    blocky = patterning_zscores(BLOCKY_EK, num_scrambles=2000, seed=1)
    mixed = patterning_zscores(MIXED_EK, num_scrambles=2000, seed=1)

    names = patterning_feature_names("8")
    idx = names.index("pos-neg")

    assert blocky[idx] > 5
    assert mixed[idx] < 0
    assert blocky[idx] > mixed[idx]


@pytest.mark.parametrize("num_scrambles", [500, 2000, 5000])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_patterning_zscore_is_stable_across_seeds(num_scrambles, seed):
    """Guard against the gamma MLE converging to a nonsense optimum.

    scipy's 3-parameter gamma fit occasionally lands in a bad local optimum and
    returns moments that do not describe the scramble distribution, which used to
    collapse this z-score from ~12 to ~0.4 for particular (seed, num_scrambles)
    combinations. _fit_gamma_moments now sanity-checks the fit, so the value must
    be stable regardless of sampling.
    """
    z = patterning_zscores(BLOCKY_EK, num_scrambles=num_scrambles, seed=seed)
    idx = patterning_feature_names("8").index("pos-neg")
    assert 8.0 < z[idx] < 18.0


def test_bad_gamma_fit_falls_back_to_sample_moments():
    """A fit whose moments disagree wildly with the data must be discarded."""
    from sparrow.nardini.patterning import _fit_gamma_moments

    rng = np.random.default_rng(0)
    column = rng.gamma(shape=2.0, scale=0.05, size=4000)

    mean, var = _fit_gamma_moments(column)

    # whichever path was taken, the answer must describe the data
    assert mean == pytest.approx(column.mean(), rel=0.5)
    assert var == pytest.approx(column.var(), rel=10.0)
    assert var > 0


def test_degenerate_scramble_distribution_is_handled():
    """An all-identical null (zero variance) must give z=0, not a divide error."""
    from sparrow.nardini.patterning import _fit_gamma_moments

    mean, var = _fit_gamma_moments(np.full(100, 0.25))
    assert mean == pytest.approx(0.25)
    assert var == 0.0


def test_rare_types_are_gated_to_zero():
    """A type below min_fraction must not be scored."""
    # only Q/N/S (all 'pol') plus a single alanine -> ala fraction way below 0.10
    sequence = "QQQQQQQQQQNNNNNNNNNNSSSSSSSSSA"
    names = patterning_feature_names("8")
    values = patterning_values(sequence)
    z = patterning_zscores(sequence, num_scrambles=200, seed=1)

    for feature in ("ala-ala", "pol-ala"):
        i = names.index(feature)
        assert values[i] == 0.0
        assert z[i] == 0.0


def test_min_fraction_is_a_strict_threshold():
    """A fraction exactly equal to min_fraction is NOT scored."""
    # 30 residues, exactly 3 alanines -> ala fraction is exactly 0.10
    sequence = "A" + "QQQQQQQQQ" + "A" + "QQQQQQQQQ" + "A" + "QQQQQQQQQ"
    assert sequence.count("A") / len(sequence) == pytest.approx(0.10)

    names = patterning_feature_names("8")
    values = patterning_values(sequence)
    assert values[names.index("ala-ala")] == 0.0


def test_patterning_is_reproducible_with_a_seed():
    a = patterning_zscores(SMARCC2_LIKE, num_scrambles=500, seed=99)
    b = patterning_zscores(SMARCC2_LIKE, num_scrambles=500, seed=99)
    assert np.array_equal(a, b)


def test_patterning_details():
    z, details = patterning_zscores(
        SMARCC2_LIKE, num_scrambles=300, seed=1, return_details=True
    )

    assert details["scramble_values"].shape == (300, 36)
    assert details["raw"].shape == (36,)
    assert details["type_fractions"].shape == (8,)
    assert details["evaluated"].dtype == bool
    # unevaluated features carry no null statistics
    assert np.all(np.isnan(details["null_mean"][~details["evaluated"]]))
    # every z-score for an unevaluated feature is exactly zero
    assert np.all(z[~details["evaluated"]] == 0)


def test_patterning_rejects_short_sequence():
    with pytest.raises(SparrowException):
        patterning_values("ACDE")  # shorter than blob_size=5


def test_patterning_rejects_nonstandard_residue():
    with pytest.raises(SparrowException):
        patterning_values("ACDEFGHIKLXMNPQ")


def test_patterning_rejects_bad_num_scrambles():
    with pytest.raises(SparrowException):
        patterning_zscores(SMARCC2_LIKE, num_scrambles=0)


def test_patterning_handles_single_type_sequence():
    """A sequence of one type has zero asymmetry, so every z-score is 0."""
    # Q, N and S are all 'pol'
    sequence = "QQQQQQQQQQNNNNNNNNNNSSSSSSSSSS"
    z = patterning_zscores(sequence, num_scrambles=200, seed=1)
    assert np.all(z == 0)


def test_nine_type_grouping_runs():
    z = patterning_zscores(SMARCC2_LIKE, num_scrambles=200, seed=1, typeset="9")
    assert z.shape == (45,)


# =====================================================================
# patterning -- alternative backends
# =====================================================================


def test_backend_registry():
    from sparrow.nardini import PATTERNING_BACKENDS

    assert set(PATTERNING_BACKENDS) == {"nardini", "kappa", "iwd"}


@pytest.mark.parametrize(
    "given,expected",
    [
        ("nardini", "nardini"),
        ("kappa", "kappa"),
        ("iwd", "iwd"),
        # the legacy sparrow.sequence_analysis.grammar names
        ("kappa_cython", "kappa"),
        ("iwd_combined", "iwd"),
        ("KAPPA", "kappa"),
        ("  Nardini ", "nardini"),
    ],
)
def test_resolve_backend(given, expected):
    from sparrow.nardini import resolve_backend

    assert resolve_backend(given) == expected


def test_resolve_backend_rejects_unknown():
    from sparrow.nardini import resolve_backend

    with pytest.raises(SparrowException, match="Unknown patterning backend"):
        resolve_backend("not-a-backend")


@pytest.mark.parametrize(
    "grammar_backend,nardini_backend,prefix",
    [("kappa_cython", "kappa", "kappa::"), ("iwd_combined", "iwd", "iwd::")],
)
@pytest.mark.parametrize(
    "sequence", [SMARCC2_LIKE, RG_RICH, BLOCKY_EK, "QQQQNNNNSSSSTTTTGGGGPPPPAAAA"]
)
def test_ported_backends_match_grammar(
    grammar_backend, nardini_backend, prefix, sequence
):
    """The ported backends must reproduce sparrow.sequence_analysis.grammar.

    Agreement is to single-precision tolerance rather than bit-for-bit: grammar
    goes through the Cython extensions, which carry the kappa intermediates in C
    ``float``, whereas these backends use the double-precision NumPy
    reimplementations. Where the two differ, the NumPy value is the more accurate
    one - see test_patterning_vectorized.py, which pins that claim.
    """
    from sparrow.sequence_analysis import grammar

    config = grammar.GrammarPatterningConfig(backend=grammar_backend)
    reference = grammar.compute_patterning_raw(sequence, config)

    names = patterning_feature_names("8")
    expected = np.array([reference[prefix + name] for name in names])
    got = patterning_values(sequence, backend=nardini_backend)

    # iwd is double precision on both sides; kappa is the single-precision one
    tolerance = 1e-12 if nardini_backend == "iwd" else 1e-5
    assert np.allclose(expected, got, rtol=tolerance, atol=1e-12)


@pytest.mark.parametrize("backend", ["nardini", "kappa", "iwd"])
def test_every_backend_produces_a_full_vector(backend):
    z = patterning_zscores(SMARCC2_LIKE, num_scrambles=200, seed=1, backend=backend)
    assert z.shape == (36,)
    assert np.all(np.isfinite(z))


@pytest.mark.parametrize("backend", ["nardini", "kappa"])
def test_window_based_backends_separate_blocky_from_mixed(backend):
    """Both window-based statistics should call a blocky sequence segregated."""
    names = patterning_feature_names("8")
    idx = names.index("pos-neg")

    blocky = patterning_zscores(BLOCKY_EK, num_scrambles=2000, seed=1, backend=backend)
    mixed = patterning_zscores(MIXED_EK, num_scrambles=2000, seed=1, backend=backend)

    assert blocky[idx] > 5
    assert mixed[idx] < 0


def test_iwd_backend_ignores_blob_size():
    """IWD is a clustering metric with no window, so blob_size cannot matter."""
    a = patterning_values(SMARCC2_LIKE, backend="iwd", blob_size=5)
    b = patterning_values(SMARCC2_LIKE, backend="iwd", blob_size=9)
    assert np.array_equal(a, b)

    # whereas the window-based backends do depend on it
    c = patterning_values(SMARCC2_LIKE, backend="nardini", blob_size=5)
    d = patterning_values(SMARCC2_LIKE, backend="nardini", blob_size=9)
    assert not np.array_equal(c, d)


@pytest.mark.parametrize("num_scrambles", [100, 200, 500, 1000, 2000, 5000])
def test_iwd_two_letter_sequence_gives_zero_zscores(num_scrambles):
    """Every position is in the merged group, so IWD is scramble-invariant.

    Not a bug: with only E and K present the merged {D,E,K,R} group covers the
    whole sequence, so the statistic cannot change under permutation and the null
    is a point mass. z=0 is the correct answer.

    Parametrized over num_scrambles deliberately. Whether ``np.var`` of N
    identical values returns *exactly* zero depends on N - NumPy's pairwise
    summation blocks the mean subtraction differently - so at 500, 1000 and 2000
    it returns 3e-30 while at 100, 200 and 5000 it returns 0.0. An earlier
    version of this test only ran at 200 and so never noticed that the
    degenerate-null guard was an equality test against zero, which let those
    middle sizes divide a ~1e-15 numerator by a ~1e-15 denominator and report a
    confident-looking z of 2.0.
    """
    names = patterning_feature_names("8")
    z, details = patterning_zscores(
        BLOCKY_EK,
        num_scrambles=num_scrambles,
        seed=1,
        backend="iwd",
        return_details=True,
    )
    idx = names.index("pos-neg")

    column = details["scramble_values"][:, idx]
    assert np.ptp(column) == 0.0, "the null should be a point mass"
    assert z[idx] == 0.0


def test_degenerate_null_is_detected_regardless_of_float_noise():
    """A point mass must be recognized even when np.var reports 1e-30, not 0."""
    from sparrow.nardini.patterning import _fit_gamma_moments

    column = np.full(500, 8.374755)
    assert np.var(column) > 0, "this fixture is pointless if np.var returns 0"

    mean, var = _fit_gamma_moments(column)
    assert mean == pytest.approx(8.374755)
    assert var == 0.0


def test_iwd_backend_is_informative_on_a_real_sequence():
    z = patterning_zscores(SMARCC2_LIKE, num_scrambles=2000, seed=1, backend="iwd")
    assert np.count_nonzero(z) > 0


def test_backends_disagree_numerically():
    """Sanity: these are different statistics, not rescalings of each other."""
    a = patterning_values(SMARCC2_LIKE, backend="nardini")
    b = patterning_values(SMARCC2_LIKE, backend="kappa")
    c = patterning_values(SMARCC2_LIKE, backend="iwd")

    assert not np.allclose(a, b)
    assert not np.allclose(a, c)
    assert not np.allclose(b, c)


def test_backend_recorded_on_result():
    from sparrow.nardini import nardini_vectors as nv

    for backend in ("nardini", "kappa", "iwd"):
        result = nv({"a": SMARCC2_LIKE}, num_scrambles=100, seed=1, backend=backend)
        assert result.backend == backend

    # legacy alias is normalized on the way in
    assert (
        nv(
            {"a": SMARCC2_LIKE}, num_scrambles=100, seed=1, backend="iwd_combined"
        ).backend
        == "iwd"
    )


def test_backend_does_not_affect_composition_block(idr_set):
    background = build_background(idr_set)
    kwargs = dict(background=background, num_scrambles=100, seed=1)

    a = nardini_vectors({"x": SMARCC2_LIKE}, backend="nardini", **kwargs)
    b = nardini_vectors({"x": SMARCC2_LIKE}, backend="iwd", **kwargs)

    assert np.array_equal(a.composition_zscores, b.composition_zscores)
    assert not np.array_equal(a.patterning_zscores, b.patterning_zscores)


# =====================================================================
# composition
# =====================================================================


def test_composition_feature_count_and_names():
    names = composition_feature_names()
    assert len(names) == 54
    assert names[0] == "Frac A"
    assert names[-1] == "RG Frac"
    assert "W Patch" not in names  # deliberately excluded, as in NARDINI+
    assert "Frac W" in names  # but the W *fraction* is present


def test_composition_values_keys_match_names():
    values = composition_values(SMARCC2_LIKE)
    assert list(values) == list(composition_feature_names())


def test_composition_known_values():
    # M E3 K4 S3 T3 D3 Q4 N4  ->  length 25
    sequence = "MEEEKKKKSSSTTTDDDQQQQNNNN"
    assert len(sequence) == 25
    values = composition_values(sequence)

    assert values["Frac E"] == pytest.approx(3 / 25)
    assert values["Frac K"] == pytest.approx(4 / 25)
    assert values["FCR"] == pytest.approx((3 + 3 + 4) / 25)  # E + D + K
    assert values["NCPR"] == pytest.approx((4 - 6) / 25)  # K - (E + D)
    assert values["Frac K+R"] == pytest.approx(4 / 25)
    assert values["Frac D+E"] == pytest.approx(6 / 25)
    # no prolines, so chain-expanding == FCR
    assert values["Frac Chain Expanding"] == pytest.approx(values["FCR"])
    # R/K with the +1 pseudo-count: log10((0+1)/(4+1))
    assert values["R/K Ratio"] == pytest.approx(math.log10(1 / 5))
    # equal E and D counts -> log10(1) == 0
    assert values["E/D Ratio"] == pytest.approx(0.0)


def test_composition_patch_features():
    # 10 consecutive Qs is comfortably a patch; a lone Q is not
    values = composition_values("AAAAAQQQQQQQQQQAAAAA")
    assert values["Q Patch"] == pytest.approx(0.5)
    assert values["A Patch"] == pytest.approx(0.5)

    values = composition_values("AAAAAQAAAAA")
    assert values["Q Patch"] == 0.0


def test_composition_rg_fraction():
    assert composition_values(RG_RICH)["RG Frac"] > 0.5
    assert composition_values("QQQQQQQQQQAAAAAAAAAA")["RG Frac"] == 0.0


def test_composition_rejects_empty_and_nonstandard():
    with pytest.raises(SparrowException):
        composition_values("")
    with pytest.raises(SparrowException):
        composition_values("ACDEFX")


def test_charge_at_ph_extremes():
    # at very low pH the acidic groups are protonated and bases are charged
    assert charge_at_pH("KKKKEEEE", pH=0.0) > 0
    # at very high pH the reverse
    assert charge_at_pH("KKKKEEEE", pH=14.0) < 0
    # no titratable residues -> zero
    assert charge_at_pH("AAAAGGGG", pH=7.4) == 0.0


def test_isoelectric_point_sanity():
    # a strongly basic sequence has a high pI, an acidic one a low pI
    assert isoelectric_point("KKKKKKRRRRRR") > 10
    assert isoelectric_point("EEEEEEDDDDDD") < 5
    # a sequence with no titratable residues has no defined pI; 7.0 by convention
    assert isoelectric_point("AAAAGGGG") == pytest.approx(7.0)


def test_composition_matrix_shape():
    from sparrow.nardini import composition_matrix

    assert composition_matrix([SMARCC2_LIKE, RG_RICH]).shape == (2, 54)
    assert composition_matrix([]).shape == (0, 54)


# =====================================================================
# background
# =====================================================================


@pytest.fixture
def idr_set():
    """A small synthetic set of IDR-like sequences."""
    rng = random.Random(5)
    alphabet = "SSSTTTGGGQQQNNNPPPEEEDDDKKKRRRAAALLLVVVMIYFWHC"
    return {
        f"idr_{i}": "".join(rng.choice(alphabet) for _ in range(rng.randint(40, 150)))
        for i in range(60)
    }


@pytest.fixture
def idr_fasta(tmp_path, idr_set):
    path = tmp_path / "idrs.fasta"
    with open(path, "w") as handle:
        for name, sequence in idr_set.items():
            handle.write(f">{name}\n{sequence}\n")
    return str(path)


def test_build_background_from_dict(idr_set):
    background = build_background(idr_set)

    assert len(background) == 54
    assert background.num_sequences == len(idr_set)
    assert tuple(background.feature_names) == composition_feature_names()
    assert np.all(np.isfinite(background.mean))
    assert np.all(background.std >= 0)


def test_build_background_from_fasta(idr_fasta, idr_set):
    background = build_background(idr_fasta)
    assert background.num_sequences == len(idr_set)
    assert background.metadata["source"] == idr_fasta


def test_paths_may_be_pathlib_objects(tmp_path, idr_fasta, idr_set):
    """Every path argument accepts os.PathLike, not just str.

    A pathlib.Path used to fall through the `isinstance(source, str)` check and
    get treated as an iterable of sequences, failing with an opaque
    "'PosixPath' object is not iterable".
    """
    from pathlib import Path

    from sparrow import Protein

    fasta = Path(idr_fasta)

    from_path = build_background(fasta)
    from_string = build_background(idr_fasta)
    assert np.array_equal(from_path.mean, from_string.mean)
    # the recorded provenance is a plain string either way, not a repr
    assert isinstance(from_path.metadata["source"], str)

    vectors = nardini_vectors(fasta, num_scrambles=100, seed=1)
    assert len(vectors) == len(idr_set)

    saved = tmp_path / "background.npz"
    from_path.save(saved)
    assert isinstance(NardiniBackground.load(saved), NardiniBackground)

    # and as the `background` argument, on both the module and Protein entry points
    assert nardini_vectors(
        idr_set, background=saved, num_scrambles=100, seed=1
    ).has_composition
    assert (
        len(Protein(next(iter(idr_set.values()))).nardini.composition(background=saved))
        == 54
    )


def test_build_background_from_list(idr_set):
    background = build_background(list(idr_set.values()))
    assert background.num_sequences == len(idr_set)


def test_build_background_from_protein_objects(idr_set):
    from sparrow import Protein

    proteins = {name: Protein(seq) for name, seq in idr_set.items()}
    background = build_background(proteins)
    assert background.num_sequences == len(idr_set)


def test_background_mean_std_match_numpy(idr_set):
    """The online accumulation must agree with a direct numpy calculation."""
    from sparrow.nardini import composition_matrix

    background = build_background(idr_set)
    matrix = composition_matrix(list(idr_set.values()))

    assert np.allclose(background.mean, matrix.mean(axis=0))
    assert np.allclose(background.std, matrix.std(axis=0))


def test_background_skips_nonstandard_sequences():
    sequences = {
        "good_1": "MEEEKKKKSSSTTTDDDQQQQNNNN",
        "bad_x": "MEEEKKKKSSSXTTTDDDQQQNNNN",
        "bad_u": "MEEEKKKKSSSUTTTDDDQQQNNNN",
        "good_2": "QQQQNNNNSSSSTTTTGGGGPPPPP",
    }
    background = build_background(sequences)

    assert background.num_sequences == 2
    assert background.metadata["num_sequences_skipped"] == 2


def test_background_min_length_filter(idr_set):
    background = build_background(idr_set, min_length=100)
    expected = sum(1 for s in idr_set.values() if len(s) >= 100)

    assert background.num_sequences == expected
    assert background.metadata["min_length"] == 100


def test_build_background_with_no_usable_sequences_raises():
    with pytest.raises(SparrowException):
        build_background({"bad": "XXXXXXXX"})


def test_build_background_missing_file_raises():
    with pytest.raises(SparrowException):
        build_background("/nonexistent/path/to/idrs.fasta")


def test_background_metadata_is_preserved(idr_set):
    background = build_background(idr_set, metadata={"species": "Homo sapiens"})
    assert background.metadata["species"] == "Homo sapiens"
    assert background.metadata["num_sequences_used"] == len(idr_set)


def test_background_save_load_roundtrip(tmp_path, idr_set):
    background = build_background(idr_set, metadata={"species": "test"})
    path = background.save(tmp_path / "bg")

    assert path.endswith(".npz")

    loaded = NardiniBackground.load(path)
    assert loaded.feature_names == background.feature_names
    assert np.array_equal(loaded.mean, background.mean)
    assert np.array_equal(loaded.std, background.std)
    assert loaded.num_sequences == background.num_sequences
    assert loaded.metadata == background.metadata


def test_background_load_missing_file_raises():
    with pytest.raises(SparrowException):
        NardiniBackground.load("/nonexistent/background.npz")


def test_background_zscores_from_mapping_and_array(idr_set):
    background = build_background(idr_set)
    values = composition_values(SMARCC2_LIKE)

    from_mapping = background.zscores(values)
    from_array = background.zscores(
        np.array([values[n] for n in composition_feature_names()])
    )

    assert np.allclose(from_mapping, from_array)
    assert from_mapping.shape == (54,)


def test_background_zscores_zero_variance_features_are_zero():
    """A feature that never varies in the background must not produce infinity."""
    # two identical sequences -> every std is exactly 0
    background = build_background(["MEEEKKKKSSSTTTDDD", "MEEEKKKKSSSTTTDDD"])
    assert np.all(background.std == 0)

    z = background.zscores(composition_values("QQQQNNNNSSSSGGGGP"))
    assert np.all(np.isfinite(z))
    assert np.all(z == 0)


def test_background_zscores_width_mismatch_raises(idr_set):
    background = build_background(idr_set)
    with pytest.raises(SparrowException):
        background.zscores(np.zeros(10))


def test_background_repr(idr_set):
    assert "54 features" in repr(build_background(idr_set))


# =====================================================================
# vectors
# =====================================================================


def test_nardini_vectors_full_shape(idr_set):
    background = build_background(idr_set)
    result = nardini_vectors(
        {"a": SMARCC2_LIKE, "b": RG_RICH},
        background=background,
        num_scrambles=200,
        seed=1,
    )

    assert len(result) == 2
    assert result.has_composition
    assert result.zscores.shape == (2, 90)
    assert len(result.feature_names) == 90
    assert result.patterning_zscores.shape == (2, 36)
    assert result.composition_zscores.shape == (2, 54)
    assert result.raw_patterning.shape == (2, 36)
    assert result.raw_composition.shape == (2, 54)


def test_nardini_vectors_patterning_only():
    result = nardini_vectors({"a": SMARCC2_LIKE}, num_scrambles=200, seed=1)

    assert not result.has_composition
    assert result.zscores.shape == (1, 36)
    assert len(result.feature_names) == 36


def test_nardini_vectors_from_fasta(idr_fasta, idr_set):
    result = nardini_vectors(idr_fasta, num_scrambles=100, seed=1)
    assert len(result) == len(idr_set)
    assert result.names[0] == "idr_0"


def test_nardini_vectors_accepts_background_path(tmp_path, idr_set):
    path = build_background(idr_set).save(tmp_path / "bg")
    result = nardini_vectors(
        {"a": SMARCC2_LIKE}, background=path, num_scrambles=100, seed=1
    )
    assert result.zscores.shape == (1, 90)


def test_nardini_vectors_dataframe(idr_set):
    background = build_background(idr_set)
    result = nardini_vectors(
        {"idr1": SMARCC2_LIKE, "idr2": RG_RICH},
        background=background,
        num_scrambles=100,
        seed=1,
    )

    frame = result.to_dataframe()
    assert frame.shape == (2, 90)
    assert list(frame.index) == ["idr1", "idr2"]
    assert frame.columns[0] == "pol-pol"
    assert frame.columns[-1] == "RG Frac"

    raw = result.to_dataframe(raw=True)
    assert raw.shape == (2, 90)
    # raw patch fractions are bounded by 0 and 1, unlike their z-scores
    assert 0.0 <= raw.loc["idr2", "RG Frac"] <= 1.0


def test_nardini_vectors_seed_is_order_independent(idr_set):
    """A sequence's z-scores must not depend on what else was in the batch.

    Every comparison below moves SMARCC2_LIKE to a different position. An earlier
    version of this test only ever checked it at index 0, which is exactly where
    the old ``seed + index`` derivation happened to agree - so the test passed
    while re-running one sequence out of a batch, or inserting a sequence ahead of
    it, silently changed its z-scores.
    """
    background = build_background(idr_set)
    kwargs = dict(background=background, num_scrambles=300, seed=7)

    alone = nardini_vectors({"a": SMARCC2_LIKE}, **kwargs)

    first = nardini_vectors({"a": SMARCC2_LIKE, "b": RG_RICH}, **kwargs)
    second = nardini_vectors({"b": RG_RICH, "a": SMARCC2_LIKE}, **kwargs)
    third = nardini_vectors({"x": BLOCKY_EK, "b": RG_RICH, "a": SMARCC2_LIKE}, **kwargs)

    for result in (first, second, third):
        index = result.names.index("a")
        assert np.allclose(result.zscores[index], alone.zscores[0]), (
            f"z-scores changed when the sequence sat at index {index}"
        )

    # the name is not part of the key either - the same sequence under a different
    # label gets the same scramble stream
    renamed = nardini_vectors({"something_else": SMARCC2_LIKE}, **kwargs)
    assert np.allclose(renamed.zscores[0], alone.zscores[0])


def test_nonstandard_residue_still_raises_a_clear_error():
    """Seed derivation must not pre-empt residue validation.

    The per-sequence seed is derived from the sequence text before the residues
    have been validated. Encoding it as ASCII made a non-standard character raise
    UnicodeEncodeError instead of sparrow's "no NARDINI type" message.
    """
    for sequence in ["MEEK\u00c5KSSSTTTQQQ", "MEEKXKSSSTTTQQQ"]:
        with pytest.raises(SparrowException, match="no NARDINI type"):
            nardini_vectors({"bad": sequence}, num_scrambles=50, seed=1)


def test_case_is_normalized_before_seeding(idr_set):
    """The same sequence in different case must give identical z-scores."""
    background = build_background(idr_set)
    kwargs = dict(background=background, num_scrambles=200, seed=4)

    upper = nardini_vectors({"a": SMARCC2_LIKE}, **kwargs)
    lower = nardini_vectors({"a": SMARCC2_LIKE.lower()}, **kwargs)

    assert np.allclose(upper.zscores, lower.zscores)


def test_protein_nardini_agrees_with_batch(idr_set):
    """Protein.nardini and nardini_vectors must not disagree on the same sequence."""
    from sparrow import Protein

    background = build_background(idr_set)

    batch = nardini_vectors(
        {"a": RG_RICH, "b": SMARCC2_LIKE},
        background=background,
        num_scrambles=300,
        seed=11,
    )
    solo = Protein(SMARCC2_LIKE).nardini.vector(
        background=background, num_scrambles=300, seed=11
    )

    assert np.allclose(solo.to_numpy(), batch.zscores[batch.names.index("b")])


def test_nardini_vectors_reproducible(idr_set):
    background = build_background(idr_set)
    kwargs = dict(background=background, num_scrambles=200, seed=3)

    a = nardini_vectors({"a": SMARCC2_LIKE}, **kwargs)
    b = nardini_vectors({"a": SMARCC2_LIKE}, **kwargs)

    assert np.array_equal(a.zscores, b.zscores)


def test_nardini_vectors_composition_block_is_seed_independent(idr_set):
    background = build_background(idr_set)
    a = nardini_vectors(
        {"a": SMARCC2_LIKE}, background=background, num_scrambles=200, seed=1
    )
    b = nardini_vectors(
        {"a": SMARCC2_LIKE}, background=background, num_scrambles=200, seed=2
    )

    assert np.allclose(a.composition_zscores, b.composition_zscores)
    assert not np.allclose(a.patterning_zscores, b.patterning_zscores)


def test_nardini_vectors_rejects_bare_sequence_string():
    with pytest.raises(SparrowException, match="bare sequence string"):
        nardini_vectors(SMARCC2_LIKE, num_scrambles=10)


def test_nardini_vectors_rejects_empty_input():
    with pytest.raises(SparrowException):
        nardini_vectors({}, num_scrambles=10)


def test_nardini_vectors_rejects_mismatched_background():
    # a background over a different feature set must be refused rather than
    # silently broadcasting against the wrong columns
    wrong = NardiniBackground(
        feature_names=["a", "b", "c"],
        mean=np.zeros(3),
        std=np.ones(3),
        num_sequences=10,
    )
    with pytest.raises(SparrowException):
        nardini_vectors({"a": SMARCC2_LIKE}, background=wrong, num_scrambles=10)


def test_top_features(idr_set):
    background = build_background(idr_set)
    result = nardini_vectors(
        {"rg": RG_RICH}, background=background, num_scrambles=200, seed=1
    )

    top = result.top_features("rg", n=5)
    assert len(top) == 5
    # sorted by descending |z|
    magnitudes = [abs(z) for _, z in top]
    assert magnitudes == sorted(magnitudes, reverse=True)
    # every name is a real feature
    for name, _ in top:
        assert name in result.feature_names


def test_top_features_by_index_and_unknown_name(idr_set):
    background = build_background(idr_set)
    result = nardini_vectors(
        {"rg": RG_RICH}, background=background, num_scrambles=100, seed=1
    )

    assert result.top_features(0, n=3) == result.top_features("rg", n=3)

    with pytest.raises(SparrowException):
        result.top_features("not-a-sequence")


def test_result_repr(idr_set):
    background = build_background(idr_set)
    result = nardini_vectors(
        {"a": SMARCC2_LIKE}, background=background, num_scrambles=100, seed=1
    )
    assert "90 features" in repr(result)


def test_rg_rich_sequence_is_rg_enriched(idr_set):
    """End-to-end smoke test: an RG-rich IDR should stand out on RG features."""
    background = build_background(idr_set)
    result = nardini_vectors(
        {"rg": RG_RICH, "q": "QQQQQQQQQQNNNNNNNNNNSSSSSSSSSS"},
        background=background,
        num_scrambles=200,
        seed=1,
    )

    # compare the raw values: whether the *z-scores* separate depends on the
    # background having any RG variance at all, which a synthetic set may not
    raw = result.to_dataframe(raw=True)
    assert raw.loc["rg", "RG Frac"] > 0.5
    assert raw.loc["q", "RG Frac"] == 0.0
    assert raw.loc["rg", "Frac G"] > raw.loc["q", "Frac G"]

    # Frac G does vary in the background, so its z-score should separate too
    z = result.to_dataframe()
    assert z.loc["rg", "Frac G"] > z.loc["q", "Frac G"]


# =====================================================================
# background quality warnings
# =====================================================================


def _diverse_idrs(count, seed=17):
    """`count` IDR-like sequences drawn from all 20 residues."""
    rng = random.Random(seed)
    alphabet = "SSSTTTGGGQQQNNNPPPEEEDDDKKKRRRAAALLLVVVMIYFWHC"
    return {
        f"idr_{i}": "".join(rng.choice(alphabet) for _ in range(rng.randint(60, 200)))
        for i in range(count)
    }


def test_build_background_warns_about_too_few_sequences():
    from sparrow.nardini.background import MIN_RECOMMENDED_SEQUENCES

    with pytest.warns(BackgroundQualityWarning, match="only 3 sequences"):
        build_background(_diverse_idrs(3))

    assert MIN_RECOMMENDED_SEQUENCES == 50


def test_build_background_warns_about_constant_features():
    """A single sequence makes every feature constant, so z-scores are all zero."""
    with pytest.warns(BackgroundQualityWarning, match="54 of 54 features are constant"):
        background = build_background({"only": SMARCC2_LIKE})

    # and the warning is telling the truth
    values = composition_values(RG_RICH)
    assert np.all(background.zscores(values) == 0.0)


def test_build_background_does_not_warn_for_an_adequate_set():
    """Enough diverse sequences must build quietly, or the warning is worthless."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", BackgroundQualityWarning)
        background = build_background(_diverse_idrs(400))

    assert background.num_sequences == 400


def test_published_backgrounds_do_not_warn():
    """The bundled IDRomes are the reference point the thresholds were set against."""
    from sparrow.nardini import REFERENCE_SPECIES, load_reference_background

    with warnings.catch_warnings():
        warnings.simplefilter("error", BackgroundQualityWarning)
        for species in REFERENCE_SPECIES:
            load_reference_background(species)


def test_background_records_degenerate_feature_count():
    with pytest.warns(BackgroundQualityWarning):
        small = build_background(_diverse_idrs(4))

    assert small.metadata["num_degenerate_features"] == int(np.sum(small.std == 0))
    assert small.metadata["num_degenerate_features"] > 0


def test_background_quality_warning_can_be_silenced_as_a_group():
    """Filtering SparrowWarning must catch it, so callers can opt out wholesale."""
    from sparrow.sparrow_exceptions import SparrowWarning

    assert issubclass(BackgroundQualityWarning, SparrowWarning)
    assert issubclass(SparrowWarning, UserWarning)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=SparrowWarning)
        build_background({"only": SMARCC2_LIKE})

    assert not caught
