"""Tests for sparrow.null, the null models.

Every null model must give nulls with exactly the length and composition of
the sequence, reproducibly from a seed, so those are checked for all of them.
Then each is checked for what it keeps: the block shuffle keeps its blocks,
and every pair of neighbours equally often; the k-let shuffle keeps every
k-let, and makes every sequence with them equally often; and the background
models carry the background's grammar into their nulls, reduce to a shuffle
when the background has none, and place each residue as often at the end of a
null as at its start.
"""

import itertools
import os
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import protfasta
import pytest
from scipy.stats import chisquare

from sparrow import Protein
from sparrow.null import BlockShuffle, KletShuffle, KmerBackground, MarkovBackground, NullModel
from sparrow.null import kmer_background, kmers, markov_background, tilt
from sparrow.null.blocks import check_block_sizes, plan_blocks
from sparrow.properties import encoding
from sparrow.sparrow_exceptions import NullModelWarning, SparrowException

FASTA = os.path.join(os.path.dirname(__file__), "test_data", "test_seqs_100.fasta")

# a sequence with every residue, some of them only once, and runs of Q and N
SEQUENCE = "MSDQQQQQNNNNSTPKKEEDDRRGGAAVVLLIIFFYYWHCMQNQNSSGG"

# the 20 residues once each, so every residue of a null says where it came from
EVERY_RESIDUE_ONCE = "".join(encoding.RESIDUES)


def _random_background(number: int, length: int, residues: str, seed: int) -> list[str]:
    """Sequences of residues drawn independently and evenly, so with no grammar at all."""
    rng = np.random.default_rng(seed)
    letters = np.array(list(residues))
    return ["".join(rng.choice(letters, size=length)) for _ in range(number)]


def _unit_background(number: int, units: int, seed: int) -> list[str]:
    """Sequences of the units 'A' and 'KE', so K is only ever followed by E."""
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(["A", "KE"], size=units)) for _ in range(number)]


def _poly_q_background(number: int, seed: int) -> list[str]:
    """Sequences of A, S, G and P with runs of six Qs, so Q is much more likely after Q."""
    rng = np.random.default_rng(seed)
    return [
        "".join(rng.choice(["A", "S", "G", "P", "QQQQQQ"], size=60, p=[0.3, 0.3, 0.2, 0.1, 0.1]))
        for _ in range(number)
    ]


@pytest.fixture(scope="module")
def background() -> list[str]:
    """A background of 300,000 residues with no grammar, enough for every 3-mer about 40 times."""
    return _random_background(1000, 300, "".join(encoding.RESIDUES), seed=1)


@pytest.fixture(scope="module")
def models(background: list[str]) -> dict[str, NullModel]:
    """One of each null model, with the background models fit."""
    return {
        "block shuffle": BlockShuffle(k=1),
        "block shuffle of 2 and 3": BlockShuffle(k=[2, 3]),
        "k-let shuffle": KletShuffle(k=2),
        "k-let shuffle of 3": KletShuffle(k=3),
        "background k-mers": KmerBackground(k=3).fit(background),
        "background k-mers of 2 and 3": KmerBackground(k=[2, 3]).fit(background),
        "background Markov chain": MarkovBackground(order=2).fit(background),
        "background Markov chain of order 0": MarkovBackground(order=0).fit(background),
    }


MODEL_NAMES = [
    "block shuffle",
    "block shuffle of 2 and 3",
    "k-let shuffle",
    "k-let shuffle of 3",
    "background k-mers",
    "background k-mers of 2 and 3",
    "background Markov chain",
    "background Markov chain of order 0",
]


# ............................................................................
# every null model
# ............................................................................


@pytest.mark.parametrize("name", MODEL_NAMES)
@pytest.mark.parametrize("sequence", [SEQUENCE, EVERY_RESIDUE_ONCE, "QQQQQQQQQQNQ", SEQUENCE * 5])
def test_every_null_has_the_length_and_composition_of_its_sequence(models, name, sequence):
    nulls = models[name].generate(sequence, count=20, seed=1)

    assert len(nulls) == 20
    for null in nulls:
        assert isinstance(null, str)
        assert Counter(null) == Counter(sequence)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_a_seed_gives_the_same_nulls_every_time(models, name):
    first = models[name].generate(SEQUENCE, count=5, seed=11)
    again = models[name].generate(SEQUENCE, count=5, seed=11)
    other = models[name].generate(SEQUENCE, count=5, seed=12)

    assert first == again
    assert first != other


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_without_a_seed_the_nulls_differ_each_time(models, name):
    assert models[name].generate(SEQUENCE, count=5) != models[name].generate(SEQUENCE, count=5)


def test_one_seed_gives_two_sequences_of_the_same_length_different_shuffles():
    # with every residue once, each null gives the order the residues were
    # shuffled into, which a shared random stream would make the same
    reversed_sequence = EVERY_RESIDUE_ONCE[::-1]
    shuffle = BlockShuffle(k=1)
    null = shuffle.generate(EVERY_RESIDUE_ONCE, seed=3)[0]
    reversed_null = shuffle.generate(reversed_sequence, seed=3)[0]

    order = [EVERY_RESIDUE_ONCE.index(residue) for residue in null]
    reversed_order = [reversed_sequence.index(residue) for residue in reversed_null]
    assert order != reversed_order


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_a_protein_or_lower_case_sequence_gives_the_same_nulls(models, name):
    expected = models[name].generate(SEQUENCE, count=3, seed=2)

    assert models[name].generate(Protein(SEQUENCE), count=3, seed=2) == expected
    assert models[name].generate(SEQUENCE.lower(), count=3, seed=2) == expected


@pytest.mark.parametrize("sequence", ["", "MKXK", "MK K", 12])
def test_a_sequence_that_is_empty_or_not_standard_residues_is_refused(sequence):
    with pytest.raises(SparrowException):
        BlockShuffle().generate(sequence)


@pytest.mark.parametrize("count", [0, -1, 1.5, True, "2"])
def test_a_count_that_is_not_a_positive_integer_is_refused(count):
    with pytest.raises(SparrowException, match="count"):
        BlockShuffle().generate(SEQUENCE, count=count)


@pytest.mark.parametrize("seed", [-1, 1.5, True, "1"])
def test_a_seed_that_is_not_a_non_negative_integer_is_refused(seed):
    with pytest.raises(SparrowException, match="seed"):
        BlockShuffle().generate(SEQUENCE, seed=seed)


@pytest.mark.parametrize("model", [BlockShuffle(), KletShuffle()])
def test_a_shuffle_needs_no_background_and_refuses_one(model, background):
    assert not model.needs_background
    assert not model.is_fitted
    model.generate(SEQUENCE)

    with pytest.raises(SparrowException, match="takes no background"):
        model.fit(background)


@pytest.mark.parametrize("model_class", [KmerBackground, MarkovBackground])
def test_a_background_model_must_be_fit_before_it_makes_nulls(model_class, background):
    model = model_class()
    assert model.needs_background
    assert not model.is_fitted

    with pytest.raises(SparrowException, match="fit"):
        model.generate(SEQUENCE)

    assert model.fit(background) is model
    assert model.is_fitted
    model.generate(SEQUENCE)


def test_each_model_says_what_it_is():
    assert repr(BlockShuffle(k=2)) == "BlockShuffle(k=2)"
    assert repr(BlockShuffle(k=[3, 2])) == "BlockShuffle(k=[2, 3])"
    assert repr(KletShuffle(k=3)) == "KletShuffle(k=3)"
    assert repr(KmerBackground(k=4)) == "KmerBackground(k=4)"
    assert repr(MarkovBackground(order=1)) == "MarkovBackground(order=1)"


# ............................................................................
# backgrounds
# ............................................................................


@pytest.mark.parametrize("model_class", [KmerBackground, MarkovBackground])
def test_a_background_can_be_a_file_a_dict_a_list_or_proteins(model_class):
    with warnings.catch_warnings():
        # the test file is far too small for 3-mers, which the model says
        warnings.simplefilter("ignore", NullModelWarning)
        records = protfasta.read_fasta(FASTA)
        expected = model_class().fit(FASTA).generate(SEQUENCE, count=3, seed=1)

        for background in [
            Path(FASTA),
            records,
            list(records.values()),
            [Protein(sequence) for sequence in records.values()],
            {name: Protein(sequence) for name, sequence in records.items()},
        ]:
            assert model_class().fit(background).generate(SEQUENCE, count=3, seed=1) == expected


@pytest.mark.parametrize(
    "background, message",
    [("not_a_file.fasta", "not a file"), ([], "no sequences"), (["MKV", 5], "item 1 is a int")],
)
def test_a_background_that_is_not_a_file_empty_or_not_sequences_is_refused(background, message):
    with pytest.raises(SparrowException, match=message):
        KmerBackground().fit(background)


def test_a_background_too_short_for_its_kmers_is_refused():
    with pytest.raises(SparrowException, match="no run of 3"):
        KmerBackground(k=3).fit(["MK", "KV"])


@pytest.mark.filterwarnings("ignore:The background has .* an average of")
def test_residues_that_are_not_standard_are_left_out_of_the_counts_and_counted_in_a_warning():
    with pytest.warns(NullModelWarning, match=r"3 residues .* not one of the 20 standard amino acids \(U: 1, X: 2\)"):
        counts = kmers.count_background_kmers(["AAXAA", "CXU"], 2, "test")

    assert counts[1].sum() == 5
    # AA, AA: none across the X, nor across the join between the two sequences
    assert counts[2].sum() == 2
    assert counts[2][encoding.RESIDUE_INDEX["A"] * 20 + encoding.RESIDUE_INDEX["A"]] == 2


@pytest.mark.filterwarnings("ignore:The background has .* an average of")
def test_kmers_are_not_counted_across_the_join_between_sequences():
    counts = kmers.count_background_kmers(["AC", "DE"], 2, "test")

    seen = {
        "".join(encoding.RESIDUES[code] for code in kmers.kmer_residues(np.array([index]), 2)[0])
        for index in np.flatnonzero(counts[2])
    }
    assert seen == {"AC", "DE"}


def test_a_background_too_small_for_its_kmers_is_warned_about():
    small = _random_background(10, 100, "".join(encoding.RESIDUES), seed=1)

    with pytest.warns(NullModelWarning, match="average of"):
        KmerBackground(k=3).fit(small)
    with warnings.catch_warnings():
        warnings.simplefilter("error", NullModelWarning)
        KmerBackground(k=1).fit(small)


def test_a_kmer_index_reads_back_as_its_residues():
    codes = np.array([[3, 0, 19], [19, 19, 1]])
    indices = codes[:, 0] * 400 + codes[:, 1] * 20 + codes[:, 2]

    assert np.array_equal(kmers.kmer_residues(indices, 3), codes)


@pytest.mark.parametrize("model_class", [KmerBackground, MarkovBackground])
def test_a_residue_the_background_does_not_have_is_refused(model_class):
    model = model_class().fit(_random_background(500, 200, "ACDEFGHIKLMNPQRSTVY", seed=1))

    with pytest.raises(SparrowException, match="no W"):
        model.generate(SEQUENCE)


# ............................................................................
# blocks
# ............................................................................


@pytest.mark.parametrize("sizes", [(1,), (2,), (3,), (2, 3), (2, 4), (1, 2, 3, 4), (3, 5)])
@pytest.mark.parametrize("length", [1, 2, 3, 5, 7, 20, 99, 100, 101, 1000])
def test_a_plan_of_blocks_adds_up_to_the_length(sizes, length):
    plan = plan_blocks(length, sizes)

    assert sum(plan) == length
    # at most one block is not one of the sizes, and it is shorter than all of them
    others = [size for size in plan if size not in sizes]
    assert len(others) <= 1
    assert all(size < min(sizes) for size in others)


def test_one_block_size_cuts_whole_blocks_and_one_shorter():
    assert Counter(plan_blocks(100, (3,))) == {3: 33, 1: 1}
    assert Counter(plan_blocks(99, (3,))) == {3: 33}


def test_several_block_sizes_share_the_residues_evenly_with_none_left_over():
    assert Counter(plan_blocks(100, (2, 3))) == {2: 26, 3: 16}
    assert Counter(plan_blocks(101, (2, 3))) == {2: 25, 3: 17}
    assert Counter(plan_blocks(100, (2, 3, 4))) == {2: 16, 3: 12, 4: 8}


def test_block_sizes_with_no_mix_adding_up_to_the_length_leave_one_shorter_block():
    assert Counter(plan_blocks(101, (2, 4))) == {2: 26, 4: 12, 1: 1}


@pytest.mark.parametrize("k", [0, -2, True, 1.5, [], [2, 2], [2, 0]])
def test_a_block_size_that_is_not_a_positive_integer_or_is_repeated_is_refused(k):
    with pytest.raises(SparrowException):
        check_block_sizes(k)


def test_block_sizes_above_the_limit_are_refused():
    assert check_block_sizes([4, 2], maximum=4) == (2, 4)
    with pytest.raises(SparrowException, match="at most 4"):
        KmerBackground(k=5)


# ............................................................................
# the block shuffle
# ............................................................................


def _kept_together(nulls: list[str], sequence: str) -> np.ndarray:
    """For each pair of neighbours in a sequence of distinct residues, the fraction of nulls that keep them, in order."""
    kept = np.zeros(len(sequence) - 1)
    for null in nulls:
        for position in range(len(sequence) - 1):
            kept[position] += sequence[position : position + 2] in null
    return kept / len(nulls)


@pytest.mark.parametrize("k, expected", [(2, 1 / 2), (4, 3 / 4)])
def test_every_pair_of_neighbours_is_kept_together_as_often_as_any_other(k, expected):
    # with fixed cuts, the pairs that straddle a cut would never be kept
    nulls = BlockShuffle(k=k).generate(EVERY_RESIDUE_ONCE, count=4000, seed=1)
    kept = _kept_together(nulls, EVERY_RESIDUE_ONCE)

    # a pair that is cut apart can still end up side by side by chance, about
    # once in the number of blocks
    assert np.all(np.abs(kept - expected) < 0.1)
    assert kept.max() - kept.min() < 0.1


def test_a_plain_shuffle_keeps_pairs_together_only_by_chance():
    nulls = BlockShuffle(k=1).generate(EVERY_RESIDUE_ONCE, count=4000, seed=1)

    assert np.all(_kept_together(nulls, EVERY_RESIDUE_ONCE) < 0.1)


def test_a_block_shuffle_keeps_its_blocks_whole():
    # with k=5 there are 4 or 5 blocks, so the null is the sequence cut in at most 4 places
    for null in BlockShuffle(k=5).generate(EVERY_RESIDUE_ONCE, count=200, seed=1):
        order = [EVERY_RESIDUE_ONCE.index(residue) for residue in null]
        cuts = sum(1 for before, after in zip(order, order[1:]) if after != before + 1)
        assert cuts <= 4


def test_a_sequence_that_is_one_block_cannot_be_shuffled_in_blocks():
    with pytest.raises(SparrowException, match="single block"):
        BlockShuffle(k=5).generate("MKVLA")


# ............................................................................
# the k-let shuffle
# ............................................................................


def _klets(sequence: str, k: int) -> Counter:
    return Counter(sequence[start : start + k] for start in range(len(sequence) - k + 1))


@pytest.mark.parametrize("k", [2, 3, 4])
def test_a_klet_shuffle_keeps_every_klet_and_both_ends(k):
    sequence = SEQUENCE * 3
    for null in KletShuffle(k=k).generate(sequence, count=50, seed=1):
        assert _klets(null, k) == _klets(sequence, k)
        assert null[: k - 1] == sequence[: k - 1]
        assert null[len(null) - k + 1 :] == sequence[len(sequence) - k + 1 :]


@pytest.mark.parametrize("sequence", ["ACAACCACAC", "AACCACCAAC"])
def test_a_klet_shuffle_makes_every_sequence_with_the_same_klets_equally_often(sequence):
    # every arrangement of its As and Cs with the same pairs and ends
    arrangements = []
    for places in itertools.combinations(range(len(sequence)), sequence.count("A")):
        arrangements.append("".join("A" if position in places else "C" for position in range(len(sequence))))
    possible = {
        arrangement
        for arrangement in arrangements
        if _klets(arrangement, 2) == _klets(sequence, 2)
        and arrangement[0] == sequence[0]
        and arrangement[-1] == sequence[-1]
    }
    nulls = KletShuffle(k=2).generate(sequence, count=20000, seed=1)
    made = Counter(nulls)

    assert set(made) == possible
    # a uniform sampler gives a p-value anywhere from 0 to 1; a biased one, a tiny one
    assert chisquare([made[arrangement] for arrangement in sorted(possible)]).pvalue > 1e-3


def test_a_klet_shuffle_with_only_one_arrangement_gives_the_sequence_back():
    assert set(KletShuffle(k=2).generate("MKMKM", count=20, seed=1)) == {"MKMKM"}


def test_a_klet_shuffle_needs_k_shorter_than_the_sequence():
    with pytest.raises(SparrowException, match="shorter than the sequence"):
        KletShuffle(k=5).generate("MKVLA")


@pytest.mark.parametrize("k", [0, True, 2.5])
def test_a_klet_length_that_is_not_a_positive_integer_is_refused(k):
    with pytest.raises(SparrowException):
        KletShuffle(k=k)


# ............................................................................
# the background models
# ............................................................................


def _fraction_of_k_followed_by_e(nulls: list[str]) -> float:
    pairs = Counter(pair for null in nulls for pair in zip(null, null[1:]) if pair[0] == "K")
    return pairs[("K", "E")] / sum(pairs.values())


@pytest.mark.filterwarnings("ignore:In .* nulls, .* blocks were drawn as smaller ones")
def test_background_kmers_carry_the_backgrounds_grammar_into_their_blocks():
    model = KmerBackground(k=2).fit(_unit_background(400, 100, seed=1))
    sequence = "KKKKKKKKKKEEEEEEEEEEAAAAAAAAAA" * 2

    # inside a block K is always followed by E; across the join between two
    # blocks, only as often as chance makes it
    with_grammar = _fraction_of_k_followed_by_e(model.generate(sequence, count=200, seed=1))
    shuffled = _fraction_of_k_followed_by_e(BlockShuffle(k=1).generate(sequence, count=200, seed=1))

    assert shuffled < 0.45
    assert with_grammar > 0.6


def test_a_background_markov_chain_carries_the_backgrounds_grammar_into_every_residue():
    model = MarkovBackground(order=1).fit(_unit_background(400, 100, seed=1))
    sequence = "KKKKKKKKKKEEEEEEEEEEAAAAAAAAAA" * 2

    with warnings.catch_warnings():
        # a null whose last residue is K has an E with nowhere to go, which is warned about
        warnings.simplefilter("ignore", NullModelWarning)
        nulls = model.generate(sequence, count=200, seed=1)

    assert _fraction_of_k_followed_by_e(nulls) > 0.95


def test_background_kmers_with_no_grammar_are_drawn_as_a_shuffle_would_draw_them():
    # a background where every 2-mer is exactly as frequent as its residues make it
    frequencies = np.linspace(1, 2, 20) / np.linspace(1, 2, 20).sum()
    counts = np.round(np.outer(frequencies, frequencies).ravel() * 1e8).astype(np.int64)
    composition = np.bincount(encoding.residue_codes(SEQUENCE), minlength=20)

    groups = kmer_background._KmerGroups.from_counts(counts, 2).made_of(composition > 0)
    weights = kmer_background._tilted_weights(groups, composition, frequencies)

    # a shuffle draws a pair of different residues twice as often as one residue twice
    orderings = groups.stop - groups.first
    assert np.allclose(weights / orderings, (weights / orderings).max(), rtol=1e-3)


def test_a_background_markov_chain_with_no_grammar_draws_every_residue_alike():
    frequencies = np.linspace(1, 2, 20) / np.linspace(1, 2, 20).sum()
    transitions = np.tile(frequencies, (20, 1))
    composition = np.bincount(encoding.residue_codes(SEQUENCE), minlength=20)

    chain, runs, stationary = markov_background._conditioned_chain(transitions, 1, composition, frequencies)

    # every residue of the sequence after every run has the same weight, so it
    # is drawn from what is left alone, as in a shuffle
    present = chain[np.ix_(runs, np.flatnonzero(composition))]
    assert np.allclose(present, present.max(), rtol=1e-3)
    # and each residue is where the chain spends its time in its proportion of the sequence
    assert np.allclose(stationary, (composition / composition.sum())[np.flatnonzero(composition)], rtol=1e-3)


def _quarters(nulls: list[str], residue: str) -> np.ndarray:
    """The fraction of one residue in each quarter of the nulls, over all of them."""
    as_array = np.array([list(null) for null in nulls])
    edges = np.linspace(0, as_array.shape[1], 5).astype(int)
    return np.array([np.mean(as_array[:, start:stop] == residue) for start, stop in zip(edges[:-1], edges[1:])])


@pytest.mark.parametrize("model", [MarkovBackground(order=2), KmerBackground(k=3)])
@pytest.mark.filterwarnings("ignore:In .* nulls, .* residues were drawn given fewer")
def test_a_residue_the_background_runs_together_is_as_common_at_the_end_of_a_null_as_at_its_start(model):
    # Q comes in runs in the background, so a chain that does not look ahead
    # uses it up early in long runs, and runs out of it at the end
    model.fit(_poly_q_background(2000, seed=1))
    sequence = "QQQQQAASSGQQPAQQQQQSGASQQQQQPGAS" * 4
    nulls = model.generate(sequence, count=400, seed=1)

    quarters = _quarters(nulls, "Q")
    assert np.all(np.abs(quarters - sequence.count("Q") / len(sequence)) < 0.03)


@pytest.mark.filterwarnings("ignore:The background has .* an average of")
def test_a_markov_chain_conditioned_on_composition_has_that_composition():
    background = _poly_q_background(500, seed=2)
    counts = kmers.count_background_kmers(background, 3, "test")
    frequencies = counts[1] / counts[1].sum()
    composition = np.bincount(encoding.residue_codes("QQQQQAASSGQQPAQQQQQSGASQQQQQPGAS"), minlength=20)
    transitions = markov_background._transitions_after(counts[3], 2)

    chain, runs, stationary = markov_background._conditioned_chain(transitions, 2, composition, frequencies)

    # the runs the chain spends its time in hold each residue in the sequence's proportion
    residues = kmers.kmer_residues(runs, 2)
    held = np.bincount(residues[:, 1], weights=stationary, minlength=20)
    assert np.allclose(
        held[composition > 0], (composition / composition.sum())[composition > 0], rtol=2 * tilt.TILT_TOLERANCE
    )


def test_background_kmers_tilted_to_a_composition_have_that_composition_on_average():
    counts = kmers.count_background_kmers(_poly_q_background(2000, seed=2), 3, "test")
    frequencies = counts[1] / counts[1].sum()
    composition = np.bincount(encoding.residue_codes("QQQQQAASSGQQPAQQQQQSGASQQQQQPGAS"), minlength=20)
    fractions = composition / composition.sum()

    groups = kmer_background._KmerGroups.from_counts(counts[3], 3).made_of(composition > 0)
    weights = kmer_background._tilted_weights(groups, composition, frequencies)

    # the tilted frequencies are the weights times each group's chance under the composition
    tilted = weights * np.prod(fractions[groups.residues], axis=1)
    held = np.zeros(20)
    for position in range(3):
        held += np.bincount(groups.residues[:, position], weights=tilted, minlength=20)
    held /= held.sum()
    assert np.allclose(held[composition > 0], fractions[composition > 0], rtol=2 * tilt.TILT_TOLERANCE)


@pytest.mark.filterwarnings("ignore:The background has .* an average of")
@pytest.mark.filterwarnings("ignore:The background could only be tilted")
def test_a_block_with_no_kmer_in_the_background_is_drawn_as_smaller_ones_with_a_warning():
    # the background has only AC and CA, so once only As are left, AA cannot be drawn
    model = KmerBackground(k=2).fit(["AC" * 50] * 20)

    with pytest.warns(NullModelWarning, match="drawn as smaller ones"):
        nulls = model.generate("AAAAAC", count=10, seed=1)
    assert all(Counter(null) == Counter("AAAAAC") for null in nulls)


@pytest.mark.filterwarnings("ignore:The background has .* an average of")
@pytest.mark.filterwarnings("ignore:The background could only be tilted")
def test_a_residue_with_no_run_before_it_in_the_background_is_drawn_from_a_shorter_one_with_a_warning():
    model = MarkovBackground(order=1).fit(["AC" * 50] * 20)

    with pytest.warns(NullModelWarning, match="fewer of the residues before them"):
        nulls = model.generate("AAAAAC", count=10, seed=1)
    assert all(Counter(null) == Counter("AAAAAC") for null in nulls)


def test_a_sequence_shorter_than_the_markov_order_is_refused(background):
    model = MarkovBackground(order=2).fit(background)

    assert all(Counter(null) == Counter("MK") for null in model.generate("MK", count=5, seed=1))
    with pytest.raises(SparrowException, match="shorter than the 2 residues"):
        model.generate("M")


@pytest.mark.parametrize("order", [-1, 4, True, 1.5])
def test_a_markov_order_outside_its_range_is_refused(order):
    with pytest.raises(SparrowException, match="order"):
        MarkovBackground(order=order)


# ............................................................................
# the pieces
# ............................................................................


def test_a_weighted_choice_never_picks_an_index_with_no_weight():
    weights = np.array([0.0, 0.0, 1.0, 0.0, 2.0, 0.0])

    assert kmers.weighted_choice(weights, 0.0) == 2
    assert kmers.weighted_choice(weights, 0.5) == 4
    assert kmers.weighted_choice(weights, 1 - 1e-17) == 4


def test_each_row_is_drawn_only_from_its_columns_with_weight():
    weights = np.array([[0.0, 1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    for uniform in [0.0, 0.5, 1 - 1e-17]:
        picked = markov_background._draw_each_row(weights, np.full(3, uniform))
        assert list(picked) == [1, 0, 2]


def test_a_tilt_that_cannot_reach_the_composition_is_warned_about():
    # a background whose composition does not respond to any tilt
    def stuck(log_tilt):
        return 0.0, np.array([0.5, 0.5])

    with pytest.warns(NullModelWarning, match="could only be tilted"):
        tilt.fit_tilt(stuck, np.array([0.9, 0.1]), np.zeros(2))
