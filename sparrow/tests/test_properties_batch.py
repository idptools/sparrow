"""Every batch function in sparrow.properties behaves the same way.

Each takes a batch and returns an array with one entry (or row) per sequence, and
a sequence gets exactly the same answer - to the last bit - whether it is computed
alone or as part of a batch. That is what lets ``Protein`` call these functions
with a batch of one and still agree with ``sparrow.nardini``, which calls them
with thousands.

``test_every_batch_function_is_covered`` fails when a public function is added to
one of these modules without an entry in ``CALLS``, so nothing escapes the check.
"""

import inspect

import numpy as np
import pytest

from sparrow.data import amino_acids
from sparrow.properties import charge, composition, encoding, patches, patterning

BATCH = [
    "MEEEKKKKSSSTTTDDDQQQQNNNNGGG",
    "",
    "A",
    "ACDEFGHIKLMNPQRSTVWY" * 4,
    "RRRRRRRRRRRRRRRRRRRK",
    "DDDDDDDDDE",
    "GSGSGSGSGSGS",
    "KEDR" * 30,
    "mEEKkk",
]

#: Every public batch function, called with representative arguments.
CALLS = {
    "composition.amino_acid_fractions": composition.amino_acid_fractions,
    "composition.fraction_in_set": lambda batch: composition.fraction_in_set(batch, "QNS"),
    "composition.fraction_positive": composition.fraction_positive,
    "composition.fraction_negative": composition.fraction_negative,
    "composition.fraction_charged": composition.fraction_charged,
    "composition.net_charge_per_residue": composition.net_charge_per_residue,
    "composition.fraction_aromatic": composition.fraction_aromatic,
    "composition.fraction_aliphatic": composition.fraction_aliphatic,
    "composition.fraction_polar": composition.fraction_polar,
    "composition.fraction_proline": composition.fraction_proline,
    "composition.fraction_chain_expanding": composition.fraction_chain_expanding,
    "composition.fraction_disorder_promoting": composition.fraction_disorder_promoting,
    "composition.log_residue_ratio": lambda batch: composition.log_residue_ratio(batch, "R", "K"),
    "composition.mean_scale": lambda batch: composition.mean_scale(batch, amino_acids.AA_hydro_KD_normalized),
    "composition.hydrophobicity": composition.hydrophobicity,
    "composition.polyproline_II_propensity": composition.polyproline_II_propensity,
    "composition.molecular_weight": composition.molecular_weight,
    "composition.seg_complexity": composition.seg_complexity,
    "charge.charge_at_pH": lambda batch: charge.charge_at_pH(batch, pH=5.0, normalize=False),
    "charge.isoelectric_point": charge.isoelectric_point,
    "patches.patch_fraction": lambda batch: patches.patch_fraction(batch, "EK", interruption=1, min_target_count=3),
    "patches.rg_patch_fraction": lambda batch: patches.rg_patch_fraction(batch, min_adjacent_rg_pairs=1),
    "patterning.sigma": lambda batch: patterning.sigma(batch, ["E", "D"], ["R", "K"]),
    "patterning.delta": lambda batch: patterning.delta(batch, ["E", "D"], ["R", "K"], 5),
    "patterning.kappa_x": lambda batch: patterning.kappa_x(batch, ["S", "G"], [], 4, flatten=False),
    "patterning.kappa": patterning.kappa,
    "patterning.binary_asymmetry": lambda batch: patterning.binary_asymmetry(batch, ["E", "D"]),
    "patterning.patterning_percentile": lambda batch: patterning.patterning_percentile(batch, ["E", "K"], 5, 50, 3),
}

#: Functions that reject sequences shorter than their window, and so are given
#: only the sequences that are long enough.
NEEDS_LONG_SEQUENCES = {"patterning.patterning_percentile"}
LONG_ENOUGH = 5

#: Functions that depend only on composition, and so also accept ResidueCounts.
TAKES_COUNTS = {name for name in CALLS if name.split(".")[0] in ("composition", "charge")} | {
    "patterning.sigma",
    "patterning.binary_asymmetry",
}

#: Public names that are not batch property functions.
NOT_BATCH_FUNCTIONS = {
    "composition.ResidueCounts",
    "composition.count_residues",
    "composition.as_counts",
    # these work on arrays of counts or arrangements, not on sequences
    "patterning.asymmetry",
    "patterning.window_delta",
    "patterning.ternary_delta",
    "patterning.segregated_arrangements",
    "patterning.kappa_from_ternary",
}

MODULES = {"composition": composition, "charge": charge, "patches": patches, "patterning": patterning}


def batch_for(name):
    """The test batch a function is given."""
    if name in NEEDS_LONG_SEQUENCES:
        return [sequence for sequence in BATCH if len(sequence) >= LONG_ENOUGH]
    return BATCH


def test_every_batch_function_is_covered():
    for short_name, module in MODULES.items():
        for name in module.__all__:
            if not inspect.isfunction(getattr(module, name)) and not inspect.isclass(getattr(module, name)):
                continue
            key = f"{short_name}.{name}"
            assert key in CALLS or key in NOT_BATCH_FUNCTIONS, f"{key} is not covered by the batch test"


@pytest.mark.parametrize("name", sorted(CALLS))
def test_batch_returns_one_entry_per_sequence(name):
    batch = batch_for(name)
    result = CALLS[name](batch)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape[0] == len(batch)


@pytest.mark.parametrize("name", sorted(CALLS))
def test_batch_matches_one_sequence_at_a_time(name):
    sequences = batch_for(name)
    batch = CALLS[name](sequences)
    one_at_a_time = np.stack([CALLS[name]([sequence])[0] for sequence in sequences])
    # assert_array_equal treats NaN as equal to NaN, which the empty sequence needs
    np.testing.assert_array_equal(batch, one_at_a_time)


@pytest.mark.parametrize("name", sorted(CALLS))
def test_a_large_batch_matches_one_sequence_at_a_time(name):
    """Large enough that numpy iterates, buffers and reduces it differently."""
    sequences = batch_for(name)
    repeats = 20 if name in NEEDS_LONG_SEQUENCES else 300
    batch = CALLS[name](sequences * repeats)
    one_at_a_time = np.stack([CALLS[name]([sequence])[0] for sequence in sequences])
    np.testing.assert_array_equal(batch, np.concatenate([one_at_a_time] * repeats))


@pytest.mark.parametrize("name", sorted(TAKES_COUNTS))
def test_counts_and_sequences_give_the_same_answer(name):
    counts = composition.count_residues(BATCH)
    np.testing.assert_array_equal(CALLS[name](counts), CALLS[name](BATCH))


@pytest.mark.parametrize("name", sorted(CALLS))
def test_an_encoded_batch_gives_the_same_answer(name):
    sequences = batch_for(name)
    np.testing.assert_array_equal(CALLS[name](encoding.encode_batch(sequences)), CALLS[name](sequences))


@pytest.mark.parametrize("name", sorted(CALLS))
def test_a_bare_string_is_a_batch_of_one(name):
    np.testing.assert_array_equal(CALLS[name](BATCH[0]), CALLS[name]([BATCH[0]]))
