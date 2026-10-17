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
from sparrow.properties import charge, composition

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
}

#: Public names that are not batch property functions.
NOT_BATCH_FUNCTIONS = {
    "composition.ResidueCounts",
    "composition.count_residues",
    "composition.as_counts",
}

MODULES = {"composition": composition, "charge": charge}


def test_every_batch_function_is_covered():
    for short_name, module in MODULES.items():
        for name in module.__all__:
            if not inspect.isfunction(getattr(module, name)) and not inspect.isclass(getattr(module, name)):
                continue
            key = f"{short_name}.{name}"
            assert key in CALLS or key in NOT_BATCH_FUNCTIONS, f"{key} is not covered by the batch test"


@pytest.mark.parametrize("name", sorted(CALLS))
def test_batch_returns_one_entry_per_sequence(name):
    result = CALLS[name](BATCH)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    assert result.shape[0] == len(BATCH)


@pytest.mark.parametrize("name", sorted(CALLS))
def test_batch_matches_one_sequence_at_a_time(name):
    batch = CALLS[name](BATCH)
    one_at_a_time = np.stack([CALLS[name]([sequence])[0] for sequence in BATCH])
    # assert_array_equal treats NaN as equal to NaN, which the empty sequence needs
    np.testing.assert_array_equal(batch, one_at_a_time)


@pytest.mark.parametrize("name", sorted(CALLS))
def test_a_large_batch_matches_one_sequence_at_a_time(name):
    """Large enough that numpy iterates, buffers and reduces it differently."""
    large = BATCH * 300
    batch = CALLS[name](large)
    one_at_a_time = np.stack([CALLS[name]([sequence])[0] for sequence in BATCH])
    np.testing.assert_array_equal(batch, np.concatenate([one_at_a_time] * 300))


@pytest.mark.parametrize("name", sorted(CALLS))
def test_counts_and_sequences_give_the_same_answer(name):
    counts = composition.count_residues(BATCH)
    np.testing.assert_array_equal(CALLS[name](counts), CALLS[name](BATCH))


@pytest.mark.parametrize("name", sorted(CALLS))
def test_a_bare_string_is_a_batch_of_one(name):
    np.testing.assert_array_equal(CALLS[name](BATCH[0]), CALLS[name]([BATCH[0]]))
