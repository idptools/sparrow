"""The batch harness: every network sparrow ships, run over many sequences at once."""

import numpy as np
import pytest
import torch

from sparrow.predictors import NETWORK_PREDICTORS, load_network_predictor
from sparrow.predictors.batch_predictor import batch_predict, predict_many, prepare_model, run_network
from sparrow.sparrow_exceptions import SparrowException

rng = np.random.default_rng(5)
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
# short, either side of the 35-residue ALBATROSS threshold, and longer than the
# 168-residue mitochondrial-targeting window; the second sequence appears twice
LENGTHS = [12, 34, 60, 170, 300]
SEQUENCES = ["".join(rng.choice(AMINO_ACIDS, size=length)) for length in LENGTHS]
SEQUENCES.append(SEQUENCES[1])

BATCH_ALGORITHMS = ["pad-n-pack", "size-collect"]

# A prediction made in a batch differs from one made alone by the precision of
# the networks' 32-bit arithmetic: about 1e-7 relative for a single value, and up
# to one rounded place for the per-residue probabilities, which are rounded to 5
# decimal places
RELATIVE_TOLERANCE = 1e-6
ABSOLUTE_TOLERANCE = 2e-4

# (network, architecture), one of each kind
ONE_OF_EACH = [("rg", "MtO"), ("dssp", "MtM")]


def assert_close(batched, alone, label):
    """Assert a batched prediction equals one made alone, to the precision a batch keeps."""
    assert type(batched) is type(alone), label
    batched, alone = np.asarray(batched, dtype=float), np.asarray(alone, dtype=float)
    assert batched.shape == alone.shape, label
    if batched.ndim == 0:
        assert np.isclose(batched, alone, rtol=RELATIVE_TOLERANCE, atol=0), label
    else:
        scale = max(1.0, float(np.max(np.abs(alone))))
        assert np.allclose(batched, alone, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE * scale), label


@pytest.mark.parametrize("batch_algorithm", BATCH_ALGORITHMS)
@pytest.mark.parametrize(("network", "architecture"), ONE_OF_EACH)
def test_the_harness_returns_each_output_as_the_network_gives_it_alone(network, architecture, batch_algorithm):
    predictor = load_network_predictor(network)
    outputs = run_network(
        predictor.network,
        SEQUENCES,
        architecture=architecture,
        device=predictor.device,
        batch_size=4,
        batch_algorithm=batch_algorithm,
    )
    assert len(outputs) == len(SEQUENCES)
    for sequence, output in zip(SEQUENCES, outputs):
        alone = predictor._raw_predict(sequence)
        assert output.shape == alone.shape
        assert np.allclose(output, alone, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("batch_algorithm", BATCH_ALGORITHMS)
@pytest.mark.parametrize("network", NETWORK_PREDICTORS)
def test_every_network_predicts_many_as_it_predicts_one(network, batch_algorithm):
    predictor = load_network_predictor(network)
    batched = predictor.predict_many(SEQUENCES, batch_size=4, batch_algorithm=batch_algorithm)
    assert len(batched) == len(SEQUENCES)
    for sequence, prediction in zip(SEQUENCES, batched):
        assert_close(prediction, predictor.predict(sequence), (network, len(sequence)))


def test_a_sequence_given_twice_gets_the_same_prediction_twice():
    predictions = predict_many("nes", SEQUENCES)
    assert predictions[1] == predictions[-1]


@pytest.mark.parametrize("network", NETWORK_PREDICTORS)
def test_batch_predict_runs_every_network(network):
    results = batch_predict(SEQUENCES[:3], network=network, show_progress_bar=False, safe=False)
    expected = predict_many(network, SEQUENCES[:3])
    assert list(results) == [0, 1, 2]
    for index, (sequence, prediction) in results.items():
        assert sequence == SEQUENCES[index]
        assert_close(prediction, expected[index], (network, index))


def test_batch_predict_takes_the_version_as_a_number_or_a_string():
    as_number = batch_predict(SEQUENCES[2:4], network="rg", version=2, show_progress_bar=False)
    as_string = batch_predict(SEQUENCES[2:4], network="rg", version="2", show_progress_bar=False)
    assert as_number == as_string


def test_batch_predict_names_the_networks_when_given_another():
    with pytest.raises(SparrowException, match="'dssp'"):
        batch_predict(SEQUENCES, network="not_a_network")


def test_an_unknown_batch_algorithm_is_refused_before_anything_runs():
    with pytest.raises(SparrowException, match="batch_algorithm"):
        batch_predict(SEQUENCES, network="dssp", batch_algorithm="all-at-once")


def test_an_unknown_network_is_refused_by_name():
    with pytest.raises(SparrowException, match="is not one of sparrow's networks"):
        load_network_predictor("not_a_network")


@pytest.mark.parametrize("network", ["rg", "dssp"])
def test_a_loaded_network_is_in_evaluation_mode(network):
    assert load_network_predictor(network).network.training is False


def test_prepare_model_loads_any_network_onto_the_device():
    device, model = prepare_model("dssp", None, 0)
    assert isinstance(device, torch.device)
    assert type(model).__name__ == "BRNN_MtM"
