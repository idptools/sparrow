"""sparrow's networks run on every device, the CPU, an Apple-silicon GPU (MPS) and an
NVIDIA GPU (CUDA), and predict there as they do on the CPU, up to float32 rounding.

The tests that need a GPU run on each GPU this machine has, and are skipped, with
the reason, on the others (see device_markers.py), so the whole set runs on a
machine with an NVIDIA GPU, and the CPU and MPS ones on an Apple-silicon Mac. How a
choice of device is resolved, and what happens when a device is not on the machine,
is tested on any machine, by making CUDA and MPS look available or not.
"""

import functools
import random
from typing import Any

import numpy as np
import pytest
import torch

import sparrow
import sparrow.predictors.batch_predictor
from sparrow import Protein
from sparrow.predictors import NETWORK_PREDICTORS, load_network_predictor
from sparrow.predictors.batch_predictor import batch_predict, prepare_model, run_network
from sparrow.predictors.devices import full_float32_precision, resolve_device
from sparrow.sparrow_exceptions import SparrowException
from sparrow.tests.device_markers import (
    DEVICE_ABSOLUTE_TOLERANCE,
    DEVICE_RELATIVE_TOLERANCE,
    DEVICES,
    requires_two_gpus,
)
from sparrow.tests.parrot2_markers import network_param

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# random sequences of 5 to 300 residues, the same on every run
SEQUENCE_SEED = 11
NUM_SEQUENCES = 12
SHORTEST, LONGEST = 5, 300
_rng = random.Random(SEQUENCE_SEED)
SEQUENCES = [
    "".join(_rng.choice(AMINO_ACIDS) for _ in range(_rng.randint(SHORTEST, LONGEST))) for _ in range(NUM_SEQUENCES)
]

# The networks whose prediction is a class for each residue. A residue whose two
# class scores are within float32 rounding of each other can be classed
# differently on a GPU, so up to this fraction of residues may differ from the
# CPU's (on MPS, none do)
CLASS_NETWORKS = ("transmembrane", "mitochondrial_targeting")
MAX_CLASS_DISAGREEMENT = 1e-3

# PyTorch's float32 settings on an NVIDIA GPU, in full precision and in TF32, as
# tf32_settings() reads them
FULL_PRECISION = ("ieee", "ieee", "ieee") if hasattr(torch.backends.cuda.matmul, "fp32_precision") else (False, False)
TF32_EVERYWHERE = ("tf32", "tf32", "tf32") if hasattr(torch.backends.cuda.matmul, "fp32_precision") else (True, True)


def fp32_precision_settings():
    """PyTorch's fp32_precision settings (2.9 and later), for matrix products, convolutions and LSTMs."""
    # Typed as Any: PyTorch's type hints do not describe these settings yet
    cudnn: Any = torch.backends.cudnn
    return (torch.backends.cuda.matmul, cudnn.conv, cudnn.rnn)


def tf32_settings():
    """PyTorch's float32 settings for matrix products, convolutions and LSTMs on an NVIDIA GPU."""
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        return tuple(setting.fp32_precision for setting in fp32_precision_settings())
    return (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


def flat(predictions):
    """Every value of a list of predictions, as one float array."""
    return np.concatenate([np.ravel(np.asarray(prediction, dtype=float)) for prediction in predictions])


def assert_close_to_cpu(on_device, on_cpu):
    """Outputs or predictions made on a device are those made on the CPU, up to float32 rounding."""
    np.testing.assert_allclose(
        flat(on_device), flat(on_cpu), rtol=DEVICE_RELATIVE_TOLERANCE, atol=DEVICE_ABSOLUTE_TOLERANCE
    )


def assert_predictions_close_to_cpu(on_device, on_cpu, network):
    """A network's predictions on a device are its predictions on the CPU, class by class for a class network."""
    assert [np.shape(prediction) for prediction in on_device] == [np.shape(prediction) for prediction in on_cpu]
    if network in CLASS_NETWORKS:
        assert np.mean(flat(on_device) != flat(on_cpu)) <= MAX_CLASS_DISAGREEMENT
    else:
        assert_close_to_cpu(on_device, on_cpu)


@functools.cache
def on_the_cpu(network):
    """A network's predictions for SEQUENCES on the CPU, alone and in a batch, and, for a
    network saved by PARROT 1, its raw output (PARROT 2's Predictor gives none)."""
    predictor = load_network_predictor(network, device="cpu")
    expected = {
        "alone": [predictor.predict(sequence) for sequence in SEQUENCES],
        "in a batch": predictor.predict_many(SEQUENCES),
    }
    if not predictor.saved_by_parrot2:
        inputs = [predictor._network_input(sequence) for sequence in SEQUENCES]
        expected["raw alone"] = [predictor._raw_predict(network_input) for network_input in inputs]
        expected["raw in a batch"] = run_network(
            predictor.network, inputs, architecture=predictor.ARCHITECTURE, device=predictor.device
        )
    return expected


def torch_module(predictor):
    """The torch module a predictor runs: its network, or for a network saved by PARROT 2, the one
    inside PARROT's Predictor."""
    return predictor.network.network if predictor.saved_by_parrot2 else predictor.network


# .................................................................
# a machine with or without each GPU, made to look so
@pytest.fixture
def two_gpus(monkeypatch):
    """A machine that looks as though it has two NVIDIA GPUs, the first the current one, and MPS."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)


@pytest.fixture
def mps_only(monkeypatch):
    """A machine that looks as though it has an Apple-silicon GPU and no NVIDIA one."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)


@pytest.fixture
def no_gpus(monkeypatch):
    """A machine that looks as though it has neither GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


@pytest.fixture
def tf32_allowed(monkeypatch):
    """TF32 allowed for every float32 computation on an NVIDIA GPU, as it may be by default."""
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        for setting in fp32_precision_settings():
            monkeypatch.setattr(setting, "fp32_precision", "tf32")
    else:
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    assert tf32_settings() == TF32_EVERYWHERE


# .................................................................
# resolving a choice of device
def test_auto_uses_cuda_where_there_is_one(two_gpus):
    assert resolve_device() == torch.device("cuda", 0)
    assert resolve_device("auto", gpuid=1) == torch.device("cuda", 1)


def test_auto_uses_the_cpu_without_cuda_even_with_mps(mps_only):
    assert resolve_device() == torch.device("cpu")
    # a GPU named for a machine with one runs on the CPU here, as gpuid always has
    assert resolve_device("auto", gpuid=1) == torch.device("cpu")


@pytest.mark.parametrize(
    ("device", "gpuid", "expected"),
    [
        ("cpu", None, torch.device("cpu")),
        ("mps", None, torch.device("mps")),
        ("cuda", None, torch.device("cuda", 0)),
        ("cuda", 1, torch.device("cuda", 1)),
        ("cuda:1", None, torch.device("cuda", 1)),
        ("cuda:1", 1, torch.device("cuda", 1)),
        (torch.device("cuda", 1), None, torch.device("cuda", 1)),
        (torch.device("cpu"), None, torch.device("cpu")),
    ],
)
def test_each_device_named_is_the_one_used(two_gpus, device, gpuid, expected):
    assert resolve_device(device, gpuid) == expected


@pytest.mark.parametrize(("device", "message"), [("cuda", "CUDA is not available"), ("mps", "MPS is not available")])
def test_a_device_this_machine_does_not_have_is_refused(no_gpus, device, message):
    with pytest.raises(SparrowException, match=message):
        resolve_device(device)


def test_a_gpu_this_machine_does_not_have_is_refused(two_gpus):
    for device in ("cuda", "auto"):
        with pytest.raises(SparrowException, match="no CUDA device 2"):
            resolve_device(device, gpuid=2)
    with pytest.raises(SparrowException, match="no CUDA device -1"):
        resolve_device("cuda", gpuid=-1)


@pytest.mark.parametrize(
    ("device", "gpuid", "message"),
    [
        ("cpu", 0, "cannot be given with device 'cpu'"),
        ("mps", 0, "cannot be given with device 'mps'"),
        ("cuda:1", 0, "names CUDA device 1, but gpuid is 0"),
    ],
)
def test_a_gpuid_that_does_not_fit_the_device_is_refused(two_gpus, device, gpuid, message):
    with pytest.raises(SparrowException, match=message):
        resolve_device(device, gpuid)


@pytest.mark.parametrize("device", ["tpu", "gpu:1", "", "meta"])
def test_an_unknown_device_is_refused(device):
    with pytest.raises(SparrowException, match="Unknown device"):
        resolve_device(device)


@pytest.mark.parametrize("gpuid", ["0", True, 1.0])
def test_a_gpuid_that_is_not_an_int_is_refused(gpuid):
    with pytest.raises(SparrowException, match="gpuid must be the index"):
        resolve_device("auto", gpuid)


def test_the_batch_functions_refuse_a_device_before_running_anything(no_gpus, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("a network was loaded")

    monkeypatch.setattr(sparrow.predictors, "load_network_predictor", fail)
    for call in (
        lambda: batch_predict(SEQUENCES, network="rg", device="cuda"),
        lambda: prepare_model("rg", device="cuda"),
        lambda: sparrow.batch(SEQUENCES, ["FCR", "predictor.radius_of_gyration"], device="cuda"),
        lambda: sparrow.batch(SEQUENCES, ["FCR"], device="tpu"),
    ):
        with pytest.raises(SparrowException, match="CUDA is not available|Unknown device"):
            call()


# .................................................................
# what sparrow.batch passes on: to sparrow's networks, and to metapredict
class _Stop(Exception):
    """Raised by a stand-in for a batch function, once it has recorded what it was given."""


@pytest.mark.parametrize(("device", "gpuid"), [("auto", None), ("cpu", None), ("mps", None), ("auto", 1)])
def test_sparrow_batch_passes_the_device_to_sparrow_s_networks(two_gpus, monkeypatch, device, gpuid):
    given = []

    def record(network, sequences, **options):
        given.append(options)
        raise _Stop

    monkeypatch.setattr(sparrow.predictors.batch_predictor, "predict_many", record)
    with pytest.raises(_Stop):
        sparrow.batch(SEQUENCES, ["predictor.radius_of_gyration"], device=device, gpuid=gpuid)
    assert given[0]["device"] == device
    assert given[0].get("gpuid") == gpuid


@pytest.mark.parametrize(
    ("device", "gpuid", "expected"),
    [
        ("auto", None, None),
        ("cpu", None, "cpu"),
        ("mps", None, "mps"),
        ("cuda", None, "cuda:0"),
        ("auto", 1, "cuda:1"),
    ],
)
@pytest.mark.parametrize("method", ["disorder", "pLDDT"])
def test_sparrow_batch_passes_the_device_to_metapredict(two_gpus, monkeypatch, method, device, gpuid, expected):
    from metapredict import meta

    given = []

    def record(sequences, **options):
        given.append(options)
        raise _Stop

    monkeypatch.setattr(meta, "predict_disorder" if method == "disorder" else "predict_pLDDT", record)
    with pytest.raises(_Stop):
        sparrow.batch(SEQUENCES, [f"predictor.{method}"], device=device, gpuid=gpuid)
    # 'auto' with no GPU named leaves metapredict to choose for itself
    assert given[0]["device"] == expected


# .................................................................
# full float32 precision on an NVIDIA GPU
def test_full_float32_precision_turns_tf32_off_and_restores_it(tf32_allowed):
    with full_float32_precision():
        assert tf32_settings() == FULL_PRECISION
    assert tf32_settings() == TF32_EVERYWHERE


def test_full_float32_precision_restores_the_settings_after_an_error(tf32_allowed):
    with pytest.raises(_Stop), full_float32_precision():
        raise _Stop
    assert tf32_settings() == TF32_EVERYWHERE


@pytest.mark.parametrize("network", ["rg", "dssp"])
def test_every_network_run_is_in_full_float32_precision(tf32_allowed, monkeypatch, network):
    # each time any LSTM runs, the float32 settings it runs with
    seen = []
    lstm_forward = torch.nn.LSTM.forward

    def recording_forward(self, *args, **kwargs):
        seen.append(tf32_settings())
        return lstm_forward(self, *args, **kwargs)

    monkeypatch.setattr(torch.nn.LSTM, "forward", recording_forward)

    predictor = load_network_predictor(network)
    predictor.predict(SEQUENCES[0])
    for batch_algorithm in ("pad-n-pack", "size-collect"):
        predictor.predict_many(SEQUENCES, batch_algorithm=batch_algorithm)
    batch_predict(SEQUENCES, network=network, show_progress_bar=False)
    sparrow.batch(SEQUENCES, ["predictor.radius_of_gyration" if network == "rg" else "predictor.dssp_helicity"])

    assert seen
    assert set(seen) == {FULL_PRECISION}
    # and the settings are as they were afterwards
    assert tf32_settings() == TF32_EVERYWHERE


# .................................................................
# on each device this machine has
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("network", [network_param(network) for network in NETWORK_PREDICTORS])
def test_every_network_runs_on_the_device_and_predicts_as_on_the_cpu(device, network):
    predictor = load_network_predictor(network, device=device)
    assert predictor.device.type == device
    assert next(torch_module(predictor).parameters()).device.type == device

    expected = on_the_cpu(network)
    if not predictor.saved_by_parrot2:
        inputs = [predictor._network_input(sequence) for sequence in SEQUENCES]
        raw_alone = [predictor._raw_predict(network_input) for network_input in inputs]
        assert_close_to_cpu(raw_alone, expected["raw alone"])
        assert_close_to_cpu(
            run_network(predictor.network, inputs, architecture=predictor.ARCHITECTURE, device=predictor.device),
            expected["raw in a batch"],
        )
    assert_predictions_close_to_cpu(
        [predictor.predict(sequence) for sequence in SEQUENCES], expected["alone"], network
    )
    assert_predictions_close_to_cpu(predictor.predict_many(SEQUENCES), expected["in a batch"], network)


@pytest.mark.parametrize("device", DEVICES)
def test_prepare_model_loads_the_network_onto_the_device(device):
    model_device, model = prepare_model("rg", device=device)
    assert model_device.type == device
    assert next(model.parameters()).device.type == device


@pytest.mark.parametrize("device", DEVICES)
def test_batch_predict_predicts_on_the_device_as_on_the_cpu(device):
    for network in ("dssp", "transmembrane"):
        predictions = batch_predict(SEQUENCES, network=network, device=device, show_progress_bar=False)
        on_device = [prediction for _, prediction in predictions.values()]
        assert_predictions_close_to_cpu(on_device, on_the_cpu(network)["in a batch"], network)


@pytest.mark.parametrize("device", DEVICES)
def test_sparrow_batch_predicts_on_the_device_as_a_protein_does_on_the_cpu(device):
    # the helicity probabilities: a probability within rounding of the threshold
    # could change the classes, and the minimum helix length a whole helix
    probability = {"mode": "probability"}
    proteins = sparrow.batch(
        SEQUENCES, ["predictor.radius_of_gyration", {"predictor.dssp_helicity": probability}], device=device
    )
    for protein, sequence in zip(proteins, SEQUENCES):
        alone = Protein(sequence).predictor
        assert_close_to_cpu([protein.predictor.radius_of_gyration()], [alone.radius_of_gyration()])
        assert_close_to_cpu([protein.predictor.dssp_helicity(**probability)], [alone.dssp_helicity(**probability)])


def test_the_batch_functions_use_cuda_where_there_is_one_and_otherwise_the_cpu():
    if torch.cuda.is_available():
        expected = torch.device("cuda", torch.cuda.current_device())
    else:
        # even on an Apple-silicon Mac, where sparrow's networks run faster on the CPU
        expected = torch.device("cpu")
    assert prepare_model("rg")[0] == expected


@requires_two_gpus
def test_gpuid_chooses_the_gpu():
    second_gpu = torch.device("cuda", 1)
    model_device, model = prepare_model("rg", gpuid=1)
    assert model_device == second_gpu
    assert next(model.parameters()).device == second_gpu
    assert load_network_predictor("dssp", device="cuda:1").device == second_gpu

    predictions = batch_predict(SEQUENCES, network="dssp", device="cuda", gpuid=1, show_progress_bar=False)
    on_device = [prediction for _, prediction in predictions.values()]
    assert_predictions_close_to_cpu(on_device, on_the_cpu("dssp")["in a batch"], "dssp")
