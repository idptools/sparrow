"""Networks saved by PARROT 2, loaded through the predictor template.

Every network sparrow ships was saved by PARROT 1; test_predictor_internals.py
covers those. These tests build small PARROT 2 networks at test time (untrained:
loading and predicting do not depend on the weights being any good), save them
as PARROT 2 does, and check that the template in
sparrow/predictors/predictor_template.py, with only its weights path changed,
predicts exactly what PARROT 2's own Predictor does, one sequence at a time and
in batches, and that a network registered as docs/predictors.rst describes runs
in batch_predict and sparrow.batch as a network saved by PARROT 1 does. They are
skipped when PARROT 2 is not installed, except those that do not need it.
"""

import importlib.util
import sys
import types

import numpy as np
import pytest
import torch

import sparrow
import sparrow.batching
from sparrow import Protein
from sparrow.predictors import METHOD_NETWORKS, NETWORK_PREDICTORS
from sparrow.predictors import Predictor as ProteinPredictor
from sparrow.predictors.base import BaseNetworkPredictor
from sparrow.predictors.batch_predictor import DEFAULT_BATCH_SIZE, batch_predict, prepare_model
from sparrow.predictors.network_loader import PARROT2_INSTALL_COMMAND, load_parrot_network
from sparrow.predictors.rg.radius_of_gyration_predictor import RgPredictor
from sparrow.sparrow_exceptions import SparrowException
from sparrow.tools.utilities import copy_of

# The module that reads PARROT 2 checkpoints, which PARROT 1 does not have
PARROT2_MODULE = "parrot.checkpoint"

requires_parrot2 = pytest.mark.skipif(
    importlib.util.find_spec(PARROT2_MODULE) is None, reason="PARROT 2 is not installed"
)

#: Seed for the random initial weights of the test networks.
WEIGHTS_SEED = 7

#: Width of a one-hot encoding over the 20 standard amino acids.
ONE_HOT_WIDTH = 20

SEQUENCES = ["MEEEKKKKSSSTTTDDDGRGRGYYPQ", "ACDEFGHIKLMNPQRSTVWY" * 3, "GSGSGS"]

#: The sequences predicted in batches; the first appears twice.
BATCHED_SEQUENCES = [*SEQUENCES, SEQUENCES[0]]

#: How far a prediction made in a batch may be from one made alone. They differ
#: at the precision of the networks' 32-bit arithmetic, because a batch is
#: summed in a different order: by at most 1.2e-6 for these networks, over 200
#: random sequences of 5 to 400 residues.
BATCH_ABSOLUTE_TOLERANCE = 1e-5

#: The name a test network is registered under, as a contributor would register
#: theirs, and the module and class its predictor is put in.
REGISTERED_NETWORK = "my_property"
REGISTERED_MODULE = "registered_parrot2_predictor"
REGISTERED_CLASS = "RegisteredPredictor"

#: (architecture, datatype, number of classes, hyperparameters) of each test
#: network. The sequence BRNN with no hyperparameters given has PARROT 2's default
#: head (attention pooling plus log length), which a PARROT 1 BRNN cannot hold.
NETWORKS = {
    "residues regression": ("brnn", "residues", 1, {}),
    "residues classification": ("brnn", "residues", 3, {}),
    "sequence regression, default head": ("brnn", "sequence", 1, {}),
    "sequence regression, PARROT 1 head": (
        "brnn",
        "sequence",
        1,
        {"attention_pooling": False, "loglength_feedforward": False},
    ),
    "sequence classification": ("brnn", "sequence", 2, {}),
    "residues regression, CNN": ("cnn", "residues", 1, {}),
}


def save_parrot2_network(path, architecture, datatype, num_classes, hyperparameters, encoding="onehot"):
    """Build an untrained network and save it as parrot-train does."""
    from parrot import Task, build_network, save_network
    from parrot.encode_sequence import get_encoder

    encoder = get_encoder(encoding)
    torch.manual_seed(WEIGHTS_SEED)
    network = build_network(architecture, encoder.input_size, Task(datatype, num_classes), hyperparameters)
    save_network(str(path), network, encoder)


def filled_in_template(weights_path_template):
    """The template predictor, with only its weights path changed, as a contributor would fill it in."""
    from sparrow.predictors.predictor_template import MyPropertyPredictor

    class FilledIn(MyPropertyPredictor):
        # an absolute path: sparrow.data_path joins it onto the data directory, which
        # os.path.join leaves as the absolute path
        NETWORK_PATH = weights_path_template

    return FilledIn


def register_network(tmp_path, monkeypatch, encoding="onehot"):
    """Save a PARROT 2 network and register it in sparrow as docs/predictors.rst says to, until the test ends."""
    save_parrot2_network(tmp_path / "network_v1.pt", "brnn", "residues", 1, {}, encoding=encoding)

    # the predictor module a contributor would add
    module = types.ModuleType(REGISTERED_MODULE)
    setattr(module, REGISTERED_CLASS, filled_in_template(str(tmp_path / "network_v{version}.pt")))
    monkeypatch.setitem(sys.modules, REGISTERED_MODULE, module)

    # 1. the network, in NETWORK_PREDICTORS
    monkeypatch.setitem(NETWORK_PREDICTORS, REGISTERED_NETWORK, (REGISTERED_MODULE, REGISTERED_CLASS))

    # 2. the Protein.predictor method that reads it
    def my_property(self, recompute=False):
        """My property at each residue."""
        return copy_of(self._prediction(REGISTERED_NETWORK, recompute))

    monkeypatch.setattr(ProteinPredictor, REGISTERED_NETWORK, my_property, raising=False)

    # 3. the method, in METHOD_NETWORKS, and so in the BATCH_METHODS sparrow.batching
    # built from it when it was imported
    monkeypatch.setitem(METHOD_NETWORKS, REGISTERED_NETWORK, REGISTERED_NETWORK)
    monkeypatch.setattr(sparrow.batching, "BATCH_METHODS", (*sparrow.batching.BATCH_METHODS, REGISTERED_NETWORK))


@requires_parrot2
@pytest.mark.parametrize("name", list(NETWORKS))
def test_the_template_predicts_what_parrot2_does(tmp_path, name):
    from parrot import Predictor

    architecture, datatype, num_classes, hyperparameters = NETWORKS[name]
    save_parrot2_network(tmp_path / "network_v1.pt", architecture, datatype, num_classes, hyperparameters)

    predictor = filled_in_template(str(tmp_path / "network_v{version}.pt"))()
    reference = Predictor(tmp_path / "network_v1.pt")

    assert predictor.saved_by_parrot2
    assert predictor.number_of_classes == num_classes
    for sequence in SEQUENCES:
        prediction = predictor.predict_my_property(sequence)
        expected = reference.predict(sequence)
        assert prediction.dtype == expected.dtype
        np.testing.assert_array_equal(prediction, expected)


@requires_parrot2
@pytest.mark.parametrize("name", list(NETWORKS))
def test_the_template_predicts_many_as_parrot2_does(tmp_path, name):
    from parrot import Predictor

    architecture, datatype, num_classes, hyperparameters = NETWORKS[name]
    save_parrot2_network(tmp_path / "network_v1.pt", architecture, datatype, num_classes, hyperparameters)

    predictor = filled_in_template(str(tmp_path / "network_v{version}.pt"))()
    reference = Predictor(tmp_path / "network_v1.pt")

    batched = predictor.predict_many(BATCHED_SEQUENCES)
    expected = reference.predict_batch(BATCHED_SEQUENCES, batch_size=DEFAULT_BATCH_SIZE)
    assert len(batched) == len(BATCHED_SEQUENCES)
    for sequence, prediction, expected_prediction in zip(BATCHED_SEQUENCES, batched, expected):
        assert prediction.dtype == expected_prediction.dtype
        np.testing.assert_array_equal(prediction, expected_prediction)
        alone = predictor.predict(sequence)
        assert prediction.shape == alone.shape
        np.testing.assert_allclose(prediction, alone, rtol=0, atol=BATCH_ABSOLUTE_TOLERANCE)


@requires_parrot2
def test_a_registered_parrot2_network_runs_in_batches_like_any_other(tmp_path, monkeypatch):
    register_network(tmp_path, monkeypatch)
    alone = [Protein(sequence).predictor.my_property() for sequence in BATCHED_SEQUENCES]

    by_name = batch_predict(BATCHED_SEQUENCES, network=REGISTERED_NETWORK, show_progress_bar=False)
    proteins = sparrow.batch(BATCHED_SEQUENCES, ["predictor.my_property"])

    def fail(*args, **kwargs):
        raise AssertionError("a prediction sparrow.batch made was made again")

    # what sparrow.batch predicted is read, not predicted again
    monkeypatch.setattr(BaseNetworkPredictor, "predict", fail)
    monkeypatch.setattr(BaseNetworkPredictor, "predict_many", fail)

    assert list(by_name) == list(range(len(BATCHED_SEQUENCES)))
    for index, sequence in enumerate(BATCHED_SEQUENCES):
        assert by_name[index][0] == sequence
        np.testing.assert_allclose(by_name[index][1], alone[index], rtol=0, atol=BATCH_ABSOLUTE_TOLERANCE)
        from_batch = proteins[index].predictor.my_property()
        np.testing.assert_allclose(from_batch, alone[index], rtol=0, atol=BATCH_ABSOLUTE_TOLERANCE)


@requires_parrot2
def test_a_bad_batch_algorithm_is_refused_for_a_parrot2_network_too(tmp_path):
    save_parrot2_network(tmp_path / "network_v1.pt", "brnn", "residues", 1, {})
    predictor = filled_in_template(str(tmp_path / "network_v{version}.pt"))()
    with pytest.raises(SparrowException, match="batch_algorithm"):
        predictor.predict_many(SEQUENCES, batch_algorithm="all-at-once")


@requires_parrot2
def test_prepare_model_gives_a_parrot2_network_to_run_on_one_hot_sequences(tmp_path, monkeypatch):
    from parrot import Predictor
    from parrot.encode_sequence import one_hot

    register_network(tmp_path, monkeypatch)
    device, model = prepare_model(REGISTERED_NETWORK, None, 0)
    reference = Predictor(tmp_path / "network_v1.pt")

    assert isinstance(model, torch.nn.Module)
    assert not model.training
    for sequence in SEQUENCES:
        # called as a network saved by PARROT 1 is, on one sequence, one-hot encoded
        with torch.no_grad():
            raw = model(one_hot(sequence).float().view(1, len(sequence), -1).to(device)).cpu().numpy()
        assert raw.shape == (1, len(sequence), 1)
        np.testing.assert_allclose(raw[0, :, 0], reference.predict(sequence), rtol=0, atol=BATCH_ABSOLUTE_TOLERANCE)


@requires_parrot2
def test_prepare_model_refuses_a_parrot2_network_trained_on_another_encoding(tmp_path, monkeypatch):
    register_network(tmp_path, monkeypatch, encoding="biophysics")
    with pytest.raises(SparrowException, match="'biophysics' encoding"):
        prepare_model(REGISTERED_NETWORK, None, 0)


def predictor_without_gpu_id(
    self, saved_weights, dtype=None, device="cpu", embedding_cache=None, embed_in_isolation=False
):
    """Predictor.__init__ as earlier development versions of PARROT 2 have it."""
    raise AssertionError("sparrow should refuse this PARROT 2 before creating a Predictor")


@requires_parrot2
@pytest.mark.parametrize("missing", ["predict_batch", "gpu_id"])
def test_a_parrot2_without_what_sparrow_needs_says_to_update_it(tmp_path, monkeypatch, missing):
    from parrot.py_predictor import Predictor

    save_parrot2_network(tmp_path / "network.pt", "brnn", "residues", 1, {})
    # as in earlier development versions
    if missing == "predict_batch":
        monkeypatch.delattr(Predictor, "predict_batch")
    else:
        monkeypatch.setattr(Predictor, "__init__", predictor_without_gpu_id)
    with pytest.raises(SparrowException, match=missing) as error:
        load_parrot_network(str(tmp_path / "network.pt"))
    assert PARROT2_INSTALL_COMMAND in str(error.value)


@requires_parrot2
def test_the_default_sequence_head_is_one_a_parrot1_brnn_cannot_hold(tmp_path):
    """Guards the case above: PARROT 2's default sequence BRNN has weights PARROT 1's has not."""
    save_parrot2_network(tmp_path / "network.pt", "brnn", "sequence", 1, {})
    saved = torch.load(tmp_path / "network.pt", map_location="cpu")
    assert any(key.startswith("attention_pool.") for key in saved["state_dict"])


@requires_parrot2
def test_the_loader_reports_what_a_parrot2_network_records(tmp_path):
    save_parrot2_network(tmp_path / "network.pt", "brnn", "residues", 3, {"hidden_size": 6, "num_layers": 2})
    # the architecture argument applies to PARROT 1 networks only, so a wrong one is ignored
    _predictor, meta = load_parrot_network(str(tmp_path / "network.pt"), architecture="MtO")
    assert meta.saved_by_parrot2
    assert meta.datatype == "residues"
    assert (meta.number_of_classes, meta.input_size) == (3, ONE_HOT_WIDTH)
    assert (meta.number_of_layers, meta.hidden_vector_size) == (2, 6)


@requires_parrot2
def test_versions_select_different_weights_files(tmp_path):
    save_parrot2_network(tmp_path / "network_v1.pt", "brnn", "residues", 1, {})
    torch.manual_seed(WEIGHTS_SEED + 1)
    save_parrot2_network(tmp_path / "network_v2.pt", "brnn", "residues", 1, {"hidden_size": 4})
    predictor_class = filled_in_template(str(tmp_path / "network_v{version}.pt"))
    assert predictor_class().hidden_vector_size != predictor_class(version="2").hidden_vector_size


@requires_parrot2
def test_raw_output_is_not_offered_for_a_parrot2_network(tmp_path):
    save_parrot2_network(tmp_path / "network_v1.pt", "brnn", "residues", 1, {})
    predictor = filled_in_template(str(tmp_path / "network_v{version}.pt"))()
    with pytest.raises(SparrowException, match="_predict"):
        predictor._raw_predict(SEQUENCES[0])


@requires_parrot2
@pytest.mark.parametrize(
    ("device", "expected"),
    [("cuda:1", ("cuda", 1)), ("cuda", ("cuda", None)), ("cpu", ("cpu", None)), ("mps", ("mps", None))],
)
def test_the_device_is_passed_on_to_parrot2_by_type_and_index(tmp_path, monkeypatch, device, expected):
    # called directly: through load_parrot_network, torch.load would first have to
    # map the weights onto that device, which needs one to exist
    import parrot.py_predictor
    from sparrow.predictors.network_loader import _load_parrot2_network

    asked = []

    class Recording(parrot.py_predictor.Predictor):
        """PARROT 2's Predictor, noting the device it is asked for and loading on the CPU, which every machine has."""

        def __init__(self, saved_weights, device="cpu", gpu_id=None):
            asked.append((device, gpu_id))
            super().__init__(saved_weights)

    monkeypatch.setattr(parrot.py_predictor, "Predictor", Recording)
    save_parrot2_network(tmp_path / "network.pt", "brnn", "residues", 1, {})
    _load_parrot2_network(str(tmp_path / "network.pt"), device)
    assert asked == [expected]


def test_a_parrot1_network_is_not_run_through_parrot2():
    predictor = RgPredictor()
    assert not predictor.saved_by_parrot2
    with pytest.raises(SparrowException, match="_raw_predict"):
        predictor._predict(SEQUENCES[0])


def test_a_parrot2_network_without_parrot2_installed_says_how_to_install_it(tmp_path, monkeypatch):
    # only the marker key matters before PARROT 2 is needed, so this runs under PARROT 1 too
    torch.save({"parrot_checkpoint_format": 1}, tmp_path / "network.pt")
    monkeypatch.setitem(sys.modules, PARROT2_MODULE, None)  # import as if PARROT 2 were absent
    with pytest.raises(SparrowException, match="saved by PARROT 2") as error:
        load_parrot_network(str(tmp_path / "network.pt"))
    assert PARROT2_INSTALL_COMMAND in str(error.value)
