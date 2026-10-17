"""Networks saved by PARROT 2, loaded through the predictor template.

Every network sparrow ships was saved by PARROT 1; test_predictor_internals.py
covers those. These tests build small PARROT 2 networks at test time (untrained:
loading and predicting do not depend on the weights being any good), save them
as PARROT 2 does, and check that the template in
sparrow/predictors/predictor_template.py, with only its weights path changed,
predicts exactly what PARROT 2's own Predictor does. They are skipped when
PARROT 2 is not installed, except the one that checks the error in that case.
"""

import importlib.util
import sys

import numpy as np
import pytest
import torch

from sparrow.predictors.network_loader import PARROT2_INSTALL_COMMAND, load_parrot_network
from sparrow.predictors.rg.radius_of_gyration_predictor import RgPredictor
from sparrow.sparrow_exceptions import SparrowException

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


def save_parrot2_network(path, architecture, datatype, num_classes, hyperparameters):
    """Build an untrained network and save it as parrot-train does."""
    from parrot import Task, build_network, save_network
    from parrot.encode_sequence import get_encoder

    torch.manual_seed(WEIGHTS_SEED)
    network = build_network(architecture, ONE_HOT_WIDTH, Task(datatype, num_classes), hyperparameters)
    save_network(str(path), network, get_encoder("onehot"))


def filled_in_template(weights_path_template):
    """The template predictor, with only its weights path changed, as a contributor would fill it in."""
    from sparrow.predictors.predictor_template import MyPropertyPredictor

    class FilledIn(MyPropertyPredictor):
        # an absolute path: sparrow.get_data joins it onto the data directory, which
        # os.path.join leaves as the absolute path
        NETWORK_PATH = weights_path_template

    return FilledIn


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
def test_a_device_index_is_refused_for_a_parrot2_network(tmp_path):
    # called directly: through load_parrot_network, torch.load would first have to
    # map the weights onto that device, which needs one to exist
    from sparrow.predictors.network_loader import _load_parrot2_network

    save_parrot2_network(tmp_path / "network.pt", "brnn", "residues", 1, {})
    with pytest.raises(SparrowException, match="device index"):
        _load_parrot2_network(str(tmp_path / "network.pt"), "cuda:1")


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
