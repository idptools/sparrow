"""Tests for the refactored predictor backend (loader, outputs, thin classes)."""

import numpy as np
import pytest

from sparrow.predictors import outputs
from sparrow.predictors.network_loader import load_parrot_network
from sparrow.sparrow_exceptions import SparrowException

SEQ = "MEEEKKKKSSSTTTDDDQQQQNNNNGGGGSSSS"


# -- outputs helpers (pure numpy) ------------------------------------------

def test_scalar_regression_takes_first_value():
    assert outputs.scalar_regression(np.array([[3.5]])) == 3.5
    assert outputs.scalar_regression(np.array([[[7.0]]])) == 7.0


def test_argmax_per_residue():
    raw = np.array([[[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]]])
    assert list(outputs.argmax_per_residue(raw)) == [1, 0, 1]


def test_per_residue_class1_probability_matches_softmax():
    raw = np.array([[2.0, 0.0], [0.0, 2.0]])
    result = outputs.per_residue_class1_probability(raw, number_of_classes=2)
    # manual softmax of each row, take class 1
    expected = [round(outputs.softmax(row)[1], 5) for row in raw]
    assert result == expected
    assert all(0.0 <= v <= 1.0 for v in result)


def test_softmax_sums_to_one():
    assert outputs.softmax(np.array([1.0, 2.0, 3.0])).sum() == pytest.approx(1.0)


# -- network loader --------------------------------------------------------

def test_load_mto_network_metadata():
    network, meta = load_parrot_network("networks/rg/rg_network_v2.pt", architecture="MtO")
    assert meta.input_size == 20
    assert meta.number_of_classes == 1            # regression network
    assert meta.number_of_layers >= 1
    assert meta.hidden_vector_size >= 1


def test_load_mtm_network_metadata():
    _network, meta = load_parrot_network(
        "networks/nuclear_import_signal/nls_predictor_network_v1.pt", architecture="MtM"
    )
    assert meta.input_size == 20
    assert meta.number_of_classes >= 2            # classification network


def test_loader_handles_module_prefix():
    # the DSSP network stores keys with a leading 'module.' prefix; the loader
    # must strip it transparently and still load_state_dict successfully.
    network, meta = load_parrot_network(
        "networks/dssp/dssp_predictor_network_v2.pt", architecture="MtM"
    )
    assert network is not None and meta.number_of_classes >= 2


def test_loader_unknown_architecture_raises():
    with pytest.raises(SparrowException):
        load_parrot_network("networks/rg/rg_network_v2.pt", architecture="bogus")


def test_loader_missing_file_raises():
    with pytest.raises(SparrowException):
        load_parrot_network("networks/rg/does_not_exist_v999.pt", architecture="MtO")


# -- thin predictor classes ------------------------------------------------

def test_rg_predictor_returns_scalar():
    from sparrow.predictors.rg.radius_of_gyration_predictor import RgPredictor

    value = RgPredictor().predict_rg(SEQ)
    assert np.isscalar(value) or np.ndim(value) == 0
    assert float(value) > 0


def test_nls_predictor_returns_per_residue_probabilities():
    from sparrow.predictors.nls.nuclear_import_signal_predictor import NLSPredictor

    scores = NLSPredictor().predict_nuclear_import_signal(SEQ)
    assert len(scores) == len(SEQ)
    assert all(0.0 <= float(s) <= 1.0 for s in scores)


def test_transmembrane_predictor_backward_compat_alias():
    from sparrow.predictors.transmembrane.transmembrane_predictor import (
        TransmembranePredictor,
    )

    # the misspelled legacy method name still resolves to the corrected one
    assert (
        TransmembranePredictor.predict_transmebrane_regions
        is TransmembranePredictor.predict_transmembrane_regions
    )


def test_predictor_facade_matches_standalone_rg():
    from sparrow import Protein
    from sparrow.predictors.rg.radius_of_gyration_predictor import RgPredictor

    p = Protein(SEQ)
    # safe=False so the short test sequence is not re-routed to the scaled network
    facade = p.predictor.radius_of_gyration(use_scaled=False, safe=False)
    standalone = RgPredictor().predict_rg(SEQ)
    assert np.isclose(facade, standalone)
