"""The pLDDT2 predictor: sparrow's own network for AlphaFold2's per-residue confidence.

The network was saved by PARROT 2, so every test here is skipped where PARROT 2
is not installed (see parrot2_markers.py). The tests that every network passes,
alone and in batches and on each device, run it too (test_batch_predictor.py,
test_devices.py, test_batch.py); these check what is particular to it.
"""

import numpy as np
import pytest

import sparrow
from sparrow import Protein
from sparrow.predictors import load_network_predictor
from sparrow.predictors.plddt2.plddt2_predictor import (
    DEFAULT_VERSION,
    PLDDT_MAXIMUM,
    PLDDT_MINIMUM,
    PLDDT_SCALE,
    PLDDT2Predictor,
)
from sparrow.tests.parrot2_markers import requires_parrot2

pytestmark = requires_parrot2

# Human p53 (UniProt P04637), whose disordered N-terminal transactivation domain
# AlphaFold2 gives a low pLDDT and whose DNA-binding domain (residues ~94-292) a
# high one
P53 = (
    "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGF"
    "RLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVV"
    "PYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDG"
    "EYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
)

# Version 1's prediction for p53 on the CPU when it was added to sparrow (November
# 2026): the first ten residues, ten in the DNA-binding domain, and the mean
P53_FIRST_TEN = [42.8022, 44.3587, 45.6468, 49.5198, 46.8588, 47.2786, 48.4153, 51.5444, 48.6599, 51.3005]
P53_DNA_BINDING_START = 100
P53_DNA_BINDING_TEN = [70.5697, 74.0836, 76.9783, 77.7255, 78.183, 81.4561, 83.8346, 85.443, 88.4088, 89.3081]
P53_MEAN = 73.7182

# The values above are rounded to 4 decimal places, and another CPU may round
# float32 arithmetic differently, by far less than this
REFERENCE_TOLERANCE = 1e-3

# The architecture version 1 records: a two-layer bidirectional LSTM with hidden
# size 64 on one-hot sequences, with one output per residue
EXPECTED_LAYERS = 2
EXPECTED_HIDDEN_SIZE = 64
ONE_HOT_WIDTH = 20

SEQUENCES = ["MEEEKKKKSSSTTTDDDGRGRGYYPQ", "ACDEFGHIKLMNPQRSTVWY" * 3, "GSGSGS", P53]


def test_the_shipped_network_is_a_parrot2_residue_regression():
    predictor = load_network_predictor("plddt2")
    assert isinstance(predictor, PLDDT2Predictor)
    assert predictor.saved_by_parrot2
    assert predictor.number_of_classes == 1
    assert predictor.input_size == ONE_HOT_WIDTH
    assert (predictor.number_of_layers, predictor.hidden_vector_size) == (EXPECTED_LAYERS, EXPECTED_HIDDEN_SIZE)


def test_the_prediction_is_parrots_times_100_clipped_to_0_100():
    from parrot import Predictor

    predictor = PLDDT2Predictor()
    weights_file = sparrow.data_path(PLDDT2Predictor.NETWORK_PATH.format(version=DEFAULT_VERSION))
    reference = Predictor(weights_file, device="cpu")
    for sequence in SEQUENCES:
        prediction = predictor.predict_plddt2(sequence)
        expected = np.clip(reference.predict(sequence) * PLDDT_SCALE, PLDDT_MINIMUM, PLDDT_MAXIMUM)
        assert prediction.dtype == np.float32
        assert prediction.shape == (len(sequence),)
        np.testing.assert_array_equal(prediction, expected)


def test_a_prediction_outside_0_100_is_clipped_to_it():
    raw = np.array([-0.05, 0.0, 0.42, 1.0, 1.03], dtype=np.float32)
    prediction = PLDDT2Predictor._from_raw(None, "AAAAA", raw)
    assert prediction.dtype == np.float32
    np.testing.assert_allclose(prediction, [0.0, 0.0, 42.0, 100.0, 100.0], rtol=1e-6)


def test_p53_is_predicted_as_when_the_network_was_added():
    prediction = Protein(P53).predictor.pLDDT2()
    dna_binding = prediction[P53_DNA_BINDING_START : P53_DNA_BINDING_START + len(P53_DNA_BINDING_TEN)]
    np.testing.assert_allclose(prediction[: len(P53_FIRST_TEN)], P53_FIRST_TEN, rtol=0, atol=REFERENCE_TOLERANCE)
    np.testing.assert_allclose(dna_binding, P53_DNA_BINDING_TEN, rtol=0, atol=REFERENCE_TOLERANCE)
    assert float(np.mean(prediction)) == pytest.approx(P53_MEAN, abs=REFERENCE_TOLERANCE)


def test_protein_pLDDT2_runs_the_network_once_and_hands_back_copies(monkeypatch):
    calls = []
    real_predict = PLDDT2Predictor.predict

    def counting_predict(self, sequence):
        calls.append(sequence)
        return real_predict(self, sequence)

    monkeypatch.setattr(PLDDT2Predictor, "predict", counting_predict)
    protein = Protein(P53)
    first = protein.predictor.pLDDT2()
    first[:] = -1.0
    second = protein.predictor.pLDDT2()
    assert calls == [P53]
    assert np.all(second >= PLDDT_MINIMUM)
    protein.predictor.pLDDT2(recompute=True)
    assert calls == [P53, P53]
