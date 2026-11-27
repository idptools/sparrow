"""Centralized loading of PARROT-trained networks for sparrow predictors.

Historically every predictor (and ``batch_predict``) carried its own copy of the
``torch.load`` + hyper-parameter-inference + ``BRNN`` construction code. This
module consolidates that logic into a single :func:`load_parrot_network`
function so individual predictors never have to touch ``torch.load`` directly.

The loader also absorbs the small ways in which the saved ``.pt`` files vary:

* Some older networks (e.g. the DSSP and pscore networks) store every
  state-dict key with a leading ``"module."`` prefix (an artefact of
  ``DataParallel`` training). This prefix is stripped automatically *only* when
  it is actually present, so newer prefix-free networks are unaffected.
* Networks are either "many-to-one" (``BRNN_MtO``, a single value per sequence)
  or "many-to-many" (``BRNN_MtM``, a value per residue). The architecture is
  selected via the ``architecture`` argument.

Networks saved by PARROT 2 are different in kind. PARROT 1 saved only a BRNN's
weights, so sparrow has to be told the architecture and assumes one-hot input.
A PARROT 2 file records its architecture (which need not be a BRNN), its
hyperparameters, its task and its sequence encoding, so it is loaded by PARROT
2's own ``Predictor``, which rebuilds exactly the network that was trained. Every
network sparrow ships but ``plddt2`` was saved by PARROT 1 and takes the first path,
unchanged; ``plddt2`` takes the second.

sparrow itself requires only PARROT 1.7.5 or newer; PARROT 2 is needed only to
load a network saved by it, and until it is released it is installed with
:data:`PARROT2_INSTALL_COMMAND`.
"""

import importlib.util
import inspect
import os
from dataclasses import dataclass

import numpy as np
import torch
from parrot import brnn_architecture

import sparrow
from sparrow.sparrow_exceptions import SparrowException

__all__ = ["NetworkMeta", "PARROT2_INSTALL_COMMAND", "load_parrot_network"]

# the network input is always a one-hot encoding over the 20 standard amino acids
_ONE_HOT_INPUT_SIZE = 20

_MODULE_PREFIX = "module."

# Key that marks a file as a self-describing PARROT 2 checkpoint
# (FORMAT_VERSION_KEY in PARROT 2's parrot/checkpoint.py)
_PARROT2_CHECKPOINT_KEY = "parrot_checkpoint_format"

# The module that reads PARROT 2 checkpoints; it does not exist in PARROT 1
_PARROT2_CHECKPOINT_MODULE = "parrot.checkpoint"

# What sparrow needs of PARROT 2's Predictor, which earlier development versions
# of PARROT 2 do not have: the method that predicts many sequences at once, which
# BaseNetworkPredictor.predict_many uses, and the argument that chooses a CUDA
# device by index, as sparrow's gpuid does
_PARROT2_BATCH_METHOD = "predict_batch"
_PARROT2_GPU_ARGUMENT = "gpu_id"

#: How to install PARROT 2, which loading a network saved by it requires. PARROT 2
#: is not yet on PyPI, where ``idptools-parrot`` is PARROT 1.
PARROT2_INSTALL_COMMAND = 'pip install "git+https://github.com/idptools/parrot.git@parrot2"'


@dataclass
class NetworkMeta:
    """Hyper-parameters of a loaded PARROT network.

    For a PARROT 1 network these are inferred from the state-dict; a PARROT 2
    network records them.

    Attributes
    ----------
    number_of_classes : int
        Size of the network output layer: 1 for regression, the number of
        classes for classification.
    input_size : int
        Size of the per-residue input encoding (20 for one-hot, which every
        PARROT 1 network uses).
    number_of_layers : int or None
        Number of stacked (bidirectional) LSTM layers. None for a PARROT 2
        network that is not a BRNN.
    hidden_vector_size : int or None
        Hidden vector size of each LSTM layer. None for a PARROT 2 network
        whose architecture has no hidden size.
    device : torch.device
        Device the weights were mapped onto.
    saved_by_parrot2 : bool
        True if the file was saved by PARROT 2, in which case the loaded network
        is a PARROT ``Predictor`` (see :func:`load_parrot_network`). Default
        False.
    datatype : str or None
        What a PARROT 2 network predicts: ``"sequence"`` (one value per
        sequence), ``"residues"`` (one per residue) or ``"matrix"``. None for a
        PARROT 1 network, whose file does not record it. Default None.
    """

    number_of_classes: int
    input_size: int
    number_of_layers: int | None
    hidden_vector_size: int | None
    device: torch.device
    saved_by_parrot2: bool = False
    datatype: str | None = None


def _strip_module_prefix(state_dict):
    """Remove a leading ``"module."`` from state-dict keys when present.

    Returns the state-dict unchanged if no key carries the prefix, so networks
    trained without ``DataParallel`` are not affected.
    """
    if not any(key.startswith(_MODULE_PREFIX) for key in state_dict):
        return state_dict

    return {
        (key[len(_MODULE_PREFIX) :] if key.startswith(_MODULE_PREFIX) else key): value
        for key, value in state_dict.items()
    }


def _infer_num_layers(state_dict):
    """Count stacked LSTM layers by walking ``lstm.weight_ih_l{n}`` keys."""
    num_layers = 0
    while f"lstm.weight_ih_l{num_layers}" in state_dict:
        num_layers += 1
    return num_layers


def _is_parrot2_checkpoint(saved):
    """True if a loaded ``.pt`` file is a PARROT 2 checkpoint rather than PARROT 1 weights."""
    return isinstance(saved, dict) and _PARROT2_CHECKPOINT_KEY in saved


def _missing_parrot2_features(predictor_class):
    """What sparrow needs of PARROT 2's ``Predictor`` that the one installed does not have.

    Parameters
    ----------
    predictor_class : type
        ``parrot.py_predictor.Predictor``.

    Returns
    -------
    list of str
        Empty if it has everything.
    """
    missing = []
    if not hasattr(predictor_class, _PARROT2_BATCH_METHOD):
        missing.append(f"Predictor.{_PARROT2_BATCH_METHOD}")
    if _PARROT2_GPU_ARGUMENT not in inspect.signature(predictor_class).parameters:
        missing.append(f"Predictor's {_PARROT2_GPU_ARGUMENT} argument")
    return missing


def _load_parrot2_network(saved_weights, device):
    """Load a network saved by PARROT 2 with PARROT 2's own ``Predictor``.

    Parameters
    ----------
    saved_weights : str
        Absolute path to the checkpoint.
    device : torch.device or str
        Device to run predictions on: ``"cpu"``, ``"mps"``, or ``"cuda"``, with
        or without a device index (without one, the current CUDA device).

    Returns
    -------
    tuple
        ``(predictor, meta)``: a ``parrot.py_predictor.Predictor`` and the
        :class:`NetworkMeta` the checkpoint records.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If PARROT 2 is not installed, or is a development version without
        ``Predictor.predict_batch`` or ``Predictor``'s ``gpu_id`` argument.
    """
    if importlib.util.find_spec(_PARROT2_CHECKPOINT_MODULE) is None:
        raise SparrowException(
            f"{saved_weights} was saved by PARROT 2, which is not installed (sparrow itself needs only PARROT 1). "
            f"Install PARROT 2 to use this network: {PARROT2_INSTALL_COMMAND}"
        )
    # imported here because this Predictor reads PARROT 2 checkpoints only in PARROT 2
    from parrot.py_predictor import Predictor

    # refused at loading, so a network that loads can be used as any network sparrow ships can be
    missing = _missing_parrot2_features(Predictor)
    if missing:
        raise SparrowException(
            f"{saved_weights} was saved by PARROT 2, and the PARROT 2 installed is a development version without "
            f"{' or '.join(missing)}, which sparrow needs to run it. Update PARROT 2 to use this network: "
            f"{PARROT2_INSTALL_COMMAND}"
        )

    # PARROT takes the device by type, and a CUDA device's index separately
    device = torch.device(device)
    gpu_id = device.index if device.type == "cuda" else None
    predictor = Predictor(saved_weights, device=device.type, gpu_id=gpu_id)
    meta = NetworkMeta(
        number_of_classes=predictor.n_classes,
        input_size=predictor.network.input_size,
        number_of_layers=predictor.num_layers,
        hidden_vector_size=predictor.hidden_vector_size,
        device=predictor.device,
        saved_by_parrot2=True,
        datatype=predictor.dtype,
    )
    return predictor, meta


def load_parrot_network(relative_path, architecture="MtO", device=None):
    """Load a PARROT-trained network shipped inside ``sparrow/data``.

    A network saved by PARROT 1 (every network sparrow ships but ``plddt2``) is rebuilt as a
    BRNN with the given ``architecture`` and one-hot input. A network saved by
    PARROT 2 records its own architecture, task and encoding, so ``architecture``
    is ignored and the network is loaded by PARROT 2's ``Predictor`` instead.

    Parameters
    ----------
    relative_path : str
        Path to the weights file relative to the sparrow ``data`` directory,
        for example ``"networks/rg/rg_network_v2.pt"``. Resolved with
        ``sparrow.data_path``.
    architecture : {"MtO", "MtM"}, optional
        Network architecture of a PARROT 1 network. ``"MtO"`` builds a
        many-to-one network (one value per sequence); ``"MtM"`` builds a
        many-to-many network (one value per residue). Default ``"MtO"``.
    device : torch.device or str or None, optional
        Device to map the weights onto. Defaults to CPU.

    Returns
    -------
    tuple
        ``(network, meta)``, where ``meta`` is a :class:`NetworkMeta`
        describing the network. For a PARROT 1 network, ``network`` is the
        instantiated, weight-loaded ``BRNN`` model, called on one-hot encoded
        sequences. For a PARROT 2 network, it is a ``parrot.py_predictor.Predictor``,
        whose ``predict(seq)`` encodes the sequence as the network was trained to
        expect.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the weights file does not exist or an unknown architecture is given;
        or, for a network saved by PARROT 2, if PARROT 2 is not installed, or
        is a development version without ``Predictor.predict_batch`` or
        ``Predictor``'s ``gpu_id`` argument.
    """
    if device is None:
        device = torch.device("cpu")

    saved_weights = sparrow.data_path(relative_path)
    if not os.path.isfile(saved_weights):
        raise SparrowException(f"Error: could not find saved weights file {saved_weights}")

    state_dict = torch.load(saved_weights, map_location=device)
    if _is_parrot2_checkpoint(state_dict):
        return _load_parrot2_network(saved_weights, device)

    state_dict = _strip_module_prefix(state_dict)

    number_of_layers = _infer_num_layers(state_dict)
    number_of_classes = np.shape(state_dict["fc.bias"])[0]
    input_size = _ONE_HOT_INPUT_SIZE
    hidden_vector_size = int(np.shape(state_dict["lstm.weight_ih_l0"])[0] / 4)

    if architecture == "MtO":
        network = brnn_architecture.BRNN_MtO(
            input_size, hidden_vector_size, number_of_layers, number_of_classes, device
        )
    elif architecture == "MtM":
        network = brnn_architecture.BRNN_MtM(
            input_size, hidden_vector_size, number_of_layers, number_of_classes, device
        )
    else:
        raise SparrowException(f"Unknown network architecture '{architecture}' (expected 'MtO' or 'MtM')")

    network.load_state_dict(state_dict)

    meta = NetworkMeta(
        number_of_classes=number_of_classes,
        input_size=input_size,
        number_of_layers=number_of_layers,
        hidden_vector_size=hidden_vector_size,
        device=device,
    )
    return network, meta
