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
"""

import os
from dataclasses import dataclass

import numpy as np
import torch
from parrot import brnn_architecture

import sparrow
from sparrow.sparrow_exceptions import SparrowException

__all__ = ["NetworkMeta", "load_parrot_network"]

# the network input is always a one-hot encoding over the 20 standard amino acids
_ONE_HOT_INPUT_SIZE = 20

_MODULE_PREFIX = "module."


@dataclass
class NetworkMeta:
    """Hyper-parameters inferred from a loaded PARROT network state-dict.

    Attributes
    ----------
    number_of_classes : int
        Size of the network output layer (``fc.bias``).
    input_size : int
        Size of the per-residue input encoding (always 20 for one-hot).
    number_of_layers : int
        Number of stacked (bidirectional) LSTM layers.
    hidden_vector_size : int
        Hidden vector size of each LSTM layer.
    device : torch.device
        Device the weights were mapped onto.
    """

    number_of_classes: int
    input_size: int
    number_of_layers: int
    hidden_vector_size: int
    device: torch.device


def _strip_module_prefix(state_dict):
    """Remove a leading ``"module."`` from state-dict keys when present.

    Returns the state-dict unchanged if no key carries the prefix, so networks
    trained without ``DataParallel`` are not affected.
    """
    if not any(key.startswith(_MODULE_PREFIX) for key in state_dict):
        return state_dict

    return {
        (key[len(_MODULE_PREFIX):] if key.startswith(_MODULE_PREFIX) else key): value
        for key, value in state_dict.items()
    }


def _infer_num_layers(state_dict):
    """Count stacked LSTM layers by walking ``lstm.weight_ih_l{n}`` keys."""
    num_layers = 0
    while f"lstm.weight_ih_l{num_layers}" in state_dict:
        num_layers += 1
    return num_layers


def load_parrot_network(relative_path, architecture="MtO", device=None):
    """Load a PARROT-trained network shipped inside ``sparrow/data``.

    Parameters
    ----------
    relative_path : str
        Path to the weights file relative to the sparrow ``data`` directory,
        for example ``"networks/rg/rg_network_v2.pt"``. Resolved with
        :func:`sparrow.get_data`.
    architecture : {"MtO", "MtM"}, optional
        Network architecture. ``"MtO"`` builds a many-to-one network (one value
        per sequence); ``"MtM"`` builds a many-to-many network (one value per
        residue). Default ``"MtO"``.
    device : torch.device or str or None, optional
        Device to map the weights onto. Defaults to CPU.

    Returns
    -------
    tuple
        ``(network, meta)`` where ``network`` is the instantiated, weight-loaded
        ``BRNN`` model and ``meta`` is a :class:`NetworkMeta` describing the
        inferred hyper-parameters.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the weights file does not exist or an unknown architecture is given.
    """
    if device is None:
        device = torch.device("cpu")

    saved_weights = sparrow.get_data(relative_path)
    if not os.path.isfile(saved_weights):
        raise SparrowException(
            f"Error: could not find saved weights file {saved_weights}"
        )

    state_dict = torch.load(saved_weights, map_location=device)
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
        raise SparrowException(
            f"Unknown network architecture '{architecture}' (expected 'MtO' or 'MtM')"
        )

    network.load_state_dict(state_dict)

    meta = NetworkMeta(
        number_of_classes=number_of_classes,
        input_size=input_size,
        number_of_layers=number_of_layers,
        hidden_vector_size=hidden_vector_size,
        device=device,
    )
    return network, meta
