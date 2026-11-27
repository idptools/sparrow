"""The tests that run a network saved by PARROT 2, skipped where PARROT 2 is not installed.

sparrow itself needs only PARROT 1, which is what its CI installs, but loading a
network saved by PARROT 2 - ``plddt2``, behind ``Protein.predictor.pLDDT2`` -
needs PARROT 2. A test that runs every network, or every prediction
``sparrow.batch`` makes, gives each as a parameter through :func:`network_param`
or :func:`prediction_param`, so the ones that need PARROT 2 are skipped, with
the reason, where it is not installed, and run everywhere else. Used by
test_batch_predictor.py, test_devices.py, test_batch.py and
test_predictors_parrot2.py.
"""

import functools
import importlib.util

import pytest
import torch

import sparrow
from sparrow.predictors import METHOD_NETWORKS, network_predictor_class
from sparrow.predictors.network_loader import _is_parrot2_checkpoint

# The module that reads PARROT 2 checkpoints, which PARROT 1 does not have
PARROT2_MODULE = "parrot.checkpoint"

PARROT2_INSTALLED = importlib.util.find_spec(PARROT2_MODULE) is not None

requires_parrot2 = pytest.mark.skipif(not PARROT2_INSTALLED, reason="PARROT 2 is not installed")

# The prefix of the names sparrow.batch takes for the Protein.predictor methods
PREDICTOR_PREFIX = "predictor."


@functools.cache
def saved_by_parrot2(network):
    """Whether one of sparrow's networks, in its default version, was saved by PARROT 2.

    Read from the weights file, as sparrow's loader tells, so it needs no PARROT 2.

    Parameters
    ----------
    network : str
        One of :data:`sparrow.predictors.NETWORK_PREDICTORS`.

    Returns
    -------
    bool
    """
    predictor_class = network_predictor_class(network)
    weights_file = sparrow.data_path(predictor_class.NETWORK_PATH.format(version=predictor_class.DEFAULT_VERSION))
    return _is_parrot2_checkpoint(torch.load(weights_file, map_location="cpu"))


def needs_parrot2(name):
    """Whether a name sparrow.batch takes reads a network saved by PARROT 2.

    Parameters
    ----------
    name : str
        A name, such as ``"predictor.pLDDT2"`` or ``"FCR"``.

    Returns
    -------
    bool
    """
    if not name.startswith(PREDICTOR_PREFIX):
        return False
    method = name[len(PREDICTOR_PREFIX) :]
    return method in METHOD_NETWORKS and saved_by_parrot2(METHOD_NETWORKS[method])


def network_param(network):
    """A network as a test parameter, skipped where it needs PARROT 2 and PARROT 2 is not installed.

    Parameters
    ----------
    network : str
        One of :data:`sparrow.predictors.NETWORK_PREDICTORS`.

    Returns
    -------
    str or pytest.param
    """
    if saved_by_parrot2(network):
        return pytest.param(network, marks=requires_parrot2)
    return network


def prediction_param(name):
    """A name sparrow.batch takes as a test parameter, skipped where it needs PARROT 2 and PARROT 2 is not installed.

    Parameters
    ----------
    name : str
        A name, such as ``"predictor.pLDDT2"`` or ``"FCR"``.

    Returns
    -------
    str or pytest.param
    """
    if needs_parrot2(name):
        return pytest.param(name, marks=requires_parrot2)
    return name


def runnable_here(names):
    """The names whose values this machine can compute: all, or without PARROT 2 those not needing it.

    For a test that asks sparrow.batch for many names at once.

    Parameters
    ----------
    names : list of str
        Names sparrow.batch takes.

    Returns
    -------
    list of str
    """
    return [name for name in names if PARROT2_INSTALLED or not needs_parrot2(name)]
