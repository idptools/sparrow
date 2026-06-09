"""Reusable post-processing helpers for sparrow network predictor outputs.

These functions capture the handful of output shapes that the predictors share,
so each predictor's ``predict_<thing>`` method stays a one-liner over the raw
network output returned by
:meth:`sparrow.predictors.base.BaseNetworkPredictor._raw_predict`.
"""

import numpy as np

__all__ = [
    "softmax",
    "scalar_regression",
    "per_residue_class1_probability",
    "argmax_per_residue",
]


def softmax(v):
    """Numerically-plain softmax over a 1D array (matches legacy behaviour)."""
    return (np.e ** v) / np.sum(np.e ** v)


def scalar_regression(raw):
    """Return a single regression value from a many-to-one network output.

    Parameters
    ----------
    raw : numpy.ndarray
        Raw network output.

    Returns
    -------
    float
        The first (and only) flattened output value.
    """
    return raw.flatten()[0]


def per_residue_class1_probability(raw, number_of_classes, ndigits=5):
    """Return per-residue probability of class 1 from a many-to-many network.

    The raw output is reshaped to ``(-1, number_of_classes)``, a softmax is
    applied per residue, and the probability of the positive class (index 1) is
    returned, rounded to ``ndigits`` decimal places.

    Parameters
    ----------
    raw : numpy.ndarray
        Raw network output.
    number_of_classes : int
        Number of output classes per residue.
    ndigits : int, optional
        Rounding applied to each probability. Default 5.

    Returns
    -------
    list of float
        Per-residue probability of the positive class.
    """
    prediction = raw.flatten().reshape(-1, number_of_classes)
    prediction = np.array(list(map(softmax, prediction)))
    return [round(val[1], ndigits) for val in prediction]


def argmax_per_residue(raw):
    """Return the per-residue arg-max class index from a many-to-many network.

    Parameters
    ----------
    raw : numpy.ndarray
        Raw network output with a leading batch dimension.

    Returns
    -------
    list
        Per-residue ``argmax`` class indices.
    """
    return [np.argmax(row) for row in raw[0]]
