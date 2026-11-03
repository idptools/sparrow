"""Tilting a background's grammar towards a sequence's composition.

The background models draw a null from the background's k-mers, or its
Markov chain, given that the null has exactly the sequence's composition.
Over a long sequence, conditioning on the composition like this changes the
background in one simple way, by the conditional limit theorem (Cover &
Thomas 2006, section 11.6, and Csiszar, Cover & Choi 1987 for Markov chains):
each residue ``a`` gets a weight ``t(a)``, the same wherever it is, and
everything the background draws is weighted by the product of the weights of
its residues. The weights are those that make the tilted background's
composition the sequence's.

Written ``t(a) = exp(theta(a))``, the weights are the ``theta`` that minimize
``A(theta) - theta . q``, where ``q`` is the sequence's composition and
``A(theta)`` is the log of the tilted background's normalizer per residue - so
``A`` is convex, and its gradient is the tilted background's composition. The
minimum is where the two compositions match, and it is found with L-BFGS.

References
----------
Cover TM, Thomas JA (2006) *Elements of Information Theory*, 2nd ed. Wiley.

Csiszar I, Cover TM, Choi BS (1987) Conditional limit theorems under Markov
conditioning. *IEEE Trans Inf Theory* 33:788-801.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from sparrow.sparrow_exceptions import NullModelWarning

__all__ = ["TILT_TOLERANCE", "fit_tilt"]

#: How closely the tilted background's composition must match the sequence's,
#: as the largest relative difference for any residue. The nulls have the
#: sequence's composition exactly whatever the tilt; the tilt only keeps the
#: residues placed early in a null in the same proportions as those placed
#: late, and a difference of a tenth of a percent is far below what any one
#: null can show.
TILT_TOLERANCE = 1e-3

# the most L-BFGS iterations allowed; with the yeast proteome as the
# background, it takes 14 to 34
_MAX_ITERATIONS = 1000

# How far the log weights may move from where they start, the tilt a
# background with no grammar would need: a million-fold either way. A
# composition that needs more is one the background cannot reach, and the
# bound keeps the arithmetic finite while the warning says how close it came.
_LARGEST_LOG_TILT_CHANGE = float(np.log(1e6))

LogNormalizer = Callable[[NDArray[np.float64]], tuple[float, NDArray[np.float64]]]


def fit_tilt(
    log_normalizer: LogNormalizer, composition: NDArray[np.float64], start: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The log weights that tilt a background to a composition.

    Parameters
    ----------
    log_normalizer : callable
        Takes log weights ``theta``, one for each residue in ``composition``,
        and returns ``A(theta)``, the log of the tilted background's
        normalizer per residue, and the tilted background's composition: the
        gradient of ``A``.
    composition : numpy.ndarray
        The composition to tilt to, as fractions summing to 1, every one
        positive.
    start : numpy.ndarray
        The log weights to start from.

    Returns
    -------
    numpy.ndarray
        The log weights ``theta``, one for each residue in ``composition``.
        Adding the same number to all of them gives the same tilt.

    Warns
    -----
    sparrow.sparrow_exceptions.NullModelWarning
        If the tilted composition is still further than
        :data:`TILT_TOLERANCE` from ``composition`` when L-BFGS stops.
    """

    def objective(theta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        log_total, tilted = log_normalizer(theta)
        return log_total - float(theta @ composition), tilted - composition

    # stopping once no residue's fraction is off by more than the tolerance
    # times the smallest fraction keeps every residue within the tolerance
    gradient_tolerance = TILT_TOLERANCE * float(np.min(composition))
    bounds = [(value - _LARGEST_LOG_TILT_CHANGE, value + _LARGEST_LOG_TILT_CHANGE) for value in start]
    result = minimize(
        objective,
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"gtol": gradient_tolerance, "ftol": 0.0, "maxiter": _MAX_ITERATIONS},
    )

    _, tilted = log_normalizer(result.x)
    largest_difference = float(np.max(np.abs(tilted / composition - 1)))
    if largest_difference > TILT_TOLERANCE:
        warnings.warn(
            f"The background could only be tilted to within {largest_difference:.1%} of the sequence's composition "
            f"(the aim is {TILT_TOLERANCE:.1%}), so the residues placed early in each null may not be in quite the "
            "same proportions as those placed late. The nulls still have the sequence's composition.",
            NullModelWarning,
            stacklevel=5,
        )
    return np.asarray(result.x, dtype=np.float64)
