"""Checks of the arguments every part of a figure is made with.

Each raises a :class:`~sparrow.sparrow_exceptions.SparrowException` that says
what is wrong.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from matplotlib.colors import is_color_like

from sparrow.sparrow_exceptions import SparrowException


def _check_position(value: object, what: str) -> int:
    """Check that a residue position is a whole number from 1.

    Parameters
    ----------
    value : object
        The position.
    what : str
        What the position is, for the error message.

    Returns
    -------
    int
        The position.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``value`` is not a whole number from 1.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise SparrowException(f"{what} must be a residue position, a whole number from 1, not {value!r}.")
    return int(value)


def _check_region(start: object, end: object, what: str) -> None:
    """Check that ``start`` and ``end`` are residue positions, and that ``end`` is not before ``start``.

    Parameters
    ----------
    start, end : object
        The first and last residue of the region.
    what : str
        What the region is, for the error message.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If either is not a residue position, or ``end`` is before ``start``.
    """
    first = _check_position(start, f"The start of {what}")
    last = _check_position(end, f"The end of {what}")
    if last < first:
        raise SparrowException(f"The end of {what} ({last}) is before its start ({first}).")


def _check_positive(value: object, what: str) -> float:
    """Check that a size is a number greater than zero.

    Parameters
    ----------
    value : object
        The size.
    what : str
        What the size is, for the error message.

    Returns
    -------
    float
        The size.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``value`` is not a finite number greater than zero.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value) or value <= 0:
        raise SparrowException(f"{what} must be a number greater than 0, not {value!r}.")
    return float(value)


def _check_count(value: object, what: str) -> int:
    """Check that a count is a whole number from 1.

    Parameters
    ----------
    value : object
        The count.
    what : str
        What is counted, for the error message.

    Returns
    -------
    int
        The count.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``value`` is not a whole number from 1.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise SparrowException(f"{what} must be a whole number from 1, not {value!r}.")
    return int(value)


def _check_color(color: object, what: str) -> None:
    """Check that a color is one matplotlib can draw.

    Parameters
    ----------
    color : object
        The color: a name such as ``'red'``, a hex string such as
        ``'#ffe699'``, or anything else matplotlib accepts.
    what : str
        What the color is for, for the error message.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If matplotlib cannot draw ``color``.
    """
    if not is_color_like(color):
        raise SparrowException(
            f"{what} must be a color matplotlib can draw, such as 'red' or '#ffe699', not {color!r}."
        )


def _check_limits(limits: object, what: str) -> tuple[float, float]:
    """Check that a range of values is two finite numbers, the first below the second.

    Parameters
    ----------
    limits : object
        The range, as ``(lower, upper)``.
    what : str
        What the range is for, for the error message.

    Returns
    -------
    tuple of float
        ``(lower, upper)``.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``limits`` is not two finite numbers with the first below the second.
    """
    problem = SparrowException(
        f"{what} must be (lower, upper), two finite numbers with lower below upper, not {limits!r}."
    )
    if not isinstance(limits, Sequence) or isinstance(limits, str) or len(limits) != 2:
        raise problem
    lower, upper = limits
    for value in (lower, upper):
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise problem
    if not lower < upper:
        raise problem
    return float(lower), float(upper)


def _first_overlap(start: int, end: int, regions: Sequence[tuple[int, int]]) -> tuple[int, int] | None:
    """The first of ``regions`` that shares a residue with ``start`` to ``end``, if any.

    Parameters
    ----------
    start, end : int
        The first and last residue of the new region.
    regions : sequence of tuple of int
        The ``(start, end)`` of each region already there.

    Returns
    -------
    tuple of int or None
        The ``(start, end)`` of the first region that overlaps, or None.
    """
    for other_start, other_end in regions:
        if start <= other_end and end >= other_start:
            return other_start, other_end
    return None


def _per_residue_values(values: npt.ArrayLike, what: str) -> npt.NDArray[np.float64]:
    """Check that per-residue values are numbers, finite or NaN, not all NaN.

    Parameters
    ----------
    values : array_like
        The values, one for each residue, or for a grid, one row of them for
        each row of the grid.
    what : str
        What the values are, for the error message.

    Returns
    -------
    numpy.ndarray
        The values, as floats.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``values`` are not numbers, are empty, include an infinite value,
        or are all NaN.
    """
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise SparrowException(f"{what} must be numbers, one for each residue.") from error
    if array.size == 0:
        raise SparrowException(f"{what} are empty: there must be one for each residue.")
    if np.isinf(array).any():
        raise SparrowException(f"{what} include an infinite value. Leave a residue with no value as NaN.")
    if np.isnan(array).all():
        raise SparrowException(f"{what} are all NaN, so there is nothing to draw.")
    return array
