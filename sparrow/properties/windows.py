"""Sliding windows along sequences.

Every windowed statistic in sparrow starts from the same thing: how much of
something each run of ``window_size`` consecutive positions holds. That is the
difference of two prefix sums, so all the windows of a whole stack of sequences
come from one cumulative sum and one subtraction, however many windows there are.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

__all__ = [
    "window_sums",
    "label_window_counts",
]


def window_sums(
    values: NDArray[np.generic],
    window_size: int,
    dtype: DTypeLike = np.int32,
    out: NDArray[np.generic] | None = None,
    prefix: NDArray[np.generic] | None = None,
) -> NDArray[Any]:
    """Sum of every window of ``window_size`` consecutive values, along the last axis.

    Parameters
    ----------
    values : numpy.ndarray
        Array of shape ``(..., n)``, typically a 0/1 indicator or a ternary
        encoding.
    window_size : int
        Window width.
    dtype : numpy dtype, optional
        Accumulator type for the prefix sum. Default ``int32``, which cannot
        overflow for any realistic sequence and moves half the memory of int64.
    out : numpy.ndarray, optional
        Array of shape ``(..., n - window_size + 1)`` to write the sums into.
    prefix : numpy.ndarray, optional
        Scratch array of shape ``(..., n + 1)`` and type ``dtype`` whose first
        column is zero, reused for the prefix sum. ``out`` and ``prefix`` let a
        caller that runs this in a hot loop avoid allocating on every call.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(..., n - window_size + 1)``: entry ``i`` is the sum of
        ``values[..., i:i + window_size]``.

    Examples
    --------
    >>> import numpy as np
    >>> window_sums(np.array([1, 0, 1, 1, 0]), 2).tolist()
    [1, 1, 2, 1]
    """
    if prefix is None:
        prefix = np.zeros(values.shape[:-1] + (values.shape[-1] + 1,), dtype=dtype)
    np.cumsum(values, axis=-1, out=prefix[..., 1:])
    result: NDArray[Any] = np.subtract(prefix[..., window_size:], prefix[..., :-window_size], out=out)
    return result


def label_window_counts(
    labels: NDArray[np.integer],
    wanted: Sequence[int],
    window_size: int,
) -> NDArray[np.int16]:
    """How many positions of each wanted label every window holds.

    Parameters
    ----------
    labels : numpy.ndarray
        Integer labels of shape ``(n_sequences, length)`` - residue types, for
        NARDINI.
    wanted : sequence of int
        The labels to count, in the order they should appear in the result.
    window_size : int
        Window width.

    Returns
    -------
    numpy.ndarray
        ``int16`` array of shape ``(n_sequences, len(wanted), length -
        window_size + 1)``.

    Notes
    -----
    Done one label at a time, reusing a single prefix-sum buffer, rather than
    one-hot encoding every label at once: that avoids allocating an
    ``(n_sequences, n_labels, length)`` cube, and lets the caller skip labels it
    does not need. ``int16`` is ample - a count cannot exceed the window - and
    halves the memory traffic of ``int32``, which matters on the NARDINI scramble
    path where this runs hundreds of thousands of times.
    """
    n_sequences, length = labels.shape
    out = np.empty((n_sequences, len(wanted), length - window_size + 1), dtype=np.int16)
    prefix = np.zeros((n_sequences, length + 1), dtype=np.int16)

    for slot, label in enumerate(wanted):
        window_sums(labels == label, window_size, out=out[:, slot, :], prefix=prefix)

    return out
