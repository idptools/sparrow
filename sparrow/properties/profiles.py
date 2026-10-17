"""Linear profiles: a property computed along a sequence, one window at a time.

A profile slides a window of ``window_size`` residues along a sequence, one
residue at a time, and scores every window, so it shows where along the sequence
a property is high or low. A sequence of ``length`` residues has ``length -
window_size + 1`` windows.

* :func:`window_profile` scores the windows with any batch property in
  :mod:`sparrow.properties` - every window of every sequence is handed to the
  property function as one batch, so profiling a property costs about as much as
  computing it for that many short sequences;
* :func:`fraction_profile` and :func:`net_charge_profile` are the local fraction
  of a set of residues and the local net charge per residue, taken straight from
  window sums (:func:`~sparrow.properties.windows.window_sums`) rather than by
  counting every window separately;
* :func:`scale_profile` is the local mean of a per-residue scale.

Each window's value is exactly what the property gives for that window as a
sequence of its own: the fractions and the net charge are an integer window sum
divided by the window size, which is how the composition functions compute them.

End modes
---------
``end_mode`` decides what the profile holds at the ends of the sequence, where no
window is centred:

* ``'extend-ends'`` (the default) - the first window's value fills the first
  ``window_size // 2`` positions and the last window's value the remaining ones
  at the end, so there is one value per residue and each window's value sits at
  its centre;
* ``'zero-ends'`` - the same positions, filled with zeros;
* ``''`` - no filling: one value per window.

Batches
-------
Every function takes a batch and returns a list with one profile per sequence -
a list rather than one array, because sequences of different lengths have
profiles of different lengths. Sequences of one length are scored together, and
each window's value does not depend on what else is in the batch.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

from sparrow.data import amino_acids
from sparrow.properties import composition, encoding
from sparrow.properties.windows import window_sums
from sparrow.sparrow_exceptions import SparrowException

__all__ = [
    "DEFAULT_PROFILE_WINDOW",
    "EXTEND_ENDS",
    "ZERO_ENDS",
    "NO_ENDS",
    "END_MODES",
    "SAVGOL_POLYNOMIAL_ORDER",
    "window_profile",
    "fraction_profile",
    "net_charge_profile",
    "scale_profile",
]

#: Window width, in residues, when none is given - as in ``Protein``'s linear
#: profiles.
DEFAULT_PROFILE_WINDOW = 8

#: The end modes (see the module docstring).
EXTEND_ENDS = "extend-ends"
ZERO_ENDS = "zero-ends"
NO_ENDS = ""
END_MODES = (EXTEND_ENDS, ZERO_ENDS, NO_ENDS)

#: Order of the polynomial a Savitzky-Golay filter fits within each smoothing
#: window. sparrow's profiles have always been smoothed with a cubic.
SAVGOL_POLYNOMIAL_ORDER = 3

#: Most residues (windows times window size) :func:`window_profile` hands to the
#: property function in one call. It only bounds memory - a batch of thousands of
#: sequences, or one very long sequence with a wide window, has millions of
#: window residues - and does not change any value, because every property
#: function gives each window the same value in a batch of any size.
_WINDOW_RESIDUES_PER_CALL = 2**20

# the formal charge of each residue code: +1 for R and K, -1 for D and E
_IS_POSITIVE = encoding.residue_membership(amino_acids.POS).astype(np.int32)
_IS_NEGATIVE = encoding.residue_membership(amino_acids.NEG).astype(np.int32)
_FORMAL_CHARGE = _IS_POSITIVE - _IS_NEGATIVE

#: What a property function receives (every window, as one batch) and returns
#: (one value per window).
PropertyFunction = Callable[[encoding.EncodedBatch], Any]

#: Scores one ``(k, length)`` group of sequences: one row of per-window values
#: for each sequence.
_BlockScorer = Callable[[NDArray[np.int8]], NDArray[np.float64]]


# =====================================================================
# checking arguments
# =====================================================================


def _is_whole_number(value: object) -> bool:
    """True for an integer (Python or NumPy), but not for a bool."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _check_arguments(window_size: int, end_mode: str, smooth: int | None) -> None:
    """Reject a window size, end mode or smoothing window that cannot be used.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``window_size`` is not a whole number of at least 1, ``end_mode`` is
        not one of :data:`END_MODES`, or ``smooth`` is neither None nor a whole
        number. A bool is not taken as a number, so ``smooth=True`` is an error
        rather than a smoothing window of 1.
    """
    if not _is_whole_number(window_size) or window_size < 1:
        raise SparrowException(f"window_size must be a whole number of residues, at least 1, not {window_size!r}")
    if end_mode not in END_MODES:
        raise SparrowException(f"end_mode must be one of {list(END_MODES)}, not {end_mode!r}")
    if smooth is not None and not _is_whole_number(smooth):
        raise SparrowException(f"smooth must be None or the smoothing window, a whole number, not {smooth!r}")


def _reject_short_sequences(lengths: NDArray[np.int64], window_size: int) -> None:
    """Raise for the first sequence in a batch that is shorter than the window.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        Always: the first sequence shorter than ``window_size``, which has no
        window at all, by its index in the batch.
    """
    index = int(np.flatnonzero(lengths < window_size)[0])
    message = f"Window size [{window_size}] is larger than sequence [{int(lengths[index])}]"
    if lengths.size > 1:
        message = f"Sequence {index} of {lengths.size} in the batch: {message}"
    raise SparrowException(message)


# =====================================================================
# from per-window values to a profile
# =====================================================================


def _fill_ends(per_window: NDArray[np.float64], length: int, window_size: int, end_mode: str) -> NDArray[np.float64]:
    """Pad one sequence's per-window values out to its length, as ``end_mode`` says."""
    if end_mode == NO_ENDS:
        return per_window.copy()

    # each window's value sits at the window's centre
    front = window_size // 2
    back = length - front - per_window.size
    if end_mode == EXTEND_ENDS:
        front_value, back_value = per_window[0], per_window[-1]
    else:
        front_value = back_value = 0.0

    profile: NDArray[np.float64] = np.concatenate([np.full(front, front_value), per_window, np.full(back, back_value)])
    return profile


def _smooth(profile: NDArray[np.float64], smooth: int) -> NDArray[np.float64]:
    """Savitzky-Golay smoothing over ``smooth`` positions, with a cubic.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the filter cannot use that window - it must be longer than the
        polynomial order and no longer than the profile.
    """
    try:
        smoothed: NDArray[np.float64] = savgol_filter(profile, smooth, SAVGOL_POLYNOMIAL_ORDER)
    except ValueError as error:
        raise SparrowException(
            f"Cannot smooth a profile of {profile.size} values with a window of {smooth}: {error}"
        ) from error
    return smoothed


def _profiles(
    sequences: str | Sequence[str] | encoding.EncodedBatch,
    window_size: int,
    end_mode: str,
    smooth: int | None,
    score: _BlockScorer,
) -> list[NDArray[np.float64]]:
    """Score every window of every sequence, then fill each profile's ends and smooth it.

    Parameters
    ----------
    sequences : str, sequence of str, or EncodedBatch
        The batch.
    window_size : int
        Window width.
    end_mode : str
        One of :data:`END_MODES`.
    smooth : int or None
        Smoothing window, or None for no smoothing.
    score : callable
        Takes the residue codes of a group of equal-length sequences, shape
        ``(k, length)``, and returns their per-window values, shape ``(k,
        length - window_size + 1)``.

    Returns
    -------
    list of numpy.ndarray
        One profile per sequence, in batch order.
    """
    _check_arguments(window_size, end_mode, smooth)
    batch = encoding.encode_batch(sequences)

    # the groups come shortest first, so the first one says whether any
    # sequence is too short for the window
    groups = batch.by_length()
    if groups and groups[0][1].shape[1] < window_size:
        _reject_short_sequences(batch.lengths, window_size)

    profiles: list[NDArray[np.float64]] = [np.empty(0)] * len(batch)
    for indices, codes in groups:
        length = codes.shape[1]
        per_window = score(codes)
        for row, index in enumerate(indices.tolist()):
            profile = _fill_ends(per_window[row], length, window_size, end_mode)
            profiles[index] = profile if smooth is None else _smooth(profile, smooth)
    return profiles


# =====================================================================
# scoring windows
# =====================================================================


def _window_means(per_residue: NDArray[np.generic], window_size: int) -> NDArray[np.float64]:
    """Each window's sum of an integer per-residue value, divided by the window size.

    The sum is exact, so this is one rounding: the same value
    :mod:`~sparrow.properties.composition` gives for a fraction of that window.
    """
    means: NDArray[np.float64] = window_sums(per_residue, window_size) / window_size
    return means


def _score_every_window(
    codes: NDArray[np.int8], property_function: PropertyFunction, window_size: int
) -> NDArray[np.float64]:
    """``property_function`` of every window of a ``(k, length)`` group of sequences.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the property function does not return one value per window.
    """
    n_sequences, length = codes.shape
    n_windows = length - window_size + 1
    n_total = n_sequences * n_windows
    windows_per_call = max(1, _WINDOW_RESIDUES_PER_CALL // window_size)
    within_window = np.arange(window_size)

    # the windows of the group are numbered sequence by sequence, and scored in
    # runs of consecutive numbers, a run being free to span sequences
    scores = np.empty(n_total, dtype=np.float64)
    for first in range(0, n_total, windows_per_call):
        window_ids = np.arange(first, min(first + windows_per_call, n_total))
        sequence_index, window_start = np.divmod(window_ids, n_windows)
        windows = codes[sequence_index[:, None], window_start[:, None] + within_window]

        window_batch = encoding.EncodedBatch(
            codes=windows.reshape(-1), lengths=np.full(window_ids.size, window_size, dtype=np.int64)
        )
        values = np.asarray(property_function(window_batch))
        if values.shape != (window_ids.size,):
            raise SparrowException(
                f"A profile's property function must return one value per window: it was given "
                f"{window_ids.size} windows and returned an array of shape {values.shape}"
            )
        scores[window_ids] = values
    return scores.reshape(n_sequences, n_windows)


# =====================================================================
# profiles
# =====================================================================


def window_profile(
    sequences: str | Sequence[str] | encoding.EncodedBatch,
    property_function: PropertyFunction,
    window_size: int = DEFAULT_PROFILE_WINDOW,
    end_mode: str = EXTEND_ENDS,
    smooth: int | None = None,
) -> list[NDArray[np.float64]]:
    """Profile of any batch property along each sequence.

    Every window of every sequence is passed to ``property_function`` as one
    :class:`~sparrow.properties.encoding.EncodedBatch`, and its value for each
    window becomes that window's point on the profile. Any batch function in
    :mod:`sparrow.properties` that accepts an ``EncodedBatch`` will do - for
    example :func:`~sparrow.properties.composition.fraction_charged`,
    :func:`~sparrow.properties.composition.seg_complexity` or
    :func:`~sparrow.properties.patterning.kappa` - and one that needs more
    arguments can be wrapped in a ``lambda``.

    Parameters
    ----------
    sequences : str, sequence of str, or EncodedBatch
        The batch.
    property_function : callable
        Takes an ``EncodedBatch`` of windows and returns one number per window.
    window_size : int, optional
        Window width, in residues. Default 8.
    end_mode : {'extend-ends', 'zero-ends', ''}, optional
        How the ends of each profile are filled (see the module docstring).
        Default ``'extend-ends'``.
    smooth : int or None, optional
        If given, the profile is smoothed with a Savitzky-Golay filter (a cubic)
        over this many positions, after its ends are filled. Default None.

    Returns
    -------
    list of numpy.ndarray
        One ``float64`` profile per sequence: as long as the sequence, or one
        value per window when ``end_mode`` is ``''``.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If a sequence is shorter than the window or contains a non-standard
        residue, an argument is invalid, the profile cannot be smoothed with
        ``smooth``, or ``property_function`` does not return one value per
        window.

    Examples
    --------
    >>> from sparrow.properties import composition
    >>> window_profile("EEEEGGGGKKKK", composition.net_charge_per_residue, 4, end_mode="")[0].tolist()
    [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    """
    return _profiles(
        sequences,
        window_size,
        end_mode,
        smooth,
        lambda codes: _score_every_window(codes, property_function, window_size),
    )


def fraction_profile(
    sequences: str | Sequence[str] | encoding.EncodedBatch,
    residues: Sequence[str],
    window_size: int = DEFAULT_PROFILE_WINDOW,
    end_mode: str = EXTEND_ENDS,
    smooth: int | None = None,
) -> list[NDArray[np.float64]]:
    """Profile of the local fraction of a set of residues along each sequence.

    Each window's value is :func:`~sparrow.properties.composition.fraction_in_set`
    of that window.

    Parameters
    ----------
    sequences : str, sequence of str, or EncodedBatch
        The batch.
    residues : sequence of str
        The residues to count. An empty set gives a profile of zeros.
    window_size, end_mode, smooth
        As for :func:`window_profile`.

    Returns
    -------
    list of numpy.ndarray
        One ``float64`` profile per sequence, each value between 0 and 1.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If a residue in ``residues`` is not a standard amino acid, and as for
        :func:`window_profile`.

    Examples
    --------
    >>> fraction_profile("QQQQAAAA", ["Q"], 4, end_mode="")[0].tolist()
    [1.0, 0.75, 0.5, 0.25, 0.0]
    """
    member = encoding.residue_membership(residues)
    return _profiles(
        sequences,
        window_size,
        end_mode,
        smooth,
        lambda codes: _window_means(member[codes], window_size),
    )


def net_charge_profile(
    sequences: str | Sequence[str] | encoding.EncodedBatch,
    window_size: int = DEFAULT_PROFILE_WINDOW,
    end_mode: str = EXTEND_ENDS,
    smooth: int | None = None,
) -> list[NDArray[np.float64]]:
    """Profile of the local net charge per residue (NCPR) along each sequence.

    Each window's value is
    :func:`~sparrow.properties.composition.net_charge_per_residue` of that
    window: its R and K count minus its D and E count, over the window size.

    Parameters
    ----------
    sequences : str, sequence of str, or EncodedBatch
        The batch.
    window_size, end_mode, smooth
        As for :func:`window_profile`.

    Returns
    -------
    list of numpy.ndarray
        One ``float64`` profile per sequence, each value between -1 and +1.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        As for :func:`window_profile`.

    Examples
    --------
    >>> net_charge_profile("EEEEGGGGKKKK", 4, end_mode="")[0].tolist()
    [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    """
    return _profiles(
        sequences,
        window_size,
        end_mode,
        smooth,
        lambda codes: _window_means(_FORMAL_CHARGE[codes], window_size),
    )


def scale_profile(
    sequences: str | Sequence[str] | encoding.EncodedBatch,
    scale: Mapping[str, float | None],
    window_size: int = DEFAULT_PROFILE_WINDOW,
    end_mode: str = EXTEND_ENDS,
    smooth: int | None = None,
) -> list[NDArray[np.float64]]:
    """Profile of the local mean of a per-residue scale along each sequence.

    Each window's value is :func:`~sparrow.properties.composition.mean_scale` of
    that window.

    Parameters
    ----------
    sequences : str, sequence of str, or EncodedBatch
        The batch.
    scale : mapping of str to float
        Value of each residue, e.g. :data:`sparrow.data.amino_acids.AA_hydro_KD`
        or an AAindex index from :func:`sparrow.data.aaindex.get_property_values`.
    window_size, end_mode, smooth
        As for :func:`window_profile`.

    Returns
    -------
    list of numpy.ndarray
        One ``float64`` profile per sequence.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If a sequence contains a residue the scale has no value for, and as for
        :func:`window_profile`.

    Examples
    --------
    >>> scale_profile("AAGG", {"A": 1.0, "G": 0.0}, 2, end_mode="")[0].tolist()
    [1.0, 0.5, 0.0]
    """
    return window_profile(
        sequences,
        lambda windows: composition.mean_scale(windows, scale),
        window_size,
        end_mode,
        smooth,
    )
