import warnings
from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np

from sparrow.sparrow_exceptions import SparrowException

# How far up the stack a renamed-name warning points: past warn_renamed and the
# __getattr__ that calls it, to the code that used the old name
_RENAMED_WARNING_STACKLEVEL = 3


def validate_keyword_option(keyword, allowed_vals, keyword_name, error_message=None):
    """
    Helper function that checks a passed keyword is only one of a set of possible
    valid keywords

    Parameters
    -----------
    keyword : str
        The actual passed keyword value

    allowed_vals : list of str
        A list of possible keywords

    keyword_name : str
        the name of the keyword as the user would select it in the function call

    error_message : str, optional
        Allows the caller to pass a custom message used in place of the default
        one when the keyword is invalid. Default is None (use the default
        message).

    Returns
    --------
    None

        No return value, but raises
        :class:`sparrow.sparrow_exceptions.SparrowException` if ``keyword`` is
        not found in the ``allowed_vals`` list.

    Raises
    -------
    sparrow.sparrow_exceptions.SparrowException
        If ``keyword`` is not in ``allowed_vals``.

    """

    if keyword not in allowed_vals:
        if error_message is None:
            error_message = (
                f"Keyword {keyword_name} passed value [{keyword}], but this is not valid."
                f"\nMust be one of: {str(allowed_vals)}"
            )

        raise SparrowException(error_message)


def cache_key(*parts: Any) -> tuple[Hashable, ...]:
    """A key to cache a method's result under, from its name and arguments.

    The methods of :class:`~sparrow.protein.Protein` cache each result under
    the method's name and the arguments it was called with. An argument can be
    a list, a dict, a set or a numpy array, none of which can be part of a dict
    key, so each is replaced by a hashable equivalent: a list becomes a tuple,
    a dict a frozenset of its items, a set a frozenset, and an array its dtype,
    shape and bytes. Two calls get the same key only if their arguments are
    equal, so a cached result is never returned for different arguments.

    Parameters
    ----------
    *parts : Any
        The method's name, then its arguments, in a fixed order.

    Returns
    -------
    tuple
        The key.

    Raises
    ------
    TypeError
        When the key is used, if an argument is of some other unhashable type.
    """
    return tuple(_hashable(part) for part in parts)


def _hashable(value: Any) -> Hashable:
    """A hashable equivalent of one argument, for :func:`cache_key`.

    Parameters
    ----------
    value : Any
        The argument.

    Returns
    -------
    Hashable
        The argument itself, if it is not a list, tuple, dict, set or array;
        otherwise its hashable equivalent, with any nested values converted
        too. The type is tagged where a list and a dict could otherwise
        produce the same key.
    """
    if isinstance(value, Mapping):
        return (dict, frozenset((key, _hashable(item)) for key, item in value.items()))
    if isinstance(value, (set, frozenset)):
        return (frozenset, frozenset(_hashable(item) for item in value))
    if isinstance(value, np.ndarray):
        return (np.ndarray, value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    return value


def warn_renamed(old_name: str, new_name: str) -> None:
    """Warn that a function or method has been renamed.

    sparrow keeps an old name working for one release after a rename: the
    module's or class's ``__getattr__``, which Python calls only for a name it
    cannot otherwise find, calls this and then returns the renamed object. The
    warning points at the code that used the old name.

    Parameters
    ----------
    old_name : str
        The old name, as the caller would have written it.
    new_name : str
        The name to use instead.

    Returns
    -------
    None
        No return value; issues a ``DeprecationWarning``.
    """
    warnings.warn(
        f"{old_name} has been renamed {new_name}. The old name still works, but will be removed in a future "
        "release of sparrow.",
        DeprecationWarning,
        stacklevel=_RENAMED_WARNING_STACKLEVEL,
    )
