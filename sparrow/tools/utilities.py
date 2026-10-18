import warnings

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
