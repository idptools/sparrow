"""The old name of :mod:`sparrow.predictors.batch_predictor`.

The module that runs the ALBATROSS networks over many sequences at once was
renamed ``batch_predictor``; the function in it is still ``batch_predict``. This
name still works, with a ``DeprecationWarning`` naming the new one, for one
release.
"""

from typing import Any

from sparrow.predictors import batch_predictor
from sparrow.tools.utilities import warn_renamed

_OLD_MODULE = "sparrow.predictors.batch_predict"
_NEW_MODULE = "sparrow.predictors.batch_predictor"


def __getattr__(name: str) -> Any:
    """Hand back an attribute of the renamed module, with a ``DeprecationWarning``.

    Parameters
    ----------
    name : str
        The attribute asked for, such as ``"batch_predict"``.

    Returns
    -------
    Any
        The attribute of :mod:`sparrow.predictors.batch_predictor`.

    Raises
    ------
    AttributeError
        If the renamed module has no such attribute either.
    """
    if not hasattr(batch_predictor, name):
        raise AttributeError(f"module {_OLD_MODULE!r} has no attribute {name!r}")
    warn_renamed(f"{_OLD_MODULE}.{name}", f"{_NEW_MODULE}.{name}")
    return getattr(batch_predictor, name)
