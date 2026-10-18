"""
sparrow
Next generation package for sequence parameter calculation
"""

import os
from typing import Any

# Add imports here
from sparrow.tools import io
from sparrow.tools.utilities import warn_renamed
from sparrow.protein import Protein
from sparrow.tools.io import read_fasta
from sparrow.batching import batch


# Generate _version.py if missing and in the Read the Docs environment
if os.getenv("READTHEDOCS") == "True" and not os.path.isfile("../sparrow/_version.py"):
    # versioningit is a build-time dependency ([build-system] requires), present
    # only in the Read the Docs environment - not a runtime dependency, so a type
    # checker resolving against a normal install cannot see it.
    import versioningit  # ty: ignore[unresolved-import]

    __version__ = versioningit.get_version("../")
else:
    from ._version import __version__

# code that allows access to the data directory
_ROOT = os.path.abspath(os.path.dirname(__file__))

# functions that lost their get_ prefix: old name -> new name. The old names
# still work, with a DeprecationWarning, for one release.
_RENAMED_FUNCTIONS = {"get_data": "data_path"}


def data_path(path):
    """The absolute path of a file in sparrow's bundled data directory.

    Parameters
    ----------
    path : str
        The file's path relative to ``sparrow/data``, such as
        ``"elm_classes.tsv"``. An absolute path is returned unchanged, as
        :func:`os.path.join` does.

    Returns
    -------
    str
        The absolute path. Whether the file exists is not checked.
    """
    return os.path.join(_ROOT, "data", path)


def __getattr__(name: str) -> Any:
    """Resolve a function's old ``get_`` name, with a ``DeprecationWarning``.

    Parameters
    ----------
    name : str
        The attribute asked for.

    Returns
    -------
    Any
        The renamed function.

    Raises
    ------
    AttributeError
        If ``name`` is not one of the renamed functions.
    """
    if name not in _RENAMED_FUNCTIONS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    new_name = _RENAMED_FUNCTIONS[name]
    warn_renamed(f"sparrow.{name}", f"sparrow.{new_name}")
    return globals()[new_name]
