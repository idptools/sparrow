"""Deprecated: the Zheng et al. (2020) estimates are now in :mod:`sparrow.polymer.zheng2020`.

``compute_nu_zheng2020`` is :func:`sparrow.polymer.zheng2020.scaling_exponent` and
``compute_rg_zheng2020`` is :func:`sparrow.polymer.zheng2020.radius_of_gyration`.
Both old names still work, with a ``DeprecationWarning``, and will be removed in a
future release, along with this module.
"""

from __future__ import annotations

from typing import Any

from sparrow.polymer import zheng2020
from sparrow.tools.utilities import warn_renamed

# old name -> the function in sparrow.polymer.zheng2020 that replaces it
_RENAMED_FUNCTIONS = {
    "compute_nu_zheng2020": "scaling_exponent",
    "compute_rg_zheng2020": "radius_of_gyration",
}


def __getattr__(name: str) -> Any:
    """Resolve a function's old name, with a ``DeprecationWarning``.

    Parameters
    ----------
    name : str
        The attribute asked for.

    Returns
    -------
    Any
        The function that replaces ``name``.

    Raises
    ------
    AttributeError
        If ``name`` is not one of the renamed functions.
    """
    if name not in _RENAMED_FUNCTIONS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    new_name = _RENAMED_FUNCTIONS[name]
    warn_renamed(f"{__name__}.{name}", f"{zheng2020.__name__}.{new_name}")
    return getattr(zheng2020, new_name)
