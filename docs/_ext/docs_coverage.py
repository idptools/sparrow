"""Check, on every build, that every public part of sparrow is documented.

Two things are checked once every page has been read, and each gap is reported
as a warning, so the strict build (``-W``) fails until it is documented:

* every public property and method of :class:`~sparrow.protein.Protein` and of
  its ``predictor``, ``polymeric``, ``nardini`` and ``visualize`` accessors has
  an entry - on the Protein pages, where each is explained;
* every public module of sparrow has a page in the API reference, apart from
  those :data:`NOT_DOCUMENTED` names, each with its reason.

A new property, method or module therefore cannot be added without the docs
saying where it lives.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import Any

from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc.mock import mock
from sphinx.util import logging

logger = logging.getLogger(__name__)

# Each class whose public members must be documented, as the docs name it: the
# module it is imported from, and its name there
DOCUMENTED_CLASSES = {
    "sparrow.protein.Protein": ("sparrow.protein", "Protein"),
    "sparrow.predictors.Predictor": ("sparrow.predictors", "Predictor"),
    "sparrow.polymer.Polymeric": ("sparrow.polymer", "Polymeric"),
    "sparrow.nardini.protein_interface.ProteinNardini": ("sparrow.nardini.protein_interface", "ProteinNardini"),
    "sparrow.visualize.protein_visualizer.ProteinVisualizer": (
        "sparrow.visualize.protein_visualizer",
        "ProteinVisualizer",
    ),
}

# Public modules the API reference leaves out, each with its reason
NOT_DOCUMENTED: dict[str, str] = {
    "sparrow.polymer.scaling_parameters": (
        "deprecated: it only forwards the old Zheng et al. names to sparrow.polymer.zheng2020, with a "
        "DeprecationWarning, and the old names are given on the polymer models page"
    ),
}

# The package whose modules are checked, and the subpackage the check skips
PACKAGE = "sparrow"
SKIPPED_PACKAGE = "sparrow.tests"


def _public_members(owner: type) -> list[str]:
    """The public properties and methods a class defines itself.

    Parameters
    ----------
    owner : type

    Returns
    -------
    list of str
    """
    names = []
    for name, member in vars(owner).items():
        if name.startswith("_"):
            continue
        if isinstance(member, property) or inspect.isfunction(member):
            names.append(name)
    return names


def _defines_anything(module: ModuleType) -> bool:
    """Whether a module has a docstring or a public name of its own, rather than only submodules.

    An empty ``__init__.py`` that only makes a directory a package has
    nothing to document.

    Parameters
    ----------
    module : types.ModuleType

    Returns
    -------
    bool
    """
    if module.__doc__ and module.__doc__.strip():
        return True
    return any(not name.startswith("_") and not inspect.ismodule(value) for name, value in vars(module).items())


def _public_modules() -> list[str]:
    """Every public module of sparrow that defines something.

    Returns
    -------
    list of str
        Dotted names, sorted, without the tests and without any module or
        package whose name starts with an underscore.
    """
    package = importlib.import_module(PACKAGE)
    names = [PACKAGE]
    for module_info in pkgutil.walk_packages(package.__path__, prefix=f"{PACKAGE}."):
        name = module_info.name
        if name == SKIPPED_PACKAGE or name.startswith(SKIPPED_PACKAGE + "."):
            continue
        if any(part.startswith("_") for part in name.split(".")):
            continue
        names.append(name)

    documentable = []
    for name in names:
        if name in NOT_DOCUMENTED:
            documentable.append(name)
            continue
        if _defines_anything(importlib.import_module(name)):
            documentable.append(name)
    return sorted(documentable)


def check_coverage(app: Sphinx, env: BuildEnvironment) -> None:
    """Warn about each public member and module that no page documents.

    Parameters
    ----------
    app : sphinx.application.Sphinx
    env : sphinx.environment.BuildEnvironment
        The environment, once every page has been read.
    """
    python_domain: Any = env.get_domain("py")
    documented_objects = set(python_domain.objects)
    documented_modules = set(python_domain.modules)

    with mock(app.config.autodoc_mock_imports):
        for documented_name, (module_name, class_name) in DOCUMENTED_CLASSES.items():
            owner = getattr(importlib.import_module(module_name), class_name)
            for member in _public_members(owner):
                if f"{documented_name}.{member}" not in documented_objects:
                    logger.warning("docs coverage: %s.%s is not documented on any page", documented_name, member)

        for module in _public_modules():
            if module in NOT_DOCUMENTED:
                continue
            if module not in documented_modules:
                logger.warning("docs coverage: module %s has no page in the API reference", module)


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the extension.

    Parameters
    ----------
    app : sphinx.application.Sphinx

    Returns
    -------
    dict
        The extension's metadata.
    """
    app.connect("env-check-consistency", check_coverage)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
