# -*- coding: utf-8 -*-

import importlib
import os
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -- Project information -----------------------------------------------------

project = "sparrow"
author = "Jeffrey Lotthammer"
copyright = f"2020-{datetime.now().year}, {author}"

try:
    release = _pkg_version("sparrow")
except PackageNotFoundError:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
]

OPTIONAL_EXTENSIONS = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
]
for ext_name in OPTIONAL_EXTENSIONS:
    try:
        importlib.import_module(ext_name)
    except ImportError:  # pragma: no cover
        if ext_name in extensions:
            extensions.remove(ext_name)

autosummary_generate = True
autosummary_imported_members = True
add_module_names = False

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": True,
    "member-order": "bysource",
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_attr_annotations = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "default"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {"navigation_depth": 4}
html_static_path = ["_static"]
htmlhelp_basename = "sparrowdoc"

# -- Extension configuration -------------------------------------------------

MOCK = os.getenv("READTHEDOCS") or os.getenv("SPARROW_DOC_BUILD")
if MOCK:
    candidates = [
        "afrc",
        "pyfamsa",
        "metapredict",
        "pandas",
        "IPython",
        "protfasta",
        "sparrow.patterning.iwd",
        "sparrow.patterning.kappa",
        "sparrow.patterning.patterning",
        "sparrow.patterning.scd",
    ]
    autodoc_mock_imports = []
    for module_name in candidates:
        try:
            importlib.import_module(module_name)
        except Exception:  # pragma: no cover - only when deps are unavailable
            autodoc_mock_imports.append(module_name)

nitpicky = False
todo_include_todos = True
suppress_warnings = [
    "autodoc.mocked_object",
    "ref",
    "bibtex.duplicate_citation",
    "bibtex.duplicate_label",
]

# -- Options for sphinxcontrib-bibtex ---------------------------------------

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"
