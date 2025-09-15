# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/stable/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# Incase the project was not installed
import os
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

project = "sparrow"
try:
    release = _pkg_version("sparrow")
except PackageNotFoundError:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

author = "Jeffrey Lotthammer"
copyright = f"2020–{datetime.now().year}, {author}"


# -- Project information -----------------------------------------------------

project = "sparrow"
copyright = (
    "2020, Holehouse lab. Project structure based on the "
    "Computational Molecular Science Python Cookiecutter version 1.5"
)
author = "Jeffrey Lotthammer"

# The short X.Y version
version = ""
# The full version, including alpha/beta/rc tags
release = ""


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
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

# Handle optional extensions gracefully
OPTIONAL_EXTENSIONS = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
]
for _ext in list(OPTIONAL_EXTENSIONS):
    if _ext in extensions:
        try:
            __import__(_ext)
        except ImportError:  # pragma: no cover - absence is acceptable
            extensions.remove(_ext)

autosummary_generate = True
add_module_names = False
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_attr_annotations = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": True,
    "member-order": "bysource",
}
autosummary_imported_members = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = []
# exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "default"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "navigation_depth": 4,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "sparrowdoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}


# -- Extension configuration -------------------------------------------------

# Mock heavy / optional modules if building docs online or env var set
MOCK = os.getenv("READTHEDOCS") or os.getenv("SPARROW_DOC_BUILD")
if MOCK:
    autodoc_mock_imports = [
        "afrc",
        "pyfamsa",
        "metapredict",
        "protfasta",
        "sparrow.patterning.iwd",
        "sparrow.patterning.kappa",
        "sparrow.patterning.scd",
    ]

# Suppress warnings for missing attribute docs
nitpicky = False


# -- Options for todo extension ------------------------------------------------
todo_include_todos = True

# -- Options for sphinxcontrib-bibtex extension ---------------------------------
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"
