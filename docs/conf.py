# -*- coding: utf-8 -*-

import importlib
import os
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

import versioningit
from sphinx.ext.autodoc.mock import mock
from sphinx.util import logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -- Project information -----------------------------------------------------

project = "sparrow"
author = "Alex Holehouse, Ryan Emenecker, Jeffrey Lotthammer"
copyright = f"2020-{datetime.now().year}, {author}"

# The version comes from the git tags through versioningit - the same source as
# sparrow.__version__ - so the docs always name the commit they were built from.
# Read the Docs does not install sparrow, so the installed package's metadata is
# only a fallback, for a copy of the docs built outside a git checkout.
DISTRIBUTION_NAME = "idptools-sparrow"
try:
    release = versioningit.get_version(ROOT)
except Exception as error:
    try:
        release = _pkg_version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        release = "unknown"
    logger.warning("versioningit could not read the version (%s); using %s", error, release)
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

# On Read the Docs only the doc requirements are installed, so sparrow's heavier
# runtime dependencies are absent. Each candidate is import-tested and mocked
# only if it actually fails, which keeps local builds (where everything is
# installed) documenting the real objects.
#
# Only third-party packages belong on this list, never sparrow's own modules: a
# sparrow module fails this plain import test whenever one of these packages is
# missing, and a mocked module is documented as empty. With these mocked, autodoc
# imports sparrow's modules for real.
MOCK = os.getenv("READTHEDOCS") or os.getenv("SPARROW_DOC_BUILD")
if MOCK:
    candidates = [
        "afrc",
        "pyfamsa",
        "metapredict",
        "pandas",
        "IPython",
        "protfasta",
        "torch",
        "tqdm",
        "parrot",
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
    # Sphinx 9 warns when a type word in a docstring ("sequence of str") matches
    # several documented attributes of that name (Protein.sequence, ELM.sequence,
    # ...). It is a word, not a link, so the ambiguity is harmless; a reference
    # that resolves to nothing is a different warning and is still reported.
    "ref.python",
    "bibtex.duplicate_citation",
    "bibtex.duplicate_label",
]

# numpy-style docstrings describe types in prose ("int, optional", "sequence of
# str", "{'raise', 'skip'}"), and napoleon turns each word into a cross-reference.
# None of these name an object, so the strict build (-n) ignores them, along with
# third-party types that have no intersphinx inventory here. Any other unresolved
# reference is a broken link, and the strict build reports it.
_TYPE_PROSE = [
    r"optional",
    r"default .*",
    r"sequence",
    r"iterable",
    r"mapping",
    r"callable",
    r"array_like",
    r"arrays?",
    r"str to .*",
    r".* object",
    r"numpy dtype",
    # the pieces of a set of allowed values, such as {'raise', 'skip'}
    r"\{.*",
    r".*\}",
    r"'.*'",
    r'".*"',
]
_UNDOCUMENTED_EXTERNAL_TYPES = [
    r"NDArray",
    r"np\..*",
    r"numpy\..*",
    r"pandas\..*",
    r"torch\..*",
    r"pyfamsa\..*",
    r"Alignment",
]
nitpick_ignore_regex = [("py:class", pattern) for pattern in _TYPE_PROSE + _UNDOCUMENTED_EXTERNAL_TYPES]

# -- Options for sphinxcontrib-bibtex ---------------------------------------

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "unsrt"
bibtex_reference_style = "author_year"


# -- Auto-generated AAindex property index reference table ------------------
def _generate_property_index_table(app):
    """Write the AAindex property reference table included by api_guides/properties.

    Regenerated on every build so the table always matches the shipped database.
    Importing sparrow needs the same third-party mocks as autodoc on Read the Docs.
    """
    try:
        with mock(app.config.autodoc_mock_imports):
            from sparrow.data import aaindex

            rows = aaindex.list_property_indices()
    except Exception as error:  # pragma: no cover - warn, but keep the rest of the docs building
        logger.warning("Could not build the AAindex property table: %s", error)
        return

    def clean(text):
        """Flatten to one line and escape the RST characters that appear in AAindex text."""
        text = str(text or "").replace("\n", " ").strip()
        # a trailing underscore makes RST try to resolve a reference; backticks and
        # pipes break table cells
        return text.replace("\\", "").replace("_", r"\_").replace("|", r"\|").replace("`", "'")

    out_path = Path(app.srcdir) / "api_guides" / "_property_index_table.rst"
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 26 11 38 25",
        "",
        "   * - Identifier",
        "     - Accession",
        "     - Description",
        "     - Source",
    ]
    for identifier, accession, description in sorted(rows):
        # the citation makes an index interpretable; the description alone often
        # does not say which of several similar scales this is
        try:
            metadata = aaindex.get_property_metadata(identifier)
        except Exception:  # pragma: no cover - never fail the build over metadata
            metadata = {}

        source = clean(metadata.get("journal"))
        reference = str(metadata.get("reference") or "")
        if reference.startswith("PMID:"):
            pmid = reference.split(":", 1)[1].strip()
            if pmid.isdigit():
                link = f"`PMID {pmid} <https://pubmed.ncbi.nlm.nih.gov/{pmid}/>`__"
                source = f"{source} {link}" if source else link

        lines.append(f"   * - ``{identifier}``")
        lines.append(f"     - {accession}")
        lines.append(f"     - {clean(description)}")
        lines.append(f"     - {source or '--'}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def setup(app):
    app.connect("builder-inited", _generate_property_index_table)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
