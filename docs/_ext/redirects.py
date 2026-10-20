"""Keep the addresses of pages that have moved working, by redirecting each to its new page.

The documentation was reorganized into sections, which moved most pages, and
links to the old addresses - from papers, notebooks and bookmarks - would
otherwise lead nowhere. After an HTML build, this writes a small page at each
old address that sends the reader on to the new one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)

# Each old page, as a path under the HTML output, and the page it moved to, as a
# path from there (with an anchor, for a page that became part of another)
MOVED_PAGES = {
    "installation.html": "overview/installation.html",
    "predictors.html": "adding_a_predictor.html",
    "plugins.html": "writing_a_plugin.html",
    "api_guides/protein.html": "protein/index.html",
    "api_guides/many_sequences.html": "batch/index.html",
    "api_guides/batch.html": "power_user/networks.html",
    "api_guides/sequence_properties.html": "power_user/properties.html",
    "api_guides/patterning.html": "power_user/patterning.html",
    "api_guides/properties.html": "power_user/aaindex.html",
    "api_guides/nardini.html": "power_user/nardini.html",
    "api_guides/sequence_annotations.html": "protein/annotations.html",
    "api_guides/sequence_analysis.html": "power_user/index.html",
}

# The patterning functions each had a page of their own, and are now documented
# together in the API reference
PATTERNING_FUNCTIONS = (
    "kappa",
    "kappa_x",
    "sigma",
    "delta",
    "patterning_percentile",
    "iwd",
    "iwd_charged_weighted",
    "bivariate_iwd_charged_weighted",
    "scd",
    "shd",
)

REDIRECT_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>This page has moved</title>
<link rel="canonical" href="{target}">
<meta http-equiv="refresh" content="0; url={target}">
</head>
<body>
<p>This page has moved to <a href="{target}">{target}</a>.</p>
</body>
</html>
"""


def moved_pages() -> dict[str, str]:
    """Every old page, mapped to where it moved.

    Returns
    -------
    dict
        :data:`MOVED_PAGES`, and each patterning function's old page, mapped to
        its entry in the API reference.
    """
    pages = dict(MOVED_PAGES)
    for function in PATTERNING_FUNCTIONS:
        name = f"sparrow.properties.patterning.{function}"
        pages[f"api_guides/generated/{name}.html"] = f"api/properties.html#{name}"
    return pages


def _relative_target(old_page: str, new_page: str) -> str:
    """The new page's address, relative to the old page's directory.

    Parameters
    ----------
    old_page : str
        The old page, as a path under the HTML output.
    new_page : str
        The new page, as a path under the HTML output, with any anchor.

    Returns
    -------
    str
        Such as ``"../protein/index.html"`` for an old page in a subdirectory.
    """
    depth = old_page.count("/")
    return "../" * depth + new_page


def write_redirects(app: Sphinx, exception: Exception | None) -> None:
    """Write a redirect page at each old address, after a successful HTML build.

    Parameters
    ----------
    app : sphinx.application.Sphinx
    exception : Exception or None
        The exception the build failed with, if it did.
    """
    if exception is not None or app.builder.format != "html":
        return

    output = Path(app.outdir)
    for old_page, new_page in moved_pages().items():
        new_file = new_page.split("#")[0]
        if not (output / new_file).exists():
            logger.warning("redirects: %s should move to %s, which was not built", old_page, new_file)
            continue
        path = output / old_page
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(REDIRECT_PAGE.format(target=_relative_target(old_page, new_page)), encoding="utf-8")


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
    app.connect("build-finished", write_redirects)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
