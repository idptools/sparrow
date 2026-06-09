"""Bundled AAindex property databases and a lightweight reader.

This subpackage ships three JSON databases (``aaindex1``/``aaindex2``/
``aaindex3``) and the :mod:`~sparrow.data.properties.aaindex_loader` helper for
reading them. See ``README.md`` in this directory for the file schemas.

The loader's public API is re-exported here for convenience::

    from sparrow.data.properties import load, available

    db = load("aaindex1")
    db.value("KYTJ820101", "W")
"""

from .aaindex_loader import AAindexDataset, available, load

__all__ = ["load", "available", "AAindexDataset"]
