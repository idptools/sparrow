"""Lightweight reader for the bundled AAindex property JSON files.

This module provides dependency-free access to the three JSON databases that
live alongside it:

* ``aaindex1.json`` -- single-residue indices (one scalar per amino acid).
* ``aaindex2.json`` -- pairwise amino-acid substitution / mutation matrices.
* ``aaindex3.json`` -- pairwise amino-acid contact / statistical potentials.

It only uses the standard library, resolves the data files relative to its own
location, and caches each parsed file, so it can be used both inside sparrow
and as a stand-alone helper.

Quick start
-----------
>>> from sparrow.data.properties import aaindex_loader as ai
>>> db = ai.load("aaindex1")
>>> db.value("KYTJ820101", "W")          # single-residue look-up
-0.9
>>> mats = ai.load("aaindex3")
>>> mats.pair("VILL220102", "W", "W")    # pairwise look-up
3.6095...
>>> db.metadata("KYTJ820101")["description"]
'Hydropathy index (Kyte-Doolittle, 1982)'

See ``README.md`` in this directory for the full schema of each entry.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Iterator, Optional, Union

__all__ = ["available", "load", "AAindexDataset"]

_DATA_DIR = Path(__file__).resolve().parent

#: Logical dataset name -> JSON file name.
_DATASETS = {
    "aaindex1": "aaindex1.json",
    "aaindex2": "aaindex2.json",
    "aaindex3": "aaindex3.json",
}

Number = Optional[float]
Entry = dict[str, object]


@functools.lru_cache(maxsize=None)
def _load_json(filename: str) -> dict[str, Entry]:
    """Load and cache one JSON file from this directory.

    Parameters
    ----------
    filename : str
        Bare file name (e.g. ``"aaindex1.json"``).

    Returns
    -------
    dict
        The parsed JSON object (mapping accession -> entry).
    """
    with open(_DATA_DIR / filename) as handle:
        return json.load(handle)


def available() -> list[str]:
    """Return the names of the available datasets.

    Returns
    -------
    list of str
        ``["aaindex1", "aaindex2", "aaindex3"]``.
    """
    return list(_DATASETS)


class AAindexDataset:
    """A thin, read-only view over one AAindex JSON file.

    Instances behave like a read-only mapping from accession to the raw entry
    dictionary (``len``, ``in``, iteration and ``dataset[accession]`` all work),
    and add convenience methods for value/matrix look-ups and metadata.

    Parameters
    ----------
    name : str
        One of the names returned by :func:`available`.

    Attributes
    ----------
    name : str
        The dataset name.
    is_pairwise : bool
        ``True`` for the matrix datasets (``aaindex2``/``aaindex3``), ``False``
        for the single-residue dataset (``aaindex1``).
    """

    def __init__(self, name: str) -> None:
        if name not in _DATASETS:
            raise ValueError(
                f"Unknown dataset {name!r}; choose from {available()}."
            )
        self.name = name
        self.is_pairwise = name in ("aaindex2", "aaindex3")
        self._data = _load_json(_DATASETS[name])

    # -- mapping protocol ---------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __contains__(self, accession: object) -> bool:
        return self._resolve(accession) is not None if isinstance(
            accession, str
        ) else False

    def __getitem__(self, accession: str) -> Entry:
        return self._data[self._require(accession)]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = "matrices" if self.is_pairwise else "indices"
        return f"<AAindexDataset {self.name!r}: {len(self._data)} {kind}>"

    # -- accession resolution ----------------------------------------------

    def _resolve(self, accession: str) -> Optional[str]:
        """Return the exact stored key for ``accession`` (case-insensitively)."""
        if accession in self._data:
            return accession
        lower = accession.lower()
        for key in self._data:
            if key.lower() == lower:
                return key
        return None

    def _require(self, accession: str) -> str:
        """Resolve ``accession`` or raise a helpful :class:`KeyError`."""
        resolved = self._resolve(accession)
        if resolved is None:
            raise KeyError(
                f"{accession!r} is not in {self.name}. Use "
                f"{self.name}_dataset.accessions() to list valid keys."
            )
        return resolved

    # -- listing / search ---------------------------------------------------

    def accessions(self) -> list[str]:
        """Return all accessions in the dataset, sorted.

        Returns
        -------
        list of str
        """
        return sorted(self._data)

    def list(self, search: Optional[str] = None) -> list[tuple[str, str]]:
        """List ``(accession, description)`` pairs, optionally filtered.

        Parameters
        ----------
        search : str, optional
            Case-insensitive substring; when given, only entries whose
            accession or description contains it are returned.

        Returns
        -------
        list of (str, str)
            Sorted ``(accession, description)`` tuples.
        """
        rows = [
            (acc, str(self._data[acc].get("description", "")))
            for acc in sorted(self._data)
        ]
        if search is None:
            return rows
        needle = search.lower()
        return [
            (acc, desc)
            for acc, desc in rows
            if needle in acc.lower() or needle in desc.lower()
        ]

    def metadata(self, accession: str) -> dict[str, str]:
        """Return descriptive metadata for one entry.

        Parameters
        ----------
        accession : str
            Entry accession (case-insensitive).

        Returns
        -------
        dict
            Keys ``accession``, ``description``, ``reference``, ``authors``,
            ``title``, ``journal`` and (for matrices) ``comment``. Missing
            fields are returned as empty strings.
        """
        entry = self[accession]
        fields = ["accession", "description", "reference", "authors", "title", "journal"]
        if self.is_pairwise:
            fields.append("comment")
        return {field: (entry.get(field) or "") for field in fields}

    # -- single-residue access (aaindex1) -----------------------------------

    def values(self, accession: str) -> dict[str, Number]:
        """Return the ``{amino acid: value}`` mapping for a single-residue index.

        Parameters
        ----------
        accession : str
            Entry accession (case-insensitive).

        Returns
        -------
        dict of str to (float or None)
            Per-residue values; ``None`` marks values undefined in the source
            database.

        Raises
        ------
        TypeError
            If called on a pairwise (matrix) dataset.
        """
        self._check_single("values")
        return dict(self[accession]["values"])  # type: ignore[arg-type]

    def value(self, accession: str, residue: str) -> Number:
        """Return the value of one amino acid for a single-residue index.

        Parameters
        ----------
        accession : str
            Entry accession (case-insensitive).
        residue : str
            One-letter amino-acid code.

        Returns
        -------
        float or None
        """
        return self.values(accession)[residue]

    # -- pairwise access (aaindex2 / aaindex3) ------------------------------

    def matrix(self, accession: str) -> dict[str, dict[str, Number]]:
        """Return the nested ``matrix[i][j]`` mapping for a pairwise entry.

        Parameters
        ----------
        accession : str
            Entry accession (case-insensitive).

        Returns
        -------
        dict of str to (dict of str to (float or None))
            For symmetric entries both ``matrix[i][j]`` and ``matrix[j][i]``
            are populated.

        Raises
        ------
        TypeError
            If called on the single-residue dataset.
        """
        self._check_pairwise("matrix")
        return self[accession]["matrix"]  # type: ignore[return-value]

    def pair(self, accession: str, residue_i: str, residue_j: str) -> Number:
        """Return the pairwise value for an ordered residue pair.

        Parameters
        ----------
        accession : str
            Entry accession (case-insensitive).
        residue_i, residue_j : str
            One-letter amino-acid codes (row and column respectively). Note
            some entries (e.g. ``VILL220103``) are asymmetric, so order matters.

        Returns
        -------
        float or None
        """
        return self.matrix(accession)[residue_i][residue_j]

    def is_symmetric(self, accession: str) -> bool:
        """Return whether a pairwise entry is symmetric.

        Parameters
        ----------
        accession : str
            Entry accession (case-insensitive).

        Returns
        -------
        bool
        """
        self._check_pairwise("is_symmetric")
        return bool(self[accession].get("symmetric", False))

    # -- guards -------------------------------------------------------------

    def _check_single(self, what: str) -> None:
        if self.is_pairwise:
            raise TypeError(
                f"{what}() is only valid for the single-residue dataset "
                f"'aaindex1', not {self.name!r}; use matrix()/pair() instead."
            )

    def _check_pairwise(self, what: str) -> None:
        if not self.is_pairwise:
            raise TypeError(
                f"{what}() is only valid for pairwise datasets "
                f"(aaindex2/aaindex3), not {self.name!r}; use values()/value()."
            )


@functools.lru_cache(maxsize=None)
def load(name: str) -> AAindexDataset:
    """Load (and cache) one of the AAindex datasets.

    Parameters
    ----------
    name : str
        One of ``"aaindex1"``, ``"aaindex2"`` or ``"aaindex3"``.

    Returns
    -------
    AAindexDataset
        A read-only view over the requested dataset.
    """
    return AAindexDataset(name)
