"""Minimal labelled containers for NARDINI results.

NARDINI produces named numbers - 90 features per sequence. These two containers
do one thing: keep the names attached to a plain NumPy array so you can say
``v["Frac K+R"]`` instead of remembering that it is index 63.

* :class:`FeatureVector` - one labelled row (a single sequence's features).
* :class:`FeatureTable` - many labelled rows (a set of sequences).

They are deliberately thin. The values are always reachable as
:attr:`~FeatureTable.values`, so any real analysis - filtering, sorting,
reshaping, plotting, export - happens with NumPy or pandas outside sparrow:

.. code-block:: python

    table = result.to_table()

    table.values                     # (n_sequences, 90) ndarray - do as you like
    table.column_names               # the matching labels
    table["pol-neg"]                 # one feature across all sequences
    table.row("IDR1")["Frac K+R"]    # one value
    table.to_dict()                  # {row: {feature: value}}

    table.to_dataframe()             # if you would rather have pandas
"""

import numpy as np

from sparrow.sparrow_exceptions import SparrowException

__all__ = ["FeatureVector", "FeatureTable"]


def _index_of(name, available, what="feature"):
    """Return the position of ``name``, with a helpful error if it is absent."""
    try:
        return list(available).index(name)
    except ValueError:
        preview = ", ".join(list(available)[:6])
        raise SparrowException(f"Unknown {what} '{name}'. Available {what}s start: {preview} ...") from None


class FeatureVector:
    """One labelled vector of named values.

    Parameters
    ----------
    values : array_like
        1D array of values.
    names : sequence of str
        One name per value.
    label : str, optional
        Name for this vector, e.g. the sequence it describes.

    Notes
    -----
    The constructor arguments are also the public attributes: ``values`` is the
    underlying 1D ``numpy.ndarray``, ``names`` the feature names as a tuple, and
    ``label`` the vector's own name (or None).

    Examples
    --------
    >>> v = FeatureVector([1.0, -2.0, 0.5], ["a", "b", "c"], label="IDR1")
    >>> v["b"]
    -2.0
    >>> v.to_dict()["a"]
    1.0
    """

    __slots__ = ("values", "names", "label")

    def __init__(self, values, names, label=None):
        self.values = np.asarray(values, dtype=np.float64)
        self.names = tuple(str(name) for name in names)
        self.label = label

        if self.values.ndim != 1:
            raise SparrowException(f"FeatureVector needs a 1D array, got shape {self.values.shape}")
        if self.values.size != len(self.names):
            raise SparrowException(f"Got {self.values.size} values but {len(self.names)} names")

    def __len__(self):
        return self.values.size

    def __iter__(self):
        """Iterate ``(name, value)`` pairs, so ``dict(v)`` works."""
        return zip(self.names, self.values)

    def __contains__(self, name):
        return name in self.names

    def __repr__(self):
        label = f" {self.label!r}" if self.label else ""
        return f"FeatureVector{label}|{len(self)} features"

    def __getitem__(self, key):
        """Look up by feature name or integer index."""
        if isinstance(key, str):
            return float(self.values[_index_of(key, self.names)])
        return self.values[key]

    def to_numpy(self):
        """Return the values as a plain NumPy array (a copy)."""
        return self.values.copy()

    def to_dict(self):
        """Return a ``{name: value}`` dict."""
        return dict(zip(self.names, (float(v) for v in self.values)))

    def to_dataframe(self):
        """Return a one-column pandas DataFrame indexed by feature name."""
        import pandas as pd

        return pd.DataFrame({self.label or "value": self.values}, index=list(self.names))


class FeatureTable:
    """A labelled 2D table - rows are sequences, columns are features.

    Parameters
    ----------
    values : array_like
        2D array of shape ``(n_rows, n_columns)``.
    row_names : sequence of str
        One name per row.
    column_names : sequence of str
        One name per column.

    Notes
    -----
    The constructor arguments are also the public attributes: ``values`` is the
    underlying 2D ``numpy.ndarray``, with ``row_names`` and ``column_names`` as
    tuples of labels.

    Examples
    --------
    >>> t = FeatureTable([[1.0, 2.0], [3.0, 4.0]], ["s1", "s2"], ["a", "b"])
    >>> t["a"]
    array([1., 3.])
    >>> t.row("s2")["b"]
    4.0
    """

    __slots__ = ("values", "row_names", "column_names")

    def __init__(self, values, row_names, column_names):
        self.values = np.atleast_2d(np.asarray(values, dtype=np.float64))
        self.row_names = tuple(str(name) for name in row_names)
        self.column_names = tuple(str(name) for name in column_names)

        if self.values.ndim != 2:
            raise SparrowException(f"FeatureTable needs a 2D array, got shape {self.values.shape}")
        if self.values.shape != (len(self.row_names), len(self.column_names)):
            raise SparrowException(
                f"Array shape {self.values.shape} does not match "
                f"{len(self.row_names)} rows x {len(self.column_names)} columns"
            )

    @property
    def shape(self):
        """``(n_rows, n_columns)``."""
        return self.values.shape

    def __len__(self):
        return self.values.shape[0]

    def __iter__(self):
        """Iterate over rows as :class:`FeatureVector` objects."""
        for index in range(len(self)):
            yield self.row(index)

    def __contains__(self, name):
        return name in self.column_names

    def __repr__(self):
        rows, columns = self.shape
        return f"FeatureTable|{rows} rows x {columns} columns"

    def __getitem__(self, key):
        """Look up a column by name, or a single value with ``(row, column)``."""
        if isinstance(key, str):
            return self.column(key)
        if isinstance(key, tuple) and len(key) == 2:
            row, column = key
            return self.row(row)[column]
        raise SparrowException(
            "Index a FeatureTable with a column name or a (row, column) pair. "
            "For anything more, work with .values directly."
        )

    def row(self, name_or_index):
        """Return one row as a :class:`FeatureVector`.

        Parameters
        ----------
        name_or_index : str or int
            Row name or positional index.

        Returns
        -------
        FeatureVector
        """
        if isinstance(name_or_index, str):
            index = _index_of(name_or_index, self.row_names, what="row")
        else:
            index = int(name_or_index)

        return FeatureVector(self.values[index], self.column_names, label=self.row_names[index])

    def column(self, name):
        """Return one column as a NumPy array, one value per row."""
        return self.values[:, _index_of(name, self.column_names)].copy()

    def to_numpy(self):
        """Return the values as a plain NumPy array (a copy)."""
        return self.values.copy()

    def to_dict(self):
        """Return nested ``{row_name: {column_name: value}}`` dicts."""
        return {name: self.row(index).to_dict() for index, name in enumerate(self.row_names)}

    def to_dataframe(self):
        """Return the table as a pandas DataFrame indexed by row name."""
        import pandas as pd

        return pd.DataFrame(self.values, index=list(self.row_names), columns=list(self.column_names))
