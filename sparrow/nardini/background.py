"""Compositional background distributions for NARDINI+ z-scores.

A NARDINI+ vector has two halves, and they get their null models from different
places:

* the **patterning** features are z-scored against scrambles of each individual
  sequence, so they need no external reference (see
  :mod:`sparrow.nardini.patterning`); while
* the **compositional** features are z-scored against the distribution of those
  same features across a *reference set* of IDRs - typically a whole IDRome.

This module builds, saves, and applies that reference set. The NARDINI+ notebook
downloads precomputed per-species means and standard deviations; here you build
them from whatever set of IDRs you like, which is what you want if you are
working with a non-model organism, a filtered sub-proteome, or a specific
experimental library.

Typical use::

    from sparrow.nardini import build_background

    background = build_background("human_idrome.fasta")
    background.save("human_idrome_background.npz")

    # later, or in another script
    from sparrow.nardini import NardiniBackground
    background = NardiniBackground.load("human_idrome_background.npz")

The means and standard deviations are accumulated with Welford's online
algorithm, so building a background over a full IDRome uses memory proportional
to the number of features rather than the number of sequences.

A background fails quietly rather than loudly: if a feature never varies across
the reference set there is no scale to divide by, so its z-score comes back as
exactly 0.0 for every sequence ever scored against it - which reads as "average"
but means "unmeasurable". :func:`build_background` therefore warns
(:class:`~sparrow.sparrow_exceptions.BackgroundQualityWarning`) when the set is
too small or too many features are constant; see
:data:`MIN_RECOMMENDED_SEQUENCES` and :data:`MAX_DEGENERATE_FRACTION`. The
background is still returned and still usable - this is advice, not an error.
"""

import json
import os
import warnings
from collections.abc import Mapping

import numpy as np

from sparrow.nardini.composition import (
    COMPOSITION_FEATURE_NAMES,
    composition_values,
)
from sparrow.sparrow_exceptions import (
    BackgroundQualityWarning,
    SparrowException,
)

__all__ = [
    "NardiniBackground",
    "build_background",
    "REFERENCE_SPECIES",
    "load_reference_background",
]

#: Species for which a published NARDINI+ compositional background ships with
#: sparrow, mapping the short key you pass to
#: :func:`load_reference_background` onto the species name.
REFERENCE_SPECIES = {
    "human": "Homo sapiens",
    "mouse": "Mus musculus",
    "rat": "Rattus norvegicus",
    "xenopus": "Xenopus tropicalis",
    "fly": "Drosophila melanogaster",
    "zebrafish": "Danio rerio",
    "yeast": "Saccharomyces cerevisiae",
    "worm": "Caenorhabditis elegans",
    "arabidopsis": "Arabidopsis thaliana",
}

#: Residues that mark a sequence as non-standard. Sequences containing any of
#: these are skipped when building a background, matching the NARDINI+ notebook.
_NONSTANDARD_RESIDUES = ("X", "U", "Z", "J", "B", "O")

_FORMAT_VERSION = 1


class NardiniBackground:
    """Per-feature mean and standard deviation of the compositional features.

    A background is normally created with :func:`build_background` rather than
    directly.

    Parameters
    ----------
    feature_names : sequence of str
        Names of the features described, in order.
    mean : numpy.ndarray
        Per-feature mean.
    std : numpy.ndarray
        Per-feature standard deviation (population, ``ddof=0``).
    num_sequences : int or None
        Number of sequences the statistics were estimated from, or None if that
        is not known (as for the bundled published priors, whose source tables do
        not report it).
    metadata : dict, optional
        Free-form provenance information (source file, date, filters applied,
        ...). Stored alongside the statistics when saved.

    Attributes
    ----------
    feature_names : tuple of str
    mean : numpy.ndarray
    std : numpy.ndarray
    num_sequences : int or None
    metadata : dict
    """

    def __init__(self, feature_names, mean, std, num_sequences, metadata=None):
        mean = np.asarray(mean, dtype=np.float64)
        std = np.asarray(std, dtype=np.float64)

        if not (len(feature_names) == mean.size == std.size):
            raise SparrowException(
                "feature_names, mean and std must all be the same length "
                f"(got {len(feature_names)}, {mean.size}, {std.size})"
            )

        self.feature_names = tuple(str(name) for name in feature_names)
        self.mean = mean
        self.std = std
        # -1 is the on-disk sentinel for "not recorded"
        if num_sequences is None or int(num_sequences) < 0:
            self.num_sequences = None
        else:
            self.num_sequences = int(num_sequences)
        self.metadata = dict(metadata or {})

    # .................................................................
    #
    def __repr__(self):
        count = "?" if self.num_sequences is None else self.num_sequences
        return f"NardiniBackground|{len(self.feature_names)} features|n={count}"

    def __len__(self):
        return len(self.feature_names)

    # .................................................................
    #
    def zscores(self, values):
        """Convert raw compositional values into z-scores.

        Features whose background standard deviation is zero - meaning every
        reference sequence had the identical value, so the feature carries no
        information - are returned as 0.0 rather than as infinity.

        Parameters
        ----------
        values : numpy.ndarray or dict
            Either an array of shape ``(n_features,)`` or ``(n_sequences,
            n_features)`` in this background's feature order, or a mapping from
            feature name to value.

        Returns
        -------
        numpy.ndarray
            Z-scores, the same shape as the input array (or 1D if a mapping was
            passed).

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If the input width does not match the background.
        """
        if isinstance(values, Mapping):
            values = np.asarray([values[name] for name in self.feature_names], dtype=np.float64)
        else:
            values = np.asarray(values, dtype=np.float64)

        if values.shape[-1] != len(self.feature_names):
            raise SparrowException(
                f"Expected {len(self.feature_names)} features but got "
                f"{values.shape[-1]}; the values and background do not match."
            )

        # only divide where the background actually varies
        usable = self.std > 0
        out = np.zeros_like(values, dtype=np.float64)
        np.divide(
            values - self.mean,
            np.where(usable, self.std, 1.0),
            out=out,
            where=np.broadcast_to(usable, values.shape),
        )
        return out

    # .................................................................
    #
    def as_dict(self):
        """Return the background as a plain dictionary of per-feature statistics.

        Returns
        -------
        collections.OrderedDict
            Mapping from feature name to ``{'mean': float, 'std': float}``.
        """
        from collections import OrderedDict

        return OrderedDict(
            (name, {"mean": float(m), "std": float(s)}) for name, m, s in zip(self.feature_names, self.mean, self.std)
        )

    # .................................................................
    #
    def save(self, filename, compressed=True):
        """Save the background to a NumPy ``.npz`` archive.

        Parameters
        ----------
        filename : str
            Destination path. A ``.npz`` extension is added if absent.
        compressed : bool, optional
            Use ``np.savez_compressed``. Default True.

        Returns
        -------
        str
            The path actually written.
        """
        if not str(filename).endswith(".npz"):
            filename = f"{filename}.npz"

        max_name_len = max(len(name) for name in self.feature_names)
        payload = {
            "feature_names": np.asarray(self.feature_names, dtype=f"<U{max_name_len}"),
            "mean": self.mean,
            "std": self.std,
            "num_sequences": np.asarray(-1 if self.num_sequences is None else self.num_sequences),
            "format_version": np.asarray(_FORMAT_VERSION),
            # JSON keeps the metadata readable and avoids pickled objects, so the
            # archive can be loaded with allow_pickle=False
            "metadata_json": np.asarray(json.dumps(self.metadata)),
        }

        if compressed:
            np.savez_compressed(filename, **payload)
        else:
            np.savez(filename, **payload)

        return filename

    # .................................................................
    #
    @classmethod
    def load(cls, filename):
        """Load a background previously written by :meth:`save`.

        Parameters
        ----------
        filename : str
            Path to the ``.npz`` archive.

        Returns
        -------
        NardiniBackground

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If the file does not exist or was written by a newer format version.
        """
        if not os.path.isfile(filename):
            raise SparrowException(f"No such NARDINI background file: {filename}")

        with np.load(filename, allow_pickle=False) as data:
            version = int(data["format_version"]) if "format_version" in data else 1
            if version > _FORMAT_VERSION:
                raise SparrowException(
                    f"{filename} was written in NARDINI background format v{version}, "
                    f"but this version of sparrow only understands v{_FORMAT_VERSION}. "
                    "Please upgrade sparrow."
                )

            feature_names = [str(x) for x in data["feature_names"].tolist()]
            mean = np.asarray(data["mean"], dtype=np.float64)
            std = np.asarray(data["std"], dtype=np.float64)
            num_sequences = int(data["num_sequences"])

            metadata = {}
            if "metadata_json" in data:
                metadata = json.loads(str(data["metadata_json"]))

        return cls(feature_names, mean, std, num_sequences, metadata)


# .................................................................
#
_REFERENCE_CACHE = {}


def load_reference_background(species="human"):
    """Load a published NARDINI+ compositional background that ships with sparrow.

    These are the per-species IDRome backgrounds from the GIN resource - the same
    tables the official NARDINI+ notebook downloads at runtime. Using one makes
    your compositional z-scores directly comparable to published NARDINI+ values,
    and the ``'human'`` background is the space the GIN cluster centroids were
    defined in, so it is what :func:`sparrow.nardini.assign_gin_clusters` expects.

    Use :func:`build_background` instead when you want a background over your own
    set of IDRs.

    Parameters
    ----------
    species : str, optional
        Which background to load. One of the keys of :data:`REFERENCE_SPECIES`:
        ``'human'`` (default), ``'mouse'``, ``'rat'``, ``'xenopus'``, ``'fly'``,
        ``'zebrafish'``, ``'yeast'``, ``'worm'``, ``'arabidopsis'``. The Latin
        name is also accepted.

    Returns
    -------
    NardiniBackground
        The published background. ``num_sequences`` is None because the source
        tables do not record it; ``metadata`` carries the provenance and citation.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``species`` is not one of the bundled backgrounds.

    Notes
    -----
    These data are redistributed from the GIN resource under the MIT licence.
    Please cite Ruff et al. (2025) if you use them - see
    ``sparrow/data/nardini/SOURCES.md``.

    Examples
    --------
    >>> background = load_reference_background("human")
    >>> len(background)
    54
    """
    import sparrow

    key = str(species).strip().lower()

    # accept the Latin name too
    if key not in REFERENCE_SPECIES:
        latin_lookup = {v.lower(): k for k, v in REFERENCE_SPECIES.items()}
        if key in latin_lookup:
            key = latin_lookup[key]

    if key not in REFERENCE_SPECIES:
        raise SparrowException(
            f"No bundled NARDINI background for '{species}'. Available: "
            f"{sorted(REFERENCE_SPECIES)}. Use build_background() to make your own."
        )

    if key in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[key]

    path = sparrow.get_data(f"nardini/gin_composition_prior_{key}_v1.npz")
    background = NardiniBackground.load(path)
    _REFERENCE_CACHE[key] = background
    return background


# .................................................................
#
def _iter_named_sequences(source, **read_fasta_kwargs):
    """Yield ``(name, sequence)`` pairs from any supported input.

    Accepts a FASTA path (``str`` or any :class:`os.PathLike`, so
    :class:`pathlib.Path` works), a mapping of name to sequence/Protein, or an
    iterable of sequences/Proteins (which get positional names).
    """
    # local import to avoid a circular import at module load
    from sparrow.protein import Protein

    def _to_sequence(value):
        if isinstance(value, Protein):
            return value.sequence
        if isinstance(value, str):
            return value.upper()
        raise SparrowException(f"Sequences must be strings or sparrow.Protein objects, got {type(value).__name__}")

    # os.PathLike covers pathlib.Path, which is otherwise iterable-ish enough to
    # fall through to the positional-names branch and fail confusingly
    if isinstance(source, (str, os.PathLike)):
        source = os.fspath(source)
        if not os.path.isfile(source):
            raise SparrowException(f"No such FASTA file: {source}")

        import protfasta

        records = protfasta.read_fasta(source, **read_fasta_kwargs)
        for name, sequence in records.items():
            yield name, sequence.upper()
        return

    if isinstance(source, Mapping):
        for name, value in source.items():
            yield str(name), _to_sequence(value)
        return

    for index, value in enumerate(source):
        yield f"seq_{index}", _to_sequence(value)


#: Below this many reference sequences, :func:`build_background` warns. The
#: relative standard error of an estimated standard deviation is about
#: ``1 / sqrt(2 * (n - 1))``, so 50 sequences puts the per-feature scale within
#: roughly 10% - which is about the point where a z-score of 2.0 versus 2.2 stops
#: being a distinction the background can actually support. The published IDRome
#: backgrounds are built from thousands.
MIN_RECOMMENDED_SEQUENCES = 50

#: Fraction of the 54 compositional features that may be constant across the
#: reference set before :func:`build_background` warns. A constant feature has no
#: scale to divide by, so its z-score is reported as 0.0 for every sequence you
#: ever score - the number looks like "average" but means "unmeasurable".
#:
#: Calibrated against real data: of the nine published IDRomes, seven have zero
#: constant features and two (yeast, worm) have one, so real reference sets sit at
#: 0-2%. Small or synthetic sets are far worse - 60 random IDR-like sequences give
#: about 8 of 54, and a single sequence gives all 54.
MAX_DEGENERATE_FRACTION = 0.10


def _warn_about_quality(count, degenerate):
    """Warn if a freshly built background is too weak to z-score against.

    Two independent failure modes, reported together so a bad background produces
    one warning rather than two. Neither is an error: the background is still
    returned and still usable, and a caller who knows their set is small can
    silence this with ``warnings.filterwarnings``.

    Parameters
    ----------
    count : int
        Number of sequences that contributed.
    degenerate : numpy.ndarray
        Indices of features whose standard deviation is exactly zero.
    """
    problems = []

    if count < MIN_RECOMMENDED_SEQUENCES:
        problems.append(
            f"it was built from only {count} sequence"
            f"{'' if count == 1 else 's'} (recommended: at least "
            f"{MIN_RECOMMENDED_SEQUENCES}), so every per-feature mean and "
            f"standard deviation is poorly determined"
        )

    total = len(COMPOSITION_FEATURE_NAMES)
    if degenerate.size > MAX_DEGENERATE_FRACTION * total:
        names = [COMPOSITION_FEATURE_NAMES[i] for i in degenerate[:5]]
        listed = ", ".join(names)
        if degenerate.size > len(names):
            listed += ", ..."
        problems.append(
            f"{degenerate.size} of {total} features are constant across the set "
            f"({listed}), so their z-scores will be reported as exactly 0.0 for "
            f"every sequence you score - not because those sequences are average, "
            f"but because the background cannot measure them"
        )

    if not problems:
        return

    warnings.warn(
        "This NARDINI+ compositional background is unlikely to give meaningful "
        "z-scores: " + "; and ".join(problems) + ". Consider using more reference "
        "sequences, or one of the published IDRome backgrounds via "
        "load_reference_background(). The patterning half of a NARDINI+ vector "
        "does not use a background at all and is unaffected.",
        BackgroundQualityWarning,
        stacklevel=3,
    )


def _is_usable(sequence, min_length):
    """Return whether a sequence should contribute to a background."""
    if len(sequence) < min_length:
        return False
    return not any(residue in sequence for residue in _NONSTANDARD_RESIDUES)


def build_background(
    source,
    min_length=1,
    metadata=None,
    show_progress=False,
    **read_fasta_kwargs,
):
    """Build a compositional background distribution from a set of IDRs.

    This is step one of the NARDINI+ workflow: estimate the mean and standard
    deviation of each of the 54 compositional features across a reference set of
    IDRs. The resulting object is then passed to
    :func:`sparrow.nardini.nardini_vectors` to convert query sequences into
    z-score vectors.

    Sequences containing non-standard residues (``X``, ``U``, ``Z``, ``J``, ``B``,
    ``O``) are skipped, matching the NARDINI+ reference implementation. The number
    skipped is recorded in the returned object's ``metadata``.

    Parameters
    ----------
    source : str, dict, or iterable
        Where to read the reference IDRs from. Either a path to a FASTA file, a
        mapping of ``{name: sequence_or_Protein}``, or an iterable of sequences /
        :class:`sparrow.protein.Protein` objects.
    min_length : int, optional
        Skip sequences shorter than this. Default 1 (keep everything), which
        matches the reference implementation. Raising it is sensible if your IDR
        set contains very short fragments whose compositional features are noisy.
    metadata : dict, optional
        Extra provenance information to store with the background, for example
        ``{'species': 'Homo sapiens', 'source': 'idrome_v3.fasta'}``.
    show_progress : bool, optional
        Print a progress bar while reading. Useful for whole IDRomes.
        Default False.
    **read_fasta_kwargs
        Additional keyword arguments forwarded to ``protfasta.read_fasta`` when
        ``source`` is a filename (for example ``invalid_sequence_action``).

    Returns
    -------
    NardiniBackground
        Per-feature mean and standard deviation over the usable sequences.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If no usable sequences were found.

    Warns
    -----
    sparrow.sparrow_exceptions.BackgroundQualityWarning
        If the background is unlikely to give meaningful z-scores - either it came
        from fewer than :data:`MIN_RECOMMENDED_SEQUENCES` sequences, or more than
        :data:`MAX_DEGENERATE_FRACTION` of the features are constant across the
        set. A constant feature has no scale to divide by, so its z-score is
        reported as ``0.0`` for every sequence scored against this background: the
        number looks like "average" but means "unmeasurable". The background is
        still returned and still usable. Silence this with::

            warnings.filterwarnings("ignore", category=BackgroundQualityWarning)

    See Also
    --------
    sparrow.nardini.nardini_vectors : Apply a background to query sequences.
    NardiniBackground.save : Persist a background for reuse.

    Examples
    --------
    >>> background = build_background("my_idrs.fasta")           # doctest: +SKIP
    >>> background.num_sequences                                  # doctest: +SKIP
    18452
    >>> background.save("my_idrs_background.npz")                 # doctest: +SKIP
    'my_idrs_background.npz'
    """
    n_features = len(COMPOSITION_FEATURE_NAMES)

    count = 0
    skipped = 0
    mean = np.zeros(n_features, dtype=np.float64)
    m2 = np.zeros(n_features, dtype=np.float64)

    iterator = _iter_named_sequences(source, **read_fasta_kwargs)
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="building NARDINI background")

    for _name, sequence in iterator:
        if not _is_usable(sequence, min_length):
            skipped += 1
            continue

        values = composition_values(sequence, validate=False)
        row = np.asarray([values[name] for name in COMPOSITION_FEATURE_NAMES], dtype=np.float64)

        # Welford's online update keeps memory flat over a whole IDRome
        count += 1
        delta = row - mean
        mean += delta / count
        m2 += delta * (row - mean)

    if count == 0:
        raise SparrowException(
            "Cannot build a NARDINI background: no usable sequences found "
            f"({skipped} skipped for containing non-standard residues or being "
            f"shorter than min_length={min_length})."
        )

    # population standard deviation, matching np.std's default
    std = np.sqrt(m2 / count)

    degenerate = np.flatnonzero(std == 0)

    resolved_metadata = {
        "num_sequences_used": count,
        "num_sequences_skipped": skipped,
        "min_length": min_length,
        "num_degenerate_features": int(degenerate.size),
    }
    if isinstance(source, (str, os.PathLike)):
        resolved_metadata["source"] = os.fspath(source)
    resolved_metadata.update(metadata or {})

    _warn_about_quality(count, degenerate)

    return NardiniBackground(
        feature_names=COMPOSITION_FEATURE_NAMES,
        mean=mean,
        std=std,
        num_sequences=count,
        metadata=resolved_metadata,
    )
