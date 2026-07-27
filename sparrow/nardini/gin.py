"""GIN (Grammars of IDRs using NARDINI+) cluster assignment.

The GIN resource partitions the human IDRome into 30 clusters of shared molecular
grammar, obtained by k-means on NARDINI+ z-score vectors. Each cluster has a
human-readable description - ``'D/E-tracts'``, ``'S patches'``,
``'Well-mixed P and G'``, and so on.

This module assigns *new* sequences to clusters: it computes each sequence's
NARDINI+ vector and finds the nearest centroid. It does not derive clusters -
:mod:`sparrow.nardini.clustering` does that, and produces a :class:`GINReference`
that :func:`assign_gin_clusters` accepts in place of the published one.

So there are two ways to use this module: against the 30 published human grammars
(the default, and what you want for human IDRs), or against centroids you built
yourself with :func:`~sparrow.nardini.clustering.build_gin_reference` (which is
what you want for anything else). :meth:`GINReference.save` and
:meth:`GINReference.load` persist the latter.

Whichever reference is used, the vectors and the centroids must live in the same
space. A reference records the patterning backend it was built from, and
:func:`assign_gin_clusters` refuses a vector computed with a different one, since
those are different statistics on different scales.

    from sparrow.nardini import (
        load_reference_background,
        nardini_vectors,
        assign_gin_clusters,
    )

    background = load_reference_background("human")
    result = nardini_vectors("my_idrs.fasta", background=background,
                             num_scrambles=10000, seed=42)
    assign_gin_clusters(result)

The assignment procedure, following the reference implementation:

1. Take the full 90-feature NARDINI+ z-score vector (36 patterning followed by 54
   compositional).
2. Clip it to the range [-3, +3]. The centroids were built from clipped vectors,
   so this keeps a few extreme features from dominating the distance.
3. Compute the squared Euclidean distance to each of the 30 centroids.
4. Assign the sequence to the nearest one.

The reported ``separation`` is the gap between the nearest and the next-nearest
centroid distance, which is a useful confidence measure: a sequence sitting
between two grammars has a small separation and should not be treated as firmly
belonging to either.

.. note::
   The reference implementation computes this gap using ``100`` as a sentinel for
   "this entry is the minimum", so a true gap larger than 100 would be reported as
   exactly 100. sparrow reports the true gap. In practice the distinction never
   arises - the centroids sit close enough together in 90 dimensions that the
   largest separation reachable by any clipped z-vector is well under 100 - and
   cluster assignments are identical either way.

.. important::
   The centroids are defined in **human** NARDINI+ z-score space. For the
   compositional half of the vector to be on the right scale, the background
   should be the human IDRome one (``load_reference_background('human')``). You
   *can* assign against a different background - the reference implementation
   allows it - but the further your background is from the human IDRome, the less
   the cluster labels mean.

How far is too far?
-------------------
Nearest-centroid assignment never abstains: every sequence gets a label,
including one that looks nothing like a human IDR. Three numbers on
:class:`GINAssignment` are there to tell you when not to believe it.

``separation``
    Gap to the runner-up cluster. Small means the sequence sits *between* two
    grammars, so the single label is arbitrary even if the sequence is otherwise
    perfectly typical. Check :meth:`~GINAssignment.ranked_clusters`.

``n_saturated``
    How many of the 90 features were pinned at the +/-3 clip. A clipped feature
    contributes a fixed amount to every centroid distance, so it carries no
    information about *which* cluster is nearest - it has been flattened. All 30
    published centroids lie within -2.32 to 2.99, so a sequence with many
    saturated features is outside the range any centroid describes.

``composition_rms_z``
    The most direct answer to "is this sequence anything like a human IDR?". The
    54 compositional features are z-scored against the reference IDRome, so a
    sequence drawn from that population has z ~ N(0, 1) per feature and an RMS
    near 1. Measured values: real and simulated IDRs land at **0.6-0.9**, a
    uniformly random sequence at **1.6**, and homopolymers or other extremes at
    **3-13**. This is computed before clipping, so it is not capped.

    Note that distance to the nearest centroid is *not* a usable absolute
    measure of typicality. In 90 dimensions even a perfectly ordinary z-vector
    sits about five times further from its nearest centroid than the centroids
    sit from each other, because the centroids are cluster means rather than
    members. Distances are only meaningful compared between sequences.

:attr:`GINAssignment.is_typical` combines the two into a coarse screen. It is a
heuristic for triage, not a statistical test: the published analysis does not
define an outlier criterion, and sparrow does not invent one. Treat a sequence it
flags as unclustered rather than as a member of whatever cluster happened to be
nearest.

This matters most for non-human sequences. A yeast or plant IDR scored against
the human background can be extreme on many compositional features at once,
saturating them and pushing the vector outside the space the centroids cover. In
that regime the *patterning* half of the vector is still perfectly meaningful -
it is z-scored against the sequence's own scrambles and involves no reference set
at all - but the GIN label is not.
"""

import csv
import json
import os

import numpy as np

from sparrow.nardini.composition import COMPOSITION_FEATURE_NAMES
from sparrow.nardini.groups import patterning_feature_names
from sparrow.nardini.patterning import DEFAULT_BACKEND
from sparrow.sparrow_exceptions import SparrowException

__all__ = [
    "GIN_CLIP",
    "GINReference",
    "GINAssignment",
    "load_gin_reference",
    "gin_cluster_definitions",
    "assign_gin_clusters",
]

#: NARDINI+ z-scores are clipped to +/- this value before distances are computed,
#: because the published centroids were derived from clipped vectors.
GIN_CLIP = 3.0

#: A sequence drawn from the same population as the background has compositional
#: z-scores distributed roughly as N(0, 1), so their RMS is about 1. Measured
#: values: IDR-like sequences land at 0.6-0.9, uniformly random sequence ~1.6,
#: homopolymers and other extremes 3-13. Above this, treat a GIN label as an
#: extrapolation - see :attr:`GINAssignment.is_typical`.
TYPICAL_COMPOSITION_RMS_Z = 2.0

#: A clipped feature contributes the same amount to every centroid distance, so
#: it cannot discriminate between clusters. More than this many, and much of the
#: vector has stopped carrying information.
TYPICAL_MAX_SATURATED = 5

#: On-disk format version for GINReference.save / .load
_REFERENCE_FORMAT_VERSION = 1

_CENTROID_FILE = "nardini/gin_centroids_human_v1.npz"
_DEFINITION_FILE = "nardini/gin_cluster_definitions_human_v1.csv"


class GINReference:
    """The published GIN cluster centroids and their descriptions.

    Load with :func:`load_gin_reference` rather than constructing directly.

    Attributes
    ----------
    feature_names : tuple of str
        The 90 NARDINI+ feature names, in the order the centroid columns use.
    clusters : numpy.ndarray
        Cluster identifiers (0-29).
    centroids : numpy.ndarray
        Array of shape ``(30, 90)`` of centroid coordinates.
    descriptions : dict
        Mapping from cluster number to its published description.
    num_idrs : dict
        Mapping from cluster number to the number of human IDRs assigned to it in
        the published analysis.
    metadata : dict
        Provenance and citation information.
    backend : str
        The patterning statistic the centroids live in. ``'nardini'`` for the
        published reference. A reference built with :func:`build_gin_reference`
        records whichever backend produced its vectors, and
        :func:`assign_gin_clusters` refuses to mix the two.
    """

    def __init__(
        self,
        feature_names,
        clusters,
        centroids,
        descriptions=None,
        num_idrs=None,
        metadata=None,
        backend=DEFAULT_BACKEND,
    ):
        self.feature_names = tuple(str(name) for name in feature_names)
        self.clusters = np.asarray(clusters, dtype=int)
        self.centroids = np.asarray(centroids, dtype=np.float64)
        self.descriptions = dict(descriptions or {})
        self.num_idrs = dict(num_idrs or {})
        self.metadata = dict(metadata or {})
        self.backend = str(backend)

        if self.centroids.shape != (self.clusters.size, len(self.feature_names)):
            raise SparrowException(
                f"GIN centroid matrix shape {self.centroids.shape} does not match "
                f"{self.clusters.size} clusters x {len(self.feature_names)} features"
            )

    def __len__(self):
        return int(self.clusters.size)

    def __repr__(self):
        return f"GINReference|{len(self)} clusters|{len(self.feature_names)} features"

    # .................................................................
    #
    def save(self, filename, compressed=True):
        """Save this reference to a NumPy ``.npz`` archive.

        Worth doing: building a reference over a real IDRome costs hours, almost
        all of it in the NARDINI+ vectors rather than the clustering. Saving also
        keeps a set of results comparable, since re-clustering with a different
        seed gives different (equally valid) centroids.

        Parameters
        ----------
        filename : str or os.PathLike
            Destination path. A ``.npz`` extension is added if absent.
        compressed : bool, optional
            Use ``np.savez_compressed``. Default True.

        Returns
        -------
        str
            The path actually written.
        """
        filename = os.fspath(filename)
        if not filename.endswith(".npz"):
            filename = f"{filename}.npz"

        width = max(len(name) for name in self.feature_names)
        payload = {
            "feature_names": np.asarray(self.feature_names, dtype=f"<U{width}"),
            "clusters": self.clusters,
            "centroids": self.centroids,
            "backend": np.asarray(self.backend),
            "format_version": np.asarray(_REFERENCE_FORMAT_VERSION),
            # JSON rather than pickled dicts, so the archive loads with
            # allow_pickle=False. Integer keys become strings and are restored.
            "descriptions_json": np.asarray(
                json.dumps({str(k): v for k, v in self.descriptions.items()})
            ),
            "num_idrs_json": np.asarray(
                json.dumps({str(k): int(v) for k, v in self.num_idrs.items()})
            ),
            "metadata_json": np.asarray(json.dumps(self.metadata)),
        }

        if compressed:
            np.savez_compressed(filename, **payload)
        else:
            np.savez(filename, **payload)

        return filename

    @classmethod
    def load(cls, filename):
        """Load a reference previously written by :meth:`save`.

        Parameters
        ----------
        filename : str or os.PathLike
            Path to a ``.npz`` archive.

        Returns
        -------
        GINReference

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If the file is missing or was not written by :meth:`save`.
        """
        filename = os.fspath(filename)
        if not os.path.isfile(filename):
            raise SparrowException(f"No such GIN reference file: {filename}")

        with np.load(filename, allow_pickle=False) as data:
            missing = {"feature_names", "clusters", "centroids"} - set(data.files)
            if missing:
                raise SparrowException(
                    f"{filename} does not look like a saved GIN reference "
                    f"(missing {sorted(missing)}). Archives are written by "
                    "GINReference.save()."
                )

            def load_json(key):
                return json.loads(str(data[key])) if key in data.files else {}

            return cls(
                feature_names=[str(name) for name in data["feature_names"]],
                clusters=data["clusters"],
                centroids=data["centroids"],
                descriptions={
                    int(k): v for k, v in load_json("descriptions_json").items()
                },
                num_idrs={int(k): v for k, v in load_json("num_idrs_json").items()},
                metadata=load_json("metadata_json"),
                backend=(
                    str(data["backend"]) if "backend" in data.files else DEFAULT_BACKEND
                ),
            )

    # .................................................................
    #
    def describe(self, cluster):
        """Return the published description of one cluster.

        Parameters
        ----------
        cluster : int
            Cluster number.

        Returns
        -------
        str
            The description, or ``'(no description available)'`` if unknown.
        """
        return self.descriptions.get(int(cluster), "(no description available)")

    def definitions(self):
        """Return the cluster definitions as a list of plain dictionaries.

        Returns
        -------
        list of dict
            One entry per cluster, with keys ``cluster``, ``description`` and
            ``num_idrs``, ordered by cluster number.

        Examples
        --------
        >>> load_gin_reference().definitions()[7]["description"]
        'D/E-tracts'
        """
        return [
            {
                "cluster": int(cluster),
                "description": self.describe(cluster),
                "num_idrs": self.num_idrs.get(int(cluster)),
            }
            for cluster in self.clusters
        ]

    @property
    def separation_scale(self):
        """Median *squared* distance from each centroid to its nearest neighbour.

        Squared Euclidean, so it is directly comparable to the distances reported
        by :func:`assign_gin_clusters` and :func:`compare_gin_references`, which
        use the same convention. About 17.8 for the published reference.

        Describes how tightly packed the cluster structure is.

        .. warning::
           This is **not** a threshold for judging whether a sequence belongs to
           its assigned cluster. In 90 dimensions even a perfectly typical
           z-vector sits roughly five times this far from its nearest centroid,
           because centroids are cluster means rather than members. Use
           :attr:`GINAssignment.composition_rms_z` for that question.

        Returns
        -------
        float
        """
        differences = self.centroids[:, None, :] - self.centroids[None, :, :]
        distances = np.nansum(differences**2, axis=2)
        np.fill_diagonal(distances, np.inf)
        return float(np.median(np.min(distances, axis=1)))

    def to_dataframe(self):
        """Return the cluster definitions as a pandas DataFrame.

        Convenience only - :meth:`definitions` gives the same information
        without needing pandas.

        Returns
        -------
        pandas.DataFrame
            Indexed by cluster number, with ``description`` and ``num_idrs``
            columns.
        """
        import pandas as pd

        frame = pd.DataFrame(self.definitions())
        return frame.set_index("cluster")


_REFERENCE_CACHE = {}


def load_gin_reference():
    """Load the bundled human GIN cluster centroids and descriptions.

    Returns
    -------
    GINReference
        The 30 published centroids, with their descriptions and IDR counts.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the bundled data files are missing or their feature order does not
        match sparrow's NARDINI+ feature definitions.

    Notes
    -----
    These data are redistributed from the GIN resource under the MIT licence.
    Please cite Ruff et al. (2025) if you use them - see
    ``sparrow/data/nardini/SOURCES.md``.

    Examples
    --------
    >>> reference = load_gin_reference()
    >>> len(reference)
    30
    >>> reference.describe(7)
    'D/E-tracts'
    """
    import sparrow

    if "human" in _REFERENCE_CACHE:
        return _REFERENCE_CACHE["human"]

    centroid_path = sparrow.get_data(_CENTROID_FILE)
    if not os.path.isfile(centroid_path):
        raise SparrowException(
            f"Bundled GIN centroid file is missing: {centroid_path}. "
            "Reinstall sparrow, or regenerate it with "
            "devtools/scripts/build_nardini_reference_data.py"
        )

    import json

    with np.load(centroid_path, allow_pickle=False) as data:
        feature_names = [str(x) for x in data["feature_names"].tolist()]
        clusters = np.asarray(data["clusters"], dtype=int)
        centroids = np.asarray(data["centroids"], dtype=np.float64)
        metadata = {}
        if "metadata_json" in data:
            metadata = json.loads(str(data["metadata_json"]))

    # the centroids are only meaningful if our feature order still matches theirs
    expected = list(patterning_feature_names("8")) + list(COMPOSITION_FEATURE_NAMES)
    if feature_names != expected:
        raise SparrowException(
            "The bundled GIN centroids do not match sparrow's NARDINI+ feature "
            f"order ({len(feature_names)} vs {len(expected)} features). This "
            "should not happen; please report it."
        )

    descriptions, num_idrs = {}, {}
    definition_path = sparrow.get_data(_DEFINITION_FILE)
    if os.path.isfile(definition_path):
        with open(definition_path, newline="") as handle:
            for row in csv.DictReader(handle):
                cluster = int(row["cluster"])
                descriptions[cluster] = row["description"]
                num_idrs[cluster] = int(row["num_idrs"])

    reference = GINReference(
        feature_names=feature_names,
        clusters=clusters,
        centroids=centroids,
        descriptions=descriptions,
        num_idrs=num_idrs,
        metadata=metadata,
    )
    _REFERENCE_CACHE["human"] = reference
    return reference


def gin_cluster_definitions(as_dataframe=False):
    """Return the 30 published GIN cluster definitions.

    Parameters
    ----------
    as_dataframe : bool, optional
        If True, return a pandas DataFrame indexed by cluster number. Default
        False, which returns plain dictionaries and needs no pandas.

    Returns
    -------
    list of dict or pandas.DataFrame
        One entry per cluster with ``cluster``, ``description`` and ``num_idrs``.

    Examples
    --------
    >>> definitions = gin_cluster_definitions()
    >>> definitions[7]["description"]
    'D/E-tracts'
    >>> definitions[7]["num_idrs"]
    763
    """
    reference = load_gin_reference()
    if as_dataframe:
        return reference.to_dataframe()
    return reference.definitions()


class GINAssignment:
    """GIN cluster assignments for a set of sequences.

    Returned by :func:`assign_gin_clusters`.

    The published description of each assigned cluster is available as the
    :attr:`descriptions` property documented below.

    Attributes
    ----------
    names : tuple of str
        Sequence identifiers.
    clusters : numpy.ndarray
        Assigned cluster number per sequence.
    distances : numpy.ndarray
        Squared Euclidean distance to the assigned centroid.
    separations : numpy.ndarray
        Gap between the nearest and next-nearest centroid distance. Small values
        mean the sequence sits between two grammars.
    all_distances : numpy.ndarray
        Array of shape ``(n_sequences, 30)`` of distances to every centroid.
    reference : GINReference
        The reference used.
    """

    def __init__(
        self,
        names,
        clusters,
        distances,
        separations,
        all_distances,
        reference,
        n_saturated=None,
        composition_rms_z=None,
    ):
        self.names = tuple(names)
        self.clusters = np.asarray(clusters, dtype=int)
        self.distances = np.asarray(distances, dtype=np.float64)
        self.separations = np.asarray(separations, dtype=np.float64)
        self.all_distances = np.asarray(all_distances, dtype=np.float64)
        self.reference = reference
        self.n_saturated = (
            np.zeros(len(self.names), dtype=int)
            if n_saturated is None
            else np.asarray(n_saturated, dtype=int)
        )
        self.composition_rms_z = (
            np.full(len(self.names), np.nan)
            if composition_rms_z is None
            else np.asarray(composition_rms_z, dtype=np.float64)
        )

    @property
    def descriptions(self):
        """Published description of each assigned cluster."""
        return tuple(self.reference.describe(c) for c in self.clusters)

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        """Iterate over per-sequence result dictionaries."""
        return iter(self.records())

    def __repr__(self):
        return f"GINAssignment|{len(self)} sequences"

    # .................................................................
    #
    @property
    def is_typical(self):
        """Whether each sequence is IDR-like enough for its GIN label to mean much.

        A coarse screen, True when **both**:

        * :attr:`composition_rms_z` is at most
          :data:`TYPICAL_COMPOSITION_RMS_Z` (2.0), i.e. the sequence's
          composition is broadly consistent with the reference IDRome; and
        * no more than :data:`TYPICAL_MAX_SATURATED` (5) of the 90 features were
          clipped at the +/-3 bound (:attr:`n_saturated`).

        This is a heuristic for triage, not a statistical test - the published
        analysis defines no outlier criterion. Treat a flagged sequence as
        unclustered rather than as a member of whatever cluster was nearest, and
        look at :meth:`ranked_clusters` before deciding.

        Returns
        -------
        numpy.ndarray of bool
        """
        composition_ok = ~(self.composition_rms_z > TYPICAL_COMPOSITION_RMS_Z)
        return composition_ok & (self.n_saturated <= TYPICAL_MAX_SATURATED)

    # .................................................................
    #
    def records(self):
        """Return the assignments as a list of plain dictionaries.

        Returns
        -------
        list of dict
            One per sequence, with keys ``name``, ``cluster``, ``description``,
            ``distance``, ``separation``, ``n_saturated``,
            ``composition_rms_z`` and ``is_typical``.
        """
        typical = self.is_typical
        return [
            {
                "name": self.names[index],
                "cluster": int(self.clusters[index]),
                "description": self.descriptions[index],
                "distance": float(self.distances[index]),
                "separation": float(self.separations[index]),
                "n_saturated": int(self.n_saturated[index]),
                "composition_rms_z": float(self.composition_rms_z[index]),
                "is_typical": bool(typical[index]),
            }
            for index in range(len(self))
        ]

    def to_dataframe(self):
        """Return the assignments as a pandas DataFrame.

        Convenience only - :meth:`records` and :meth:`to_table` give the same
        information without needing pandas.

        Returns
        -------
        pandas.DataFrame
            Indexed by sequence name.
        """
        import pandas as pd

        frame = pd.DataFrame(self.records())
        return frame.set_index("name")

    def ranked_clusters(self, name_or_index, n=5):
        """Return the nearest clusters for one sequence, closest first.

        Useful when ``separation`` is small and the single best cluster is not
        the whole story.

        Parameters
        ----------
        name_or_index : str or int
            Which sequence to report on.
        n : int, optional
            How many clusters to return. Default 5.

        Returns
        -------
        list of tuple
            ``(cluster, description, distance)``, nearest first.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If the requested sequence name is not present.
        """
        if isinstance(name_or_index, str):
            if name_or_index not in self.names:
                raise SparrowException(
                    f"No sequence named '{name_or_index}' in this assignment."
                )
            index = self.names.index(name_or_index)
        else:
            index = int(name_or_index)

        row = self.all_distances[index]
        order = np.argsort(row)[:n]
        return [
            (
                int(self.reference.clusters[i]),
                self.reference.describe(self.reference.clusters[i]),
                float(row[i]),
            )
            for i in order
        ]


def assign_gin_clusters(result, reference=None, clip=GIN_CLIP):
    """Assign sequences to their nearest published GIN cluster.

    Parameters
    ----------
    result : NardiniResult or numpy.ndarray
        Either a :class:`~sparrow.nardini.vectors.NardiniResult` computed *with* a
        compositional background, or a raw array of shape ``(n_sequences, 90)`` /
        ``(90,)`` of NARDINI+ z-scores in canonical feature order.
    reference : GINReference, optional
        Reference centroids. Defaults to the bundled human GIN reference.
    clip : float, optional
        Clip z-scores to ``[-clip, +clip]`` before computing distances. Default
        3.0, matching how the centroids were built. Pass None to disable, though
        that is not recommended - a couple of extreme compositional features can
        otherwise dominate the distance entirely.

    Returns
    -------
    GINAssignment
        Cluster assignments, distances and separations.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``result`` has no compositional block (GIN needs the full 90-feature
        vector), or if its width does not match the reference.

    See Also
    --------
    sparrow.nardini.load_reference_background : Get the human background the
        centroids were defined against.
    gin_cluster_definitions : What each cluster means.

    Examples
    --------
    >>> from sparrow.nardini import (                            # doctest: +SKIP
    ...     load_reference_background, nardini_vectors, assign_gin_clusters
    ... )
    >>> background = load_reference_background("human")          # doctest: +SKIP
    >>> result = nardini_vectors("my_idrs.fasta", background=background,
    ...                          num_scrambles=10000, seed=42)   # doctest: +SKIP
    >>> assign_gin_clusters(result).to_dataframe()               # doctest: +SKIP
    """
    if reference is None:
        reference = load_gin_reference()

    # accept either a NardiniResult or a bare array
    if hasattr(result, "zscores"):
        if not getattr(result, "has_composition", False):
            raise SparrowException(
                "GIN cluster assignment needs the full 90-feature NARDINI+ vector, "
                "but this result has only the patterning block. Recompute with a "
                "compositional background, e.g. nardini_vectors(..., "
                "background=load_reference_background('human'))."
            )

        # Centroids live in the space of whatever statistic produced them.
        # Assigning a vector from a different backend would compare quantities on
        # entirely different scales and return meaningless-but-plausible labels,
        # so the two must agree. The published reference is 'nardini', which is
        # why assigning a 'kappa' vector against it is refused - but a reference
        # you built yourself from 'kappa' vectors accepts them happily.
        backend = getattr(result, "backend", DEFAULT_BACKEND)
        if backend != reference.backend:
            published = reference.metadata.get("derived_from") is None
            hint = (
                "Recompute with nardini_vectors(..., "
                f"backend='{reference.backend}'), or build a reference in "
                f"'{backend}' space with build_gin_reference()."
            )
            raise SparrowException(
                f"Cannot assign a '{backend}' vector against centroids defined in "
                f"'{reference.backend}' space - the two statistics are on "
                f"different scales, so the labels would be meaningless. "
                + ("The bundled GIN reference is 'nardini'. " if published else "")
                + hint
            )

        vectors = np.atleast_2d(np.asarray(result.zscores, dtype=np.float64))
        names = list(result.names)
    else:
        vectors = np.atleast_2d(np.asarray(result, dtype=np.float64))
        names = [f"seq_{i}" for i in range(vectors.shape[0])]

    n_features = len(reference.feature_names)
    if vectors.shape[1] != n_features:
        raise SparrowException(
            f"Expected {n_features} NARDINI+ features for GIN assignment but got "
            f"{vectors.shape[1]}. The vector must be the 36 patterning features "
            "followed by the 54 compositional features."
        )

    # How far is this sequence from the population the background describes?
    # The 54 compositional features are standardized against that population, so
    # a member of it has z ~ N(0, 1) per feature and an RMS near 1. Computed on
    # the UNCLIPPED vector, since clipping would cap exactly the signal we want.
    n_composition = 54
    if vectors.shape[1] >= n_composition:
        composition_block = vectors[:, -n_composition:]
        composition_rms_z = np.sqrt(np.nanmean(composition_block**2, axis=1))
    else:  # pragma: no cover - width is validated above
        composition_rms_z = np.full(vectors.shape[0], np.nan)

    # count how many features are extreme enough to be pinned at the clip bound.
    # Those features carry no positional information once clipped, so a sequence
    # with many of them sits outside the region the centroids describe and its
    # assignment is an extrapolation rather than a lookup.
    if clip is not None:
        bound = abs(clip)
        n_saturated = np.count_nonzero(np.abs(vectors) >= bound, axis=1)
        vectors = np.clip(vectors, -bound, bound)
    else:
        n_saturated = np.zeros(vectors.shape[0], dtype=int)

    # squared Euclidean distance to every centroid; nan-tolerant so that an
    # unscored feature does not wipe out the whole distance
    diff = reference.centroids[None, :, :] - vectors[:, None, :]
    all_distances = np.nansum(diff**2, axis=2)

    best = np.argmin(all_distances, axis=1)
    clusters = reference.clusters[best]
    distances = all_distances[np.arange(vectors.shape[0]), best]

    # separation = gap to the next-nearest *distinct* centroid distance
    separations = np.empty(vectors.shape[0], dtype=np.float64)
    for row in range(vectors.shape[0]):
        ordered = np.sort(all_distances[row])
        runner_up = ordered[ordered > ordered[0]]
        separations[row] = float(runner_up[0] - ordered[0]) if runner_up.size else 0.0

    return GINAssignment(
        names=names,
        clusters=clusters,
        distances=distances,
        separations=separations,
        all_distances=all_distances,
        reference=reference,
        n_saturated=n_saturated,
        composition_rms_z=composition_rms_z,
    )
