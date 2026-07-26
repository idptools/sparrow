"""NARDINI access from a :class:`sparrow.protein.Protein` object.

Reached as the cached ``Protein.nardini`` accessor rather than constructed
directly::

    from sparrow import Protein

    p = Protein("MENDPRVRKFASDGAQWAIK...")

    p.nardini.vector()             # 90-feature NARDINI+ z-score vector
    p.nardini.patterning()         # just the 36 patterning z-scores
    p.nardini.composition()        # just the 54 compositional z-scores
    p.nardini.gin_cluster()        # nearest published human GIN grammar

Everything returns a :class:`~sparrow.nardini.table.FeatureVector` - a labelled
NumPy array you can index by feature name - or, for the GIN call, a plain dict.

The background
--------------
Compositional z-scores need a reference set of IDRs to be scored against. For a
single protein there is nothing sensible to build one from, so these methods
default to the **published human IDRome background** that ships with sparrow.
That is also the space the GIN centroids live in, so it is the right default for
cluster assignment. Pass ``background=`` to override with another bundled species
or one of your own::

    p.nardini.vector(background="yeast")
    p.nardini.vector(background=my_background)

The patterning half never uses the background - it is z-scored against scrambles
of the sequence itself - so ``patterning()`` gives identical answers regardless.
"""

import hashlib
import os

import numpy as np

from sparrow.nardini.background import (
    NardiniBackground,
    load_reference_background,
)
from sparrow.nardini.patterning import (
    DEFAULT_BACKEND,
    DEFAULT_BLOB_SIZE,
    DEFAULT_MIN_FRACTION,
    DEFAULT_NUM_SCRAMBLES,
)
from sparrow.sparrow_exceptions import SparrowException

__all__ = ["ProteinNardini"]

#: Background used when none is given. The published human IDRome prior, which is
#: also the space the GIN centroids were defined in.
DEFAULT_REFERENCE = "human"


def _background_key(background):
    """Return a stable cache key identifying a background by its *contents*.

    This used to be ``id(background)`` for :class:`NardiniBackground` objects,
    which is unsound. The cache stores only the key, so it keeps no reference to
    the background: once the caller drops theirs the object is collected and
    CPython is free to allocate the next background at the same address. The
    memoized result for the *old* background was then returned for the new one -
    silently, and wrongly. It reproduces in a handful of allocations, and was
    observed reporting a compositional z-score of +19.8 where the correct answer
    was -4.7.

    Hashing the mean/std arrays fixes that, and has the side benefit that two
    separately-built but identical backgrounds now share a cache entry.

    Parameters
    ----------
    background : NardiniBackground, str, os.PathLike, or None
        The ``background`` argument as passed by the caller.

    Returns
    -------
    hashable
        A key that depends on the background's contents rather than its identity.
        Species names and paths are keyed by their own string value, as before.
    """
    if background is None:
        return None

    if isinstance(background, NardiniBackground):
        digest = hashlib.blake2b(digest_size=16)
        digest.update(repr(background.feature_names).encode("utf-8"))
        for array in (background.mean, background.std):
            digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
        return digest.hexdigest()

    if isinstance(background, os.PathLike):
        # so Path("bg.npz") and "bg.npz" share a cache entry
        return os.fspath(background)

    return background


def resolve_background(background):
    """Turn a ``background`` argument into a NardiniBackground, or None.

    Accepts a bundled species key, a path to a saved ``.npz`` (``str`` or
    :class:`os.PathLike`), an existing
    :class:`~sparrow.nardini.background.NardiniBackground`, or None.

    Shared with :func:`sparrow.nardini.batch.nardini_proteins`, which resolves the
    background once in the parent process so that workers receive a small pickled
    object instead of each re-reading a file or re-deriving a species key.

    Parameters
    ----------
    background : str, os.PathLike, NardiniBackground, or None

    Returns
    -------
    NardiniBackground or None

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the argument is none of the accepted forms.
    """
    if background is None:
        return None
    if isinstance(background, NardiniBackground):
        return background
    if isinstance(background, os.PathLike):
        # a path object can only be a saved background, never a species key
        return NardiniBackground.load(background)
    if isinstance(background, str):
        # a bundled species key, or a path to a saved background
        if background.endswith(".npz"):
            return NardiniBackground.load(background)
        return load_reference_background(background)
    raise SparrowException(
        "background must be a species name, a path to a saved .npz "
        f"background, a NardiniBackground, or None - got {type(background).__name__}"
    )


class ProteinNardini:
    """NARDINI / NARDINI+ calculations for a single protein.

    Not constructed directly - reached as ``Protein.nardini``. Results are
    memoized per unique set of options, so repeated access is free.

    Parameters
    ----------
    protein_obj : sparrow.protein.Protein
        The protein these calculations describe.
    """

    def __init__(self, protein_obj):
        self.__protein = protein_obj
        self.__cache = {}

    # .................................................................
    #
    def __repr__(self):
        return f"ProteinNardini|L = {len(self.__protein)}"

    def _resolve_background(self, background):
        """Turn the ``background`` argument into a NardiniBackground, or None."""
        return resolve_background(background)

    @staticmethod
    def _cache_key(
        background, num_scrambles, seed, typeset, blob_size, min_fraction, backend
    ):
        """The memoization key for one set of options.

        Shared with :func:`sparrow.nardini.batch.nardini_proteins`, which computes
        results out-of-process and stores them here. Both must agree on the key or
        a precomputed result would be invisible and get recomputed.
        """
        return (
            _background_key(background),
            num_scrambles,
            seed,
            str(typeset),
            blob_size,
            min_fraction,
            backend,
        )

    def _store(self, result, **options):
        """Adopt an externally computed NardiniResult for this protein.

        Used by :func:`sparrow.nardini.batch.nardini_proteins` to attach results
        computed in worker processes. The stored result must be exactly what this
        protein would have computed for itself with the same options - which holds
        because per-sequence seeds derive from the sequence rather than its
        position in a batch, so a vector does not depend on how it was batched.

        Parameters
        ----------
        result : sparrow.nardini.vectors.NardiniResult
            A single-sequence result for this protein's sequence.
        **options
            The option set it was computed with, as accepted by :meth:`_cache_key`.
        """
        self.__cache[self._cache_key(**options)] = result

    def _patterning_result(
        self, num_scrambles, seed, typeset, blob_size, min_fraction, backend
    ):
        """A result whose patterning block matches these options, at any background.

        The patterning half is z-scored against scrambles of the sequence itself and
        never touches the background, so its values are identical for every
        background - verified in test_nardini_batch.py. Any cached result with the
        same patterning options therefore already holds the answer, and recomputing
        it would repeat the scrambling, which is essentially the entire cost.

        This matters after :func:`sparrow.nardini.batch.nardini_proteins`, which
        caches results under the background it was given: without this,
        ``patterning()`` would key on ``background=None``, miss, and re-run hours of
        scrambling that had already been done.
        """
        wanted = (num_scrambles, seed, str(typeset), blob_size, min_fraction, backend)
        for key, cached in self.__cache.items():
            if key[1:] == wanted:
                return cached

        return self._result(
            None, num_scrambles, seed, typeset, blob_size, min_fraction, backend
        )

    def _result(
        self,
        background,
        num_scrambles,
        seed,
        typeset,
        blob_size,
        min_fraction,
        backend,
    ):
        """Compute (and cache) the underlying NardiniResult for one option set."""
        from sparrow.nardini.vectors import nardini_vectors

        key = self._cache_key(
            background, num_scrambles, seed, typeset, blob_size, min_fraction, backend
        )
        if key not in self.__cache:
            self.__cache[key] = nardini_vectors(
                {"sequence": self.__protein.sequence},
                background=self._resolve_background(background),
                num_scrambles=num_scrambles,
                seed=seed,
                typeset=typeset,
                blob_size=blob_size,
                min_fraction=min_fraction,
                backend=backend,
            )
        return self.__cache[key]

    # .................................................................
    #
    def vector(
        self,
        background=DEFAULT_REFERENCE,
        num_scrambles=DEFAULT_NUM_SCRAMBLES,
        seed=None,
        raw=False,
        typeset="8",
        blob_size=DEFAULT_BLOB_SIZE,
        min_fraction=DEFAULT_MIN_FRACTION,
        backend=DEFAULT_BACKEND,
    ):
        """Return the full NARDINI+ feature vector for this protein.

        36 patterning z-scores (against scrambles of this sequence) followed by
        54 compositional z-scores (against ``background``).

        Parameters
        ----------
        background : str, NardiniBackground, or None, optional
            Reference set for the compositional half. A bundled species key
            (default ``'human'``), a path to a saved ``.npz``, a
            :class:`~sparrow.nardini.background.NardiniBackground`, or None to
            omit the compositional block and return only the 36 patterning
            features.
        num_scrambles : int, optional
            Scrambles used for the patterning null distribution. Default 10,000.
        seed : int or None, optional
            Seed for reproducible patterning z-scores.
        raw : bool, optional
            Return the raw values instead of z-scores. Default False.
        typeset : {'8', '9'}, optional
            Residue grouping for the patterning features. Default ``'8'``.
        blob_size : int, optional
            Patterning window size. Default 5.
        min_fraction : float, optional
            Abundance threshold for scoring a patterning feature. Default 0.10.
        backend : str, optional
            Patterning statistic - ``'nardini'`` (default), ``'kappa'`` or
            ``'iwd'``. See
            :data:`sparrow.nardini.patterning.PATTERNING_BACKENDS`.

        Returns
        -------
        sparrow.nardini.table.FeatureVector
            Labelled vector of 90 values (or 36 if ``background`` is None).

        Examples
        --------
        >>> v = p.nardini.vector(seed=42)              # doctest: +SKIP
        >>> v["Frac K+R"], v["pol-neg"]                # doctest: +SKIP
        >>> v.to_numpy().shape                         # doctest: +SKIP
        (90,)
        """
        result = self._result(
            background, num_scrambles, seed, typeset, blob_size, min_fraction, backend
        )
        vector = result.vector(0, raw=raw)
        vector.label = "sequence"
        return vector

    def patterning(
        self,
        num_scrambles=DEFAULT_NUM_SCRAMBLES,
        seed=None,
        raw=False,
        typeset="8",
        blob_size=DEFAULT_BLOB_SIZE,
        min_fraction=DEFAULT_MIN_FRACTION,
        backend=DEFAULT_BACKEND,
    ):
        """Return only the patterning half of the NARDINI+ vector.

        36 z-scores (45 for the 9-type grouping) describing how segregated each
        residue type, and each pair of types, is along the sequence. Positive
        means blockier than this sequence's own composition predicts, negative
        means more evenly interspersed.

        No background is involved, so this needs no reference set and is
        comparable across sequences of any composition or origin.

        Parameters
        ----------
        num_scrambles, seed, raw, typeset, blob_size, min_fraction, backend
            As for :meth:`vector`.

        Returns
        -------
        sparrow.nardini.table.FeatureVector

        Examples
        --------
        >>> p.nardini.patterning(seed=1).to_dict()     # doctest: +SKIP
        {'pol-pol': 0.41, 'pol-neg': -1.84, ...}
        """
        from sparrow.nardini.table import FeatureVector

        result = self._patterning_result(
            num_scrambles, seed, typeset, blob_size, min_fraction, backend
        )
        values = result.raw_patterning[0] if raw else result.patterning_zscores[0]
        return FeatureVector(
            values, result.patterning_feature_names, label="patterning"
        )

    def composition(self, background=DEFAULT_REFERENCE, raw=False):
        """Return only the compositional half of the NARDINI+ vector.

        54 features describing what the sequence is made of. Cheap - no
        scrambling involved.

        Parameters
        ----------
        background : str, NardiniBackground, or None, optional
            Reference set to z-score against, default ``'human'``. Pass None to
            get the raw values with no reference at all (equivalent to
            ``raw=True``).
        raw : bool, optional
            Return raw values rather than z-scores. Default False.

        Returns
        -------
        sparrow.nardini.table.FeatureVector

        Examples
        --------
        >>> p.nardini.composition()["Frac K+R"]        # doctest: +SKIP
        1.83
        >>> p.nardini.composition(raw=True)["Frac K+R"]  # doctest: +SKIP
        0.19
        """
        from sparrow.nardini.composition import (
            COMPOSITION_FEATURE_NAMES,
            composition_values,
        )
        from sparrow.nardini.table import FeatureVector

        values = composition_values(self.__protein.sequence)
        raw_array = [values[name] for name in COMPOSITION_FEATURE_NAMES]

        if raw or background is None:
            return FeatureVector(
                raw_array, COMPOSITION_FEATURE_NAMES, label="composition (raw)"
            )

        resolved = self._resolve_background(background)
        return FeatureVector(
            resolved.zscores(values), COMPOSITION_FEATURE_NAMES, label="composition"
        )

    # .................................................................
    #
    def gin_cluster(
        self,
        background=DEFAULT_REFERENCE,
        num_scrambles=DEFAULT_NUM_SCRAMBLES,
        seed=None,
        reference=None,
    ):
        """Assign this protein to its nearest published human GIN cluster.

        Returns the single-sequence view of
        :func:`sparrow.nardini.gin.assign_gin_clusters`.

        .. important::
           The GIN clusters describe the **human** IDRome. Every sequence gets a
           label, including ones that look nothing like a human IDR, so check the
           returned ``is_typical`` flag before believing it. See
           :meth:`gin_report` for a readable summary, and the notes in
           :mod:`sparrow.nardini.gin` for what the diagnostics mean.

        Parameters
        ----------
        background : str or NardiniBackground, optional
            Reference set for the compositional half. Default ``'human'``, which
            is the space the centroids were defined in - changing it makes the
            labels progressively less meaningful.
        num_scrambles : int, optional
            Scrambles for the patterning half. Default 10,000.
        seed : int or None, optional
            Seed for reproducibility.
        reference : sparrow.nardini.gin.GINReference, optional
            Alternative centroids. Defaults to the bundled human reference.

        Returns
        -------
        dict
            ``name``, ``cluster``, ``description``, ``distance``,
            ``separation``, ``n_saturated``, ``within_cluster_scale`` and
            ``is_typical``.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If ``background`` is None - GIN needs the full 90-feature vector.

        Examples
        --------
        >>> p.nardini.gin_cluster(seed=42)["description"]    # doctest: +SKIP
        'Weak positive charge'
        """
        if background is None:
            raise SparrowException(
                "GIN cluster assignment needs the compositional half of the "
                "vector, so a background is required. Use the default "
                "background='human' unless you have a specific reason not to."
            )

        from sparrow.nardini.gin import assign_gin_clusters

        result = self._result(
            background,
            num_scrambles,
            seed,
            "8",
            DEFAULT_BLOB_SIZE,
            DEFAULT_MIN_FRACTION,
            "nardini",
        )
        assignment = assign_gin_clusters(result, reference=reference)
        record = assignment.records()[0]
        record["name"] = getattr(self.__protein, "name", None) or "sequence"
        return record

    def gin_report(
        self,
        background=DEFAULT_REFERENCE,
        num_scrambles=DEFAULT_NUM_SCRAMBLES,
        seed=None,
        n_alternatives=3,
    ):
        """Return a readable summary of the GIN assignment and how much to trust it.

        Parameters
        ----------
        background : str or NardiniBackground, optional
            As for :meth:`gin_cluster`.
        num_scrambles : int, optional
            Scrambles for the patterning half. Default 10,000.
        seed : int or None, optional
            Seed for reproducibility.
        n_alternatives : int, optional
            How many runner-up clusters to list. Default 3.

        Returns
        -------
        str
            A short multi-line summary.

        Examples
        --------
        >>> print(p.nardini.gin_report(seed=42))       # doctest: +SKIP
        """
        from sparrow.nardini.gin import assign_gin_clusters

        result = self._result(
            background,
            num_scrambles,
            seed,
            "8",
            DEFAULT_BLOB_SIZE,
            DEFAULT_MIN_FRACTION,
            "nardini",
        )
        assignment = assign_gin_clusters(result)
        record = assignment.records()[0]

        from sparrow.nardini.gin import (
            TYPICAL_COMPOSITION_RMS_Z,
            TYPICAL_MAX_SATURATED,
        )

        lines = [
            f"GIN cluster {record['cluster']}: {record['description']}",
            f"  distance to centroid   {record['distance']:.2f}",
            f"  gap to runner-up       {record['separation']:.2f}",
            f"  saturated features     {record['n_saturated']} of 90",
            f"  composition RMS z      {record['composition_rms_z']:.2f}"
            f"   (~1 = typical of the reference IDRome)",
        ]

        if record["is_typical"]:
            lines.append(
                "  -> composition is consistent with the reference IDRome; the"
            )
            lines.append("     label is usable.")
        else:
            reasons = []
            if record["composition_rms_z"] > TYPICAL_COMPOSITION_RMS_Z:
                reasons.append(
                    f"composition RMS z of {record['composition_rms_z']:.1f} is far "
                    "from the reference IDRome"
                )
            if record["n_saturated"] > TYPICAL_MAX_SATURATED:
                reasons.append(
                    f"{record['n_saturated']} features pinned at the +/-3 clip"
                )
            lines.append("  -> CAUTION: " + "; ".join(reasons) + ".")
            lines.append(
                "     This sequence sits outside the region the human GIN clusters"
            )
            lines.append(
                "     describe, so the label is an extrapolation. The patterning"
            )
            lines.append(
                "     half of the vector is unaffected and remains meaningful."
            )

        alternatives = assignment.ranked_clusters(0, n=n_alternatives + 1)[1:]
        if alternatives:
            lines.append("  nearest alternatives:")
            for cluster, description, distance in alternatives:
                lines.append(f"    {cluster:2d}  {description:<45s} d={distance:7.2f}")

        return "\n".join(lines)
