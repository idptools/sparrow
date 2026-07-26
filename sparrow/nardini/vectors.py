"""Top-level NARDINI+ vector calculation.

Combines the two halves of a NARDINI+ vector - scramble-based patterning
z-scores and background-based compositional z-scores - into a single feature
vector per sequence.

For the default 8-type grouping the full vector has 90 features: 36 patterning
plus 54 compositional. Patterning features are named ``'<type>-<type>'`` (for
example ``'pol-neg'``) and compositional features carry their NARDINI+ names (for
example ``'Frac K+R'``, ``'Iso point'``, ``'RG Frac'``).

Interpreting the numbers
------------------------
Both halves are z-scores, but they answer different questions:

* A patterning z-score compares the sequence to *scrambles of itself*. Positive
  means the residue type is more segregated (blockier) than its own composition
  would predict; negative means more evenly interspersed.
* A compositional z-score compares the sequence to the *reference IDR set* you
  built the background from. Positive means enriched relative to that reference.

So a patterning z-score is composition-independent by construction, while a
compositional z-score is only as meaningful as the background behind it.
"""

import hashlib
import os

import numpy as np

from sparrow.nardini.background import NardiniBackground
from sparrow.nardini.composition import (
    COMPOSITION_FEATURE_NAMES,
    composition_values,
)
from sparrow.nardini.groups import DEFAULT_TYPESET, patterning_feature_names
from sparrow.nardini.patterning import (
    DEFAULT_BACKEND,
    DEFAULT_BLOB_SIZE,
    DEFAULT_MIN_FRACTION,
    DEFAULT_NUM_SCRAMBLES,
    patterning_zscores,
    resolve_backend,
)
from sparrow.nardini.table import FeatureTable
from sparrow.sparrow_exceptions import SparrowException

__all__ = ["NardiniResult", "nardini_vectors"]


class NardiniResult:
    """The result of a NARDINI+ analysis over one or more sequences.

    Not normally constructed directly - returned by :func:`nardini_vectors`.

    The combined z-score matrix and its column names are available as the
    :attr:`zscores` and :attr:`feature_names` properties documented below.

    Attributes
    ----------
    names : tuple of str
        Sequence identifiers, in input order.
    sequences : tuple of str
        The analyzed sequences, in input order.
    patterning_feature_names : tuple of str
        Just the patterning feature names.
    composition_feature_names : tuple of str
        Just the compositional feature names (empty if no background was used).
    patterning_zscores : numpy.ndarray
        Array of shape ``(n_sequences, 36)``.
    composition_zscores : numpy.ndarray
        Array of shape ``(n_sequences, 54)``, or shape ``(n_sequences, 0)`` if no
        background was used.
    raw_patterning : numpy.ndarray
        The un-z-scored patterning asymmetries.
    raw_composition : numpy.ndarray
        The un-z-scored compositional values.
    backend : str
        The patterning statistic these z-scores were computed with. Recorded
        because the numbers are not comparable across backends, and because GIN
        cluster assignment requires ``'nardini'``.
    """

    def __init__(
        self,
        names,
        sequences,
        patterning_names,
        composition_names,
        patterning_zscores,
        composition_zscores,
        raw_patterning,
        raw_composition,
        backend=DEFAULT_BACKEND,
    ):
        self.names = tuple(names)
        self.sequences = tuple(sequences)
        self.patterning_feature_names = tuple(patterning_names)
        self.composition_feature_names = tuple(composition_names)
        self.patterning_zscores = patterning_zscores
        self.composition_zscores = composition_zscores
        self.raw_patterning = raw_patterning
        self.raw_composition = raw_composition
        self.backend = backend

    # .................................................................
    #
    @property
    def has_composition(self):
        """Whether the compositional z-score block is present."""
        return len(self.composition_feature_names) > 0

    @property
    def feature_names(self):
        """Names of the columns of :attr:`zscores`."""
        return self.patterning_feature_names + self.composition_feature_names

    @property
    def zscores(self):
        """The full z-score matrix, patterning block followed by composition."""
        if not self.has_composition:
            return self.patterning_zscores
        return np.hstack([self.patterning_zscores, self.composition_zscores])

    def __len__(self):
        return len(self.sequences)

    def __repr__(self):
        return f"NardiniResult|{len(self)} sequences|{len(self.feature_names)} features"

    # .................................................................
    #
    def _block(self, raw):
        """Return ``(values, column_names)`` for the z-score or raw block."""
        if not raw:
            return self.zscores, list(self.feature_names)

        blocks = [self.raw_patterning]
        columns = list(self.patterning_feature_names)
        if self.has_composition:
            blocks.append(self.raw_composition)
            columns += list(self.composition_feature_names)
        values = np.hstack(blocks) if len(blocks) > 1 else blocks[0]
        return values, columns

    def to_table(self, raw=False):
        """Return the results as a :class:`~sparrow.nardini.table.FeatureTable`.

        This is the default tabular view: the values as a plain NumPy array with
        the row and column names kept alongside them.

        Parameters
        ----------
        raw : bool, optional
            If True, return the raw (un-z-scored) values instead of z-scores.
            Default False.

        Returns
        -------
        sparrow.nardini.table.FeatureTable
            One row per sequence, one column per feature.

        Examples
        --------
        >>> table = result.to_table()                        # doctest: +SKIP
        >>> table["pol-neg"]                                 # doctest: +SKIP
        array([-1.8423,  0.4412])
        >>> table["IDR1", "RG Frac"]                         # doctest: +SKIP
        12.4
        >>> table.values.shape                               # doctest: +SKIP
        (2, 90)
        """
        values, columns = self._block(raw)
        return FeatureTable(values, self.names, columns)

    def to_dataframe(self, raw=False):
        """Return the results as a pandas DataFrame.

        Convenience only - :meth:`to_table` is the default view.

        Parameters
        ----------
        raw : bool, optional
            If True, return the raw (un-z-scored) values instead of z-scores.
            Default False.

        Returns
        -------
        pandas.DataFrame
            One row per sequence, indexed by sequence name, with one column per
            feature.
        """
        return self.to_table(raw=raw).to_dataframe()

    def vector(self, name_or_index=0, raw=False):
        """Return one sequence's features as a labelled vector.

        Parameters
        ----------
        name_or_index : str or int, optional
            Which sequence. Default the first.
        raw : bool, optional
            Return raw values instead of z-scores. Default False.

        Returns
        -------
        sparrow.nardini.table.FeatureVector

        Examples
        --------
        >>> v = result.vector("IDR1")                    # doctest: +SKIP
        >>> v["Frac K+R"], v["pol-neg"]                  # doctest: +SKIP
        """
        return self.to_table(raw=raw).row(name_or_index)

    # .................................................................
    #
    def top_features(self, name_or_index, n=10):
        """Return a sequence's most extreme features by absolute z-score.

        Parameters
        ----------
        name_or_index : str or int
            Which sequence to report on, either its name or its positional index.
        n : int, optional
            How many features to return. Default 10.

        Returns
        -------
        list of tuple
            ``(feature_name, zscore)`` pairs, most extreme first.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If the requested sequence name is not present.
        """
        if isinstance(name_or_index, str):
            if name_or_index not in self.names:
                raise SparrowException(
                    f"No sequence named '{name_or_index}' in this result."
                )
            index = self.names.index(name_or_index)
        else:
            index = int(name_or_index)

        row = self.zscores[index]
        order = np.argsort(-np.abs(row))[:n]
        return [(self.feature_names[i], float(row[i])) for i in order]

    # .................................................................
    #
    def gin_clusters(self, reference=None, clip=None):
        """Assign these sequences to their nearest published GIN cluster.

        Convenience wrapper over
        :func:`sparrow.nardini.gin.assign_gin_clusters`. Requires that this result
        was computed with a compositional background, since GIN needs the full
        90-feature vector.

        Parameters
        ----------
        reference : sparrow.nardini.gin.GINReference, optional
            Reference centroids. Defaults to the bundled human GIN reference.
        clip : float, optional
            Clip applied to the z-scores before computing distances. Defaults to
            :data:`sparrow.nardini.gin.GIN_CLIP` (3.0).

        Returns
        -------
        sparrow.nardini.gin.GINAssignment

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If this result has no compositional block.

        Examples
        --------
        >>> result.gin_clusters().to_dataframe()     # doctest: +SKIP
        """
        from sparrow.nardini.gin import GIN_CLIP, assign_gin_clusters

        return assign_gin_clusters(
            self, reference=reference, clip=GIN_CLIP if clip is None else clip
        )


# .................................................................
#
def _derive_seed(seed, sequence):
    """Derive a stable per-sequence seed from the master seed and the sequence.

    Keyed on the sequence itself rather than on its position in the batch. That is
    what makes the documented guarantee true: a given sequence gets the same
    scramble stream, and therefore the same patterning z-scores, however it is
    batched. Scoring one sequence on its own, or as part of a set, or as part of a
    differently-ordered set, all agree exactly.

    This was previously ``seed + index``, which is reproducible for an identical
    set in an identical order but nothing more - re-running a single sequence from
    a batch gave different numbers, and inserting a sequence at the front shifted
    every subsequent one.

    ``hashlib`` rather than ``hash()``: Python salts string hashing per process,
    so ``hash()`` would not be reproducible across runs at all.

    Parameters
    ----------
    seed : int or None
        The master seed. None means "unseeded", which is propagated as None.
    sequence : str
        The sequence being scored.

    Returns
    -------
    int or None
        A 64-bit seed, or None if ``seed`` was None.
    """
    if seed is None:
        return None

    # utf-8 rather than ascii: this runs before the residues have been validated,
    # and a non-ASCII character must still reach that validation and get
    # sparrow's "non-standard residue" error rather than a UnicodeEncodeError
    payload = f"{int(seed)}:{sequence}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _resolve_input(sequences, **read_fasta_kwargs):
    """Normalize the sequence input into ``(names, sequences)`` lists."""
    from sparrow.nardini.background import _iter_named_sequences

    names, seqs = [], []
    for name, sequence in _iter_named_sequences(sequences, **read_fasta_kwargs):
        names.append(name)
        seqs.append(sequence)
    return names, seqs


def nardini_vectors(
    sequences,
    background=None,
    num_scrambles=DEFAULT_NUM_SCRAMBLES,
    seed=None,
    typeset=DEFAULT_TYPESET,
    blob_size=DEFAULT_BLOB_SIZE,
    min_fraction=DEFAULT_MIN_FRACTION,
    backend=DEFAULT_BACKEND,
    show_progress=False,
    **read_fasta_kwargs,
):
    """Compute NARDINI+ z-score vectors for one or more sequences.

    This is step two of the NARDINI+ workflow. For each sequence it computes:

    * 36 **patterning** z-scores, each against a null distribution built by
      scrambling that sequence ``num_scrambles`` times; and
    * 54 **compositional** z-scores against ``background``, if one is given.

    Parameters
    ----------
    sequences : str, dict, or iterable
        The sequences to analyze. Either a path to a FASTA file, a mapping of
        ``{name: sequence_or_Protein}``, a single sequence string is *not*
        accepted (wrap it in a list or dict so it has a name), or an iterable of
        sequences / :class:`sparrow.protein.Protein` objects.
    background : NardiniBackground or str, optional
        The compositional background to z-score against, either a
        :class:`~sparrow.nardini.background.NardiniBackground` or a path to one
        saved with :meth:`~sparrow.nardini.background.NardiniBackground.save`.
        If None (default) only the 36 patterning features are returned - those
        need no background. Build one with
        :func:`sparrow.nardini.build_background`.
    num_scrambles : int, optional
        Scrambles used per sequence for the patterning null distribution.
        Default 10,000. The NARDINI paper uses 100,000; 1,000 is usually adequate
        for low-complexity sequences. Cost is linear in this value.
    seed : int or None, optional
        Seed for the scramble generator, for reproducible z-scores. Each sequence
        is seeded deterministically from this value, so results do not depend on
        how many sequences you analyze at once or in what order. Default None
        (non-reproducible).
    typeset : {'8', '9'} or dict, optional
        Residue grouping for the patterning features. Default ``'8'``.
    blob_size : int, optional
        Sliding-window size for the patterning features. Default 5.
    min_fraction : float, optional
        A residue type's fraction must exceed this for its patterning features to
        be scored; others are reported as 0.0. Default 0.10.
    backend : str, optional
        Which patterning statistic to use, one of
        :data:`sparrow.nardini.patterning.PATTERNING_BACKENDS`:

        * ``'nardini'`` (default) - the published NARDINI omega/kappa window
          asymmetry. **Required for GIN cluster assignment**, and the only
          backend whose numbers are comparable to published NARDINI values.
        * ``'kappa'`` - sparrow's generalized kappa (``kappa_x``), normalized
          against the maximally-segregated sequence and bounded 0-1.
        * ``'iwd'`` - sparrow's inverse-weighted distance, which measures
          clustering and ignores ``blob_size``.

        The legacy names ``'kappa_cython'`` and ``'iwd_combined'`` from
        :mod:`sparrow.sequence_analysis.grammar` are accepted as aliases.
        Only the patterning block changes; the compositional block is identical
        across backends.
    show_progress : bool, optional
        Show a progress bar across sequences. Default False.
    **read_fasta_kwargs
        Forwarded to ``protfasta.read_fasta`` when ``sequences`` is a filename.

    Returns
    -------
    NardiniResult
        Object holding the z-scores, the raw values, and the feature names, with
        a :meth:`~NardiniResult.to_dataframe` convenience method.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If no sequences were supplied, if a sequence is shorter than
        ``blob_size`` or contains non-standard residues, or if ``background``
        does not match the compositional feature set.

    See Also
    --------
    sparrow.nardini.build_background : Build the compositional background.
    sparrow.nardini.patterning.patterning_zscores : Patterning half only.

    Notes
    -----
    Patterning z-scores are self-referential (each sequence against its own
    scrambles) and so are comparable across sequences of different composition.
    Compositional z-scores are only interpretable relative to the background used,
    so record which background produced a given set of vectors - the background's
    ``metadata`` is designed for exactly that.

    Examples
    --------
    >>> from sparrow.nardini import build_background, nardini_vectors
    >>> background = build_background("idrome.fasta")            # doctest: +SKIP
    >>> result = nardini_vectors(                                 # doctest: +SKIP
    ...     "my_idrs.fasta", background=background, num_scrambles=10000, seed=42
    ... )
    >>> result.to_dataframe().shape                               # doctest: +SKIP
    (12, 90)
    """
    if isinstance(sequences, str) and not sequences.lower().endswith(
        (".fasta", ".fa", ".fas", ".fsa", ".faa", ".txt")
    ):
        # a bare sequence string is a likely mistake: it would be read as a path
        # and fail with a confusing "no such file" error
        if not os.path.isfile(sequences):
            raise SparrowException(
                "nardini_vectors() expects a FASTA filename, a dict of "
                "{name: sequence}, or an iterable of sequences. It looks like a "
                "bare sequence string was passed - wrap it so it has a name, "
                "e.g. nardini_vectors({'IDR1': 'MEEK...'}, background=bg)."
            )

    backend = resolve_backend(backend)

    if isinstance(background, (str, os.PathLike)):
        background = NardiniBackground.load(background)

    if background is not None:
        if tuple(background.feature_names) != COMPOSITION_FEATURE_NAMES:
            raise SparrowException(
                "The supplied background does not describe sparrow's NARDINI "
                f"compositional features (expected {len(COMPOSITION_FEATURE_NAMES)} "
                f"features, got {len(background.feature_names)}). Rebuild it with "
                "sparrow.nardini.build_background()."
            )

    names, seqs = _resolve_input(sequences, **read_fasta_kwargs)

    if len(seqs) == 0:
        raise SparrowException("No sequences were supplied to nardini_vectors().")

    patterning_names = patterning_feature_names(typeset)

    n_seqs = len(seqs)
    patt_z = np.zeros((n_seqs, len(patterning_names)), dtype=np.float64)
    patt_raw = np.zeros((n_seqs, len(patterning_names)), dtype=np.float64)

    iterator = range(n_seqs)
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="NARDINI", total=n_seqs)

    for index in iterator:
        sequence_seed = _derive_seed(seed, seqs[index])

        zscores, details = patterning_zscores(
            seqs[index],
            num_scrambles=num_scrambles,
            typeset=typeset,
            blob_size=blob_size,
            min_fraction=min_fraction,
            seed=sequence_seed,
            backend=backend,
            return_details=True,
        )
        patt_z[index] = zscores
        patt_raw[index] = details["raw"]

    # compositional block
    if background is None:
        comp_names = ()
        comp_z = np.zeros((n_seqs, 0), dtype=np.float64)
        comp_raw = np.zeros((n_seqs, 0), dtype=np.float64)
    else:
        comp_names = COMPOSITION_FEATURE_NAMES
        comp_raw = np.zeros((n_seqs, len(comp_names)), dtype=np.float64)
        for index, sequence in enumerate(seqs):
            values = composition_values(sequence)
            comp_raw[index] = [values[name] for name in comp_names]
        comp_z = background.zscores(comp_raw)

    return NardiniResult(
        names=names,
        sequences=seqs,
        patterning_names=patterning_names,
        composition_names=comp_names,
        patterning_zscores=patt_z,
        composition_zscores=comp_z,
        raw_patterning=patt_raw,
        raw_composition=comp_raw,
        backend=backend,
    )
