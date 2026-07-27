"""NARDINI / NARDINI+ analysis of intrinsically disordered regions.

NARDINI asks whether the residues in an IDR are arranged non-randomly. It
coarse-grains the amino acids into a handful of biophysical types and, for each
type and each pair of types, compares the observed sequence to scrambled versions
of itself. NARDINI+ extends that with a block of compositional features scored
against a reference set of IDRs.

**1. Get a compositional background.** Either use a published per-species IDRome
background that ships with sparrow, or build one from your own set of IDRs:

.. code-block:: python

    from sparrow.nardini import load_reference_background, build_background

    background = load_reference_background("human")     # published GIN prior
    # or
    background = build_background("my_idrome.fasta")
    background.save("my_idrome_background.npz")

**2. Compute NARDINI+ vectors for the IDRs you care about**

.. code-block:: python

    from sparrow.nardini import nardini_vectors

    result = nardini_vectors(
        "my_idrs.fasta",
        background=background,
        num_scrambles=10000,
        seed=42,
    )

    result.to_table()                   # 90 columns, one row per IDR
    result.top_features("IDR1", n=5)    # most non-random features
    result.to_dataframe()               # ... or pandas, if you prefer

The 90 features are 36 patterning z-scores (against each sequence's own
scrambles) plus 54 compositional z-scores (against the background). The
patterning half needs no background, so ``background`` may be omitted to get just
those 36 features.

**3. Optionally assign each IDR to a published GIN cluster**

.. code-block:: python

    result.gin_clusters().records()
    # [{'name': 'IDR1', 'cluster': 7, 'description': 'D/E-tracts',
    #   'distance': 41.2, 'separation': 6.31, 'is_typical': True, ...}, ...]

This maps each sequence onto one of the 30 published human GIN grammars. It needs
the full 90-feature vector, so a background is required - and because the
centroids live in human NARDINI+ z-score space, ``load_reference_background(
'human')`` is the background to use for results comparable to the published ones.

**4. For a whole proteome, compute in parallel**

.. code-block:: python

    from sparrow.nardini import nardini_proteins

    proteins = nardini_proteins(idr_sequences, num_scrambles=10000, seed=42,
                                show_progress=True)

    proteins["IDR1"].nardini.vector(seed=42, num_scrambles=10000)   # cached

:func:`~sparrow.nardini.batch.nardini_proteins` takes sequences or
:class:`~sparrow.protein.Protein` objects and returns Protein objects with the
results attached, so the expensive part happens once. Parallelism does not change
the numbers: per-sequence seeds derive from the sequence rather than its position
in a batch, so a vector is unaffected by how the input was chunked.

**5. Or derive your own grammars instead of using the published ones**

.. code-block:: python

    from sparrow.nardini import build_gin_reference, compare_gin_references

    reference = build_gin_reference(result, k=30, seed=0)
    reference.save("my_grammars.npz")
    compare_gin_references(reference)      # how close is this to published GIN?

:func:`~sparrow.nardini.clustering.build_gin_reference` runs the same clustering
procedure over your own vectors, which is what you want for a non-human proteome
or a designed library. Reproducing the *published* centroids exactly is a
different matter - see that module for why.

Where things live
-----------------
``vectors`` is the entry point and stitches the two halves together;
``patterning`` and ``composition`` compute those halves; ``background`` provides
the reference statistics the compositional half is scored against; ``gin``
assigns to cluster centroids and ``clustering`` builds new ones; ``batch``
parallelizes over many sequences; ``protein_interface`` is what
``Protein.nardini`` resolves to; ``table`` holds the small labelled containers
everything returns; ``groups`` defines the residue typesets. ``architecture.md``
in this directory describes how they fit together in more detail.

Choosing a patterning statistic
-------------------------------

The 36 patterning features can be computed three ways, selected with ``backend``:

* ``'nardini'`` (default) - the published NARDINI omega/kappa window asymmetry.
  Required for GIN cluster assignment.
* ``'kappa'`` - sparrow's generalized kappa (``kappa_x``), normalized against the
  maximally-segregated sequence and bounded 0-1.
* ``'iwd'`` - sparrow's inverse-weighted distance, a clustering metric that
  ignores ``blob_size``.

.. code-block:: python

    nardini_vectors(seqs, background=background, backend="kappa")

Only the patterning block changes; the compositional block is identical across
backends. The choice is recorded as ``result.backend``, because values from
different backends are not comparable and must not be mixed.

The last two supersede the ``'kappa_cython'`` and ``'iwd_combined'`` backends of
the deprecated :mod:`sparrow.sequence_analysis.grammar`, whose values they
reproduce exactly; those legacy names are accepted as aliases.

References
----------
Cohan, M. C., Shinn, M. K., Lalmansingh, J. M. & Pappu, R. V. Uncovering
non-random binary patterns within sequences of intrinsically disordered
proteins. J. Mol. Biol. 434, 167373 (2022).

King, M. R. et al. Macromolecular condensation organizes nucleolar sub-phases
and sets up a pH gradient. Cell 187, 1889-1906.e24 (2024). [NARDINI+]

Ruff, K. M. et al. Molecular grammars of predicted intrinsically disordered
regions that span the human proteome. Cell (2025). [GIN resource]

If you use this module, please cite the original NARDINI work above.
"""

from sparrow.nardini.background import (
    REFERENCE_SPECIES,
    NardiniBackground,
    build_background,
    load_reference_background,
)
from sparrow.nardini.composition import (
    COMPOSITION_FEATURE_NAMES,
    composition_feature_names,
    composition_matrix,
    composition_values,
)
from sparrow.nardini.batch import nardini_proteins
from sparrow.nardini.clustering import (
    DEFAULT_NUM_CLUSTERS,
    build_gin_reference,
    compare_gin_references,
    describe_centroid,
)
from sparrow.nardini.gin import (
    GIN_CLIP,
    TYPICAL_COMPOSITION_RMS_Z,
    TYPICAL_MAX_SATURATED,
    GINAssignment,
    GINReference,
    assign_gin_clusters,
    gin_cluster_definitions,
    load_gin_reference,
)
from sparrow.nardini.table import FeatureTable, FeatureVector
from sparrow.nardini.groups import (
    TYPES_8,
    TYPES_9,
    get_typeset,
    patterning_feature_names,
    type_names,
)
from sparrow.nardini.patterning import (
    DEFAULT_BACKEND,
    DEFAULT_BLOB_SIZE,
    DEFAULT_MIN_FRACTION,
    DEFAULT_NUM_SCRAMBLES,
    PATTERNING_BACKENDS,
    patterning_values,
    patterning_zscores,
    resolve_backend,
)
from sparrow.nardini.vectors import NardiniResult, nardini_vectors

__all__ = [
    # step 1 - background
    "build_background",
    "load_reference_background",
    "NardiniBackground",
    "REFERENCE_SPECIES",
    # step 2 - vectors
    "nardini_vectors",
    "NardiniResult",
    # step 3 - GIN cluster assignment
    "assign_gin_clusters",
    "gin_cluster_definitions",
    "load_gin_reference",
    "GINAssignment",
    "GINReference",
    "GIN_CLIP",
    # whole-proteome computation
    "nardini_proteins",
    # building your own clusters
    "build_gin_reference",
    "compare_gin_references",
    "describe_centroid",
    "DEFAULT_NUM_CLUSTERS",
    "TYPICAL_COMPOSITION_RMS_Z",
    "TYPICAL_MAX_SATURATED",
    # labelled containers
    "FeatureTable",
    "FeatureVector",
    # patterning
    "patterning_zscores",
    "patterning_values",
    "patterning_feature_names",
    "PATTERNING_BACKENDS",
    "resolve_backend",
    "DEFAULT_BACKEND",
    # composition
    "composition_values",
    "composition_matrix",
    "composition_feature_names",
    "COMPOSITION_FEATURE_NAMES",
    # groupings
    "TYPES_8",
    "TYPES_9",
    "get_typeset",
    "type_names",
    # defaults
    "DEFAULT_BLOB_SIZE",
    "DEFAULT_MIN_FRACTION",
    "DEFAULT_NUM_SCRAMBLES",
]
