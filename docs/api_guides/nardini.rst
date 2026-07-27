NARDINI / NARDINI+ (``sparrow.nardini``)
========================================

`NARDINI <https://www.sciencedirect.com/science/article/pii/S0022283621006100>`_
asks a specific question about an intrinsically disordered region: **are its
residues arranged non-randomly?** It coarse-grains the amino acids into a handful
of biophysical types and, for each type and each pair of types, compares the
observed sequence against scrambled versions of itself. Because scrambling
preserves composition exactly, the result isolates *patterning* from
*composition*.

NARDINI+ adds a second block of features describing composition itself, scored
against a reference set of IDRs. On top of that, the **GIN** resource groups the
human IDRome into 30 clusters of shared molecular grammar, and this module can
assign new sequences to them.

This module supersedes sparrow's older ``sequence_analysis.grammar`` -- see
:ref:`nardini-migration`.

.. tip::
   Two worked example notebooks live in ``examples/`` in the repository, both
   executed against a real human IDR:

   * ``nardini_from_protein_objects.ipynb`` -- the single-sequence interface,
     via ``Protein.nardini``.
   * ``nardini_module.ipynb`` -- the batch interface: building your own
     compositional background, scoring a set of sequences, and GIN assignment
     across a dataset.

.. contents:: On this page
   :local:
   :depth: 2

.. note::
   Working on the module rather than with it? ``sparrow/nardini/architecture.md``
   describes how the files fit together, how they connect to ``Protein``, and the
   invariants that are not obvious from any single file.

From a ``Protein``
------------------

For a single sequence, everything is reachable from the
:attr:`~sparrow.protein.Protein.nardini` accessor:

.. code-block:: python

   from sparrow import Protein

   p = Protein("MENDPRVRKFASDGAQWAIKWQKK...")

   p.nardini.vector(seed=42)        # 90-feature NARDINI+ vector
   p.nardini.patterning(seed=42)    # just the 36 patterning z-scores
   p.nardini.composition()          # just the 54 compositional z-scores
   p.nardini.gin_cluster(seed=42)   # nearest published human GIN grammar
   print(p.nardini.gin_report(seed=42))

Compositional z-scores need a reference IDR set, and for one protein there is
nothing to build one from, so these default to the **published human IDRome
background** bundled with sparrow -- which is also the space the GIN centroids
live in. Override with ``background="yeast"``, a saved ``.npz`` path, or a
:class:`~sparrow.nardini.background.NardiniBackground`. The patterning half never
uses it.

See :ref:`nardini-protein` for the full accessor.

The batch workflow
------------------

.. code-block:: python

   from sparrow.nardini import load_reference_background, nardini_vectors

   # 1. get a compositional background -- published, or built from your own IDRs
   background = load_reference_background("human")

   # 2. compute NARDINI+ vectors for the IDRs you care about
   result = nardini_vectors(
       "my_idrs.fasta",
       background=background,
       num_scrambles=10000,
       seed=42,
   )

   result.to_table()                    # 90 columns, one row per IDR
   result.top_features("IDR1", n=5)     # the most non-random features

   # 3. optionally, assign each IDR to a published GIN cluster
   result.gin_clusters().records()

.. important::
   Only the **compositional** half of a NARDINI+ vector uses the background. The
   **patterning** half is z-scored against scrambles of each individual sequence,
   so it needs no external reference at all. If you only want patterning
   features, omit ``background`` and you will get a 36-feature vector.

Step 1 -- getting a background
------------------------------

Published per-species backgrounds
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

sparrow bundles the per-species IDRome backgrounds from the GIN resource -- the
same tables the official NARDINI+ notebook downloads at runtime. Using one makes
your compositional z-scores directly comparable to published NARDINI+ values:

.. code-block:: python

   from sparrow.nardini import load_reference_background

   background = load_reference_background("human")

Available species keys: ``human``, ``mouse``, ``rat``, ``xenopus``, ``fly``,
``zebrafish``, ``yeast``, ``worm``, ``arabidopsis``. The Latin name works too
(``load_reference_background("Homo sapiens")``).

.. important::
   The GIN cluster centroids were defined in **human** NARDINI+ z-score space, so
   use the ``'human'`` background if you intend to assign GIN clusters (step 3).

These are redistributed under the MIT licence and **must be cited** -- see
:ref:`nardini-citations`.

Building your own
^^^^^^^^^^^^^^^^^

:func:`~sparrow.nardini.background.build_background` estimates the mean and
standard deviation of each of the 54 compositional features across a reference
set of IDRs of your choosing. The reference set defines what "typical" means, so
choose it to match the question you are asking -- a whole IDRome for "unusual for
this organism", a filtered sub-proteome for something narrower.

.. code-block:: python

   from sparrow.nardini import build_background

   # from a FASTA file
   background = build_background("idrome.fasta")

   # or from a dict of sequences / Protein objects
   background = build_background({"idr1": "MEEEKKK...", "idr2": "QQQQNNN..."})

   # or from any iterable of sequences
   background = build_background(list_of_sequences)

Useful options:

* ``min_length`` -- skip sequences shorter than this. The default of 1 keeps
  everything (matching the reference implementation); raising it is sensible if
  your IDR set contains very short fragments, whose compositional features are
  noisy.
* ``metadata`` -- free-form provenance stored with the background, e.g.
  ``{'species': 'Homo sapiens', 'idrome_version': '3'}``. Recording this matters,
  because a compositional z-score is only interpretable relative to its
  background.
* ``show_progress`` -- progress bar, worth it for a whole IDRome.

Sequences containing non-standard residues (``X``, ``U``, ``Z``, ``J``, ``B``,
``O``) are skipped, matching the reference implementation. The counts of used and
skipped sequences are recorded in ``background.metadata``.

Backgrounds are cheap to compute but worth saving so that results stay
comparable across analyses:

.. code-block:: python

   background.save("idrome_background.npz")

   from sparrow.nardini import NardiniBackground
   background = NardiniBackground.load("idrome_background.npz")

Means and standard deviations are accumulated online (Welford's algorithm), so
memory does not grow with the size of the IDRome.

How good is my background?
^^^^^^^^^^^^^^^^^^^^^^^^^^

A weak background does not fail loudly - it quietly produces z-scores you should
not believe. There are two ways to get one, and
:func:`~sparrow.nardini.background.build_background` warns about both with a
:class:`~sparrow.sparrow_exceptions.BackgroundQualityWarning`:

**Too few sequences.** The relative standard error on an estimated standard
deviation is about ``1 / sqrt(2 (n - 1))``, so at 50 sequences the per-feature
scale is only good to ~10% and at 13 sequences to ~20%. Below
:data:`~sparrow.nardini.background.MIN_RECOMMENDED_SEQUENCES` (50), sparrow warns.
The published IDRomes are built from thousands.

**Constant features.** If a feature never varies across your reference set, there
is no scale to divide by, so its z-score is reported as exactly ``0.0`` for every
sequence you ever score against it. That reads as "perfectly average" when it
actually means "unmeasurable". sparrow warns above
:data:`~sparrow.nardini.background.MAX_DEGENERATE_FRACTION` (10% of the 54
features), and records the count in ``background.metadata`` either way:

.. code-block:: python

   background.metadata["num_degenerate_features"]

For calibration: of the nine published IDRomes, seven have **zero** constant
features and two (yeast, worm) have one - so a good reference set sits at 0-2%.
Small or synthetic sets are much worse; 60 randomly-generated IDR-like sequences
give about 8 of 54, and a single sequence gives all 54.

The patch features are the usual culprits, since a patch needs an actual run of a
residue to register. If you are working from a small set and only care about
patterning, pass ``background=None`` - the patterning half never uses a background
at all, and is unaffected by any of this.

To silence the warning when you know your set is small:

.. code-block:: python

   import warnings
   from sparrow.sparrow_exceptions import SparrowWarning

   warnings.filterwarnings("ignore", category=SparrowWarning)

Step 2 -- computing NARDINI+ vectors
------------------------------------

:func:`~sparrow.nardini.vectors.nardini_vectors` accepts a FASTA path, a dict of
``{name: sequence_or_Protein}``, or an iterable of sequences:

.. code-block:: python

   result = nardini_vectors(
       {"IDR1": seq1, "IDR2": seq2},
       background=background,
       num_scrambles=10000,
       seed=42,
   )

   result.zscores                  # (n_sequences, 90) plain ndarray
   result.feature_names            # 36 patterning names + 54 composition names
   result.patterning_zscores       # (n_sequences, 36)
   result.composition_zscores      # (n_sequences, 54)

   result.to_table()               # names attached to the array (default view)
   result.to_table(raw=True)       # the un-z-scored values
   result.vector("IDR1")           # one sequence, names attached

Results are plain NumPy, optionally with the feature names attached - see
:ref:`nardini-containers`. Nothing returns a DataFrame unless you ask for one
with ``.to_dataframe()``.

Key options:

* ``num_scrambles`` -- scrambles per sequence for the patterning null
  distribution. Default 10,000. The published protocol uses 100,000; 1,000 is
  usually adequate for low-complexity sequences. Cost is linear in this value.
* ``seed`` -- makes the patterning z-scores reproducible. Each sequence is seeded
  deterministically from this value, so a given sequence's vector does not depend
  on how many other sequences were in the batch or in what order.
* ``typeset`` -- ``'8'`` (default) or ``'9'``; see below.
* ``n_processes`` -- compute across this many processes. Default 1. The result is
  bit-identical to the serial one however the work is divided, since each
  sequence's seed comes from its own text rather than its position. Measured on 64
  IDRs at 10,000 scrambles: 22.1 s serial, 5.9 s on four processes, 3.3 s on eight
  (6.8x). Use :func:`~sparrow.nardini.batch.nardini_proteins` instead if you want
  ``Protein`` objects back, or need to skip unscoreable sequences.
* ``min_fraction`` -- abundance threshold, default 0.10; see below.
* ``backend`` -- which patterning statistic to use; see the next section.
* ``show_progress`` -- progress bar across sequences.

.. _nardini-backends:

Choosing a patterning backend
-----------------------------

"How non-randomly are these residues arranged?" can be quantified in more than
one way, and sparrow offers three. The ``backend`` argument selects between them.
**It affects only the 36 patterning features** -- the 54 compositional features
are computed identically regardless.

.. list-table::
   :header-rows: 1
   :widths: 12 30 30 28

   * - ``backend``
     - Statistic
     - Range / behaviour
     - Use it when
   * - ``'nardini'``
       *(default)*
     - The published NARDINI omega/kappa window asymmetry.
     - Unbounded and positive; scales with how uneven the windows are.
     - You want NARDINI, NARDINI+, or GIN. **Required for GIN cluster
       assignment.**
   * - ``'kappa'``
     - sparrow's generalized kappa (``kappa_x``), normalized against the
       maximally-segregated sequence.
     - Bounded 0-1, so raw values are comparable between sequences before
       z-scoring.
     - You want a normalized, interpretable raw patterning value, or you are
       continuing work that used ``Protein.extract_feature_vector``.
   * - ``'iwd'``
     - sparrow's inverse-weighted distance -- a clustering metric over residue
       positions.
     - Unbounded; **ignores** ``blob_size`` entirely, since it has no window.
     - You care about clustering at any lengthscale rather than within a fixed
       window.

.. code-block:: python

   nardini_vectors(seqs, background=bg, backend="nardini")   # published NARDINI
   nardini_vectors(seqs, background=bg, backend="kappa")     # normalized kappa_x
   nardini_vectors(seqs, background=bg, backend="iwd")       # clustering-based

The backend used is recorded on the result as ``result.backend``, because the
numbers are **not comparable across backends** -- do not mix them in one analysis
or feed one backend's vectors into a model trained on another's.

The legacy names from the deprecated ``sparrow.sequence_analysis.grammar`` module
are accepted as aliases: ``'kappa_cython'`` → ``'kappa'`` and ``'iwd_combined'``
→ ``'iwd'``. Those two backends reproduce that module's values *exactly*, so
migrating changes no numbers.

.. _nardini-delta-vs-kappa:

``'nardini'`` and ``'kappa'`` measure the same thing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Worth being explicit about, because the naming invites confusion. Both are built
on the same Das & Pappu window asymmetry, but they stop at different points:

* ``'nardini'`` z-scores **delta** -- the mean squared deviation of the per-window
  asymmetry from the whole-sequence asymmetry. This is the published NARDINI
  statistic, and it is *unnormalized*.
* ``'kappa'`` z-scores **kappa**, which is that same delta divided by
  ``delta_max``: the delta of the most segregated arrangement of the same
  composition. This is sparrow's :func:`~sparrow.patterning.kappa.kappa_x`,
  bounded 0-1.

**The two z-scores are the same quantity.** ``delta_max`` depends only on
composition, and scrambling preserves composition exactly, so ``delta_max`` is the
*same constant* for a sequence and for every member of its null distribution. A
constant factor cancels out of a z-score, so normalizing changes nothing once the
value has been referenced against its own scrambles.

In practice they are very close but not bit-identical, for two reasons that have
nothing to do with the normalization: ``kappa_x`` scores a *winged* (end-padded)
array, which changes slightly which windows exist, and it caps values at 1, which
is a nonlinear step. Measured over a range of sequences at 4,000 scrambles, the
per-feature z-scores correlate at 0.97-1.00, differ by at most ~0.2, and agree in
sign for 31 of 32 evaluated features.

So **``'kappa'`` is offered for completeness, not because it tells you something
different**: it exists so that results from the deprecated
``sequence_analysis.grammar`` module can be reproduced, and so the normalized form
is available if you want it. Use ``'nardini'`` -- the default -- unless you have a
specific reason not to. It is also the fastest of the three, and the only one GIN
cluster assignment accepts.

What the choice looks like in practice, on a blocky vs. a well-mixed E/K
sequence (``pos-neg`` z-score, 500 scrambles):

.. code-block:: text

   backend     blocky EK    mixed EK
   nardini        +12.81       -1.26
   kappa          +11.45       -0.93
   iwd              0.00        0.00

The two window-based statistics agree on the qualitative answer. ``iwd`` returns
zero here for a legitimate reason worth understanding: with only E and K present,
the merged ``{D,E,K,R}`` group covers *every* position, so the inverse-weighted
distance cannot change under scrambling. The null is a point mass, and a z-score
of 0 is the correct report. On real, compositionally diverse IDRs ``iwd``
behaves normally.

.. _nardini-backend-performance:

Backend performance
^^^^^^^^^^^^^^^^^^^

All three backends are vectorized over scrambles, and all are fast enough that
the choice is a scientific one. ``'nardini'``, the default, is now also the
quickest: its inner loop skips the features and residue types that the abundance
gate has already excluded, which the other two cannot do as cleanly. Wall-clock
for 10,000 scrambles of a single sequence:

.. code-block:: text

   length     nardini     kappa       iwd
     200        0.28 s    0.43 s    0.36 s
     500        0.51 s    0.90 s    0.67 s
    1000        0.81 s    1.50 s    1.45 s

Scaled to the published 100,000 scrambles, a 1000-residue IDR takes roughly
8 s, 15 s and 15 s respectively. For a whole proteome, divide by however many
cores you give :func:`~sparrow.nardini.batch.nardini_proteins`.

This is a recent change. ``'kappa'`` and ``'iwd'`` originally delegated to
sparrow's Cython extensions, which score one sequence per call and made those
backends 9-22x slower than ``'nardini'`` (9 minutes for that same 1000-residue
IDR). Both statistics have since been reimplemented in vectorized NumPy in
:mod:`sparrow.patterning.vectorized`, which evaluates a whole block of scrambles
per call.

.. note::
   The NumPy reimplementations carry their intermediates in ``float64`` where the
   Cython extensions use C ``float``. For IWD both are double precision and agree
   to ~1e-15; for kappa they agree to ~1e-6, and the NumPy value is the more
   accurate of the two. ``sparrow/tests/test_patterning_vectorized.py`` pins both
   claims, including a test that replays the Cython arithmetic in single
   precision to demonstrate the residual gap is the compiled code's precision and
   not an error here.

Step 3 -- GIN cluster assignment
--------------------------------

The GIN resource partitions the human IDRome into **30 clusters of shared
molecular grammar** -- ``'D/E-tracts'``, ``'S patches'``, ``'Well-mixed P and G'``
and so on -- obtained by k-means on NARDINI+ z-score vectors of human IDRs of
length 100-300. sparrow bundles those published centroids and can assign new
sequences to them:

.. code-block:: python

   from sparrow.nardini import load_reference_background, nardini_vectors

   result = nardini_vectors(
       "my_idrs.fasta",
       background=load_reference_background("human"),
       num_scrambles=10000,
       seed=42,
   )

   result.gin_clusters().records()

.. code-block:: text

                 cluster                              description  distance  separation
   Name
   de_tract            7                               D/E-tracts     19.72       24.90
   s_patch             6                                S patches     28.41       17.24
   q_tract            11                                 Q-tracts     14.06       36.41
   notebook_idr       17                     Weak positive charge     33.11        6.00

This is *assignment*, not clustering: your sequences are matched to the published
grammars, nothing is re-clustered. The procedure is the published one --

1. take the full 90-feature NARDINI+ z-score vector;
2. clip it to ±3 (the centroids were built from clipped vectors);
3. compute the squared Euclidean distance to each of the 30 centroids;
4. assign to the nearest.

Interpreting ``separation``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``separation`` is the gap between the best and second-best centroid distance, and
it is the number to watch. Every sequence gets *some* cluster -- nearest-centroid
assignment never abstains -- so ``separation`` is what tells you whether the label
is worth believing.

**A small separation means the sequence sits between two grammars** and should not
be treated as firmly belonging to either. There is no universal cutoff; compare
within your own dataset. In the table above, ``q_tract`` (36.4) is unambiguous
while ``notebook_idr`` (6.0) is genuinely borderline. When that happens, look at
the runners-up rather than trusting the single label:

.. code-block:: python

   assignment = result.gin_clusters()
   assignment.ranked_clusters("notebook_idr", n=5)
   # [(17, 'Weak positive charge', 33.11),
   #  (29, 'High R fraction', 39.17),
   #  (10, 'Well-mixed hydrophobics, enriched in M', 44.33), ...]

Here the top three are all positive-charge-flavoured grammars, so the *family* is
solid even though the exact cluster is not -- a much more honest summary than
"cluster 17".

What the clusters mean
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sparrow.nardini import gin_cluster_definitions

   gin_cluster_definitions()

Returns all 30 published clusters with a description and the number of human IDRs
assigned to each in the original analysis (24,508 in total). A few examples:

.. code-block:: text

   cluster  description                                    num_idrs
         6  S patches                                          1833
         7  D/E-tracts                                          763
        11  Q-tracts                                            338
        20  A blocks                                            803
        23  K blocks                                           1043
        27  P patches                                          1386
        29  High R fraction                                    1482

The counts are useful context: an assignment to cluster 2 (163 IDRs) is a much
more specific statement than one to cluster 6 (1,833).

Everything else on the result
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   assignment.clusters           # (n,) assigned cluster number
   assignment.descriptions       # (n,) published description of each
   assignment.distances          # (n,) distance to the assigned centroid
   assignment.separations        # (n,) gap to the runner-up
   assignment.n_saturated        # (n,) features pinned at the +/-3 clip
   assignment.composition_rms_z  # (n,) how far from the reference IDRome
   assignment.is_typical         # (n,) bool - is the label trustworthy?
   assignment.all_distances      # (n, 30) distance to every centroid

   assignment.records()          # list of dicts (the default view)
   assignment.to_dataframe()     # pandas, if you prefer

``all_distances`` is the raw material for anything more sophisticated -- soft
assignments, distance-to-cluster heatmaps, or nearest-neighbour analyses across
your dataset.

Requirements and failure modes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GIN assignment is strict about its two preconditions, because getting either
wrong produces plausible-looking but meaningless labels rather than an error:

1. **It needs the full 90-feature vector.** A result computed without a
   ``background`` has only the 36 patterning features and raises.
2. **It needs the** ``'nardini'`` **backend.** The centroids were fit to vectors
   built with the published NARDINI statistic, so a ``'kappa'`` or ``'iwd'``
   vector is on an entirely different scale. sparrow refuses rather than
   assigning:

   .. code-block:: python

      result = nardini_vectors(seqs, background=bg, backend="kappa")
      result.gin_clusters()
      # SparrowException: GIN cluster assignment requires the 'nardini'
      # patterning backend, but this result was computed with 'kappa'. ...

.. important::
   The centroids live in **human** NARDINI+ z-score space. Pass
   ``load_reference_background('human')`` for results comparable to the published
   ones. Assignment against another background is allowed (the reference
   implementation permits it) but the further that background is from the human
   IDRome, the less the cluster labels mean. sparrow does not block this, because
   it is sometimes what you want -- just be aware of what you are asking.

A complete GIN workflow
^^^^^^^^^^^^^^^^^^^^^^^

Putting it together, from a FASTA of IDRs to an annotated table:

.. code-block:: python

   import numpy as np

   from sparrow.nardini import (
       load_reference_background,
       nardini_vectors,
       gin_cluster_definitions,
   )

   # 1. published human background -- the space the GIN centroids live in
   background = load_reference_background("human")

   # 2. full NARDINI+ vectors, published settings, reproducible
   result = nardini_vectors(
       "my_idrs.fasta",
       background=background,
       num_scrambles=100000,     # the published protocol
       seed=42,
       backend="nardini",        # the default; stated here for clarity
       show_progress=True,
   )

   # 3. assign, and keep only the calls worth believing
   assignment = result.gin_clusters()
   sizes = {d["cluster"]: d["num_idrs"] for d in gin_cluster_definitions()}

   for record in assignment.records():
       if not record["is_typical"]:
           print(f"{record['name']}: outside the human IDRome "
                 f"(composition RMS z = {record['composition_rms_z']:.1f}) - skipping")
           continue
       if record["separation"] < 10:
           print(f"{record['name']}: sits between grammars, "
                 f"see {assignment.ranked_clusters(record['name'], n=3)}")
       print(f"{record['name']}: cluster {record['cluster']} "
             f"({record['description']}, {sizes[record['cluster']]} human IDRs)")

   # and keep the vectors themselves for downstream modelling
   np.savetxt(
       "my_idrs_nardini_zscores.csv", result.zscores, delimiter=",",
       header=",".join(result.feature_names), comments="",
   )
   # or, via pandas:
   # result.to_dataframe().to_csv("my_idrs_nardini_zscores.csv")

.. _nardini-proteome:

A whole proteome, in parallel
-----------------------------

:func:`~sparrow.nardini.batch.nardini_proteins` computes vectors for many
sequences across multiple processes and hands back
:class:`~sparrow.protein.Protein` objects with the results already attached, so the
expensive work happens once and everything afterwards is free:

.. code-block:: python

   from sparrow.nardini import nardini_proteins

   proteins = nardini_proteins(idr_sequences, num_scrambles=10000, seed=42,
                               show_progress=True)

   proteins["IDR1"].nardini.vector(seed=42, num_scrambles=10000)   # cached
   proteins["IDR1"].nardini.gin_cluster(seed=42, num_scrambles=10000)

Input and output shapes match: a dict returns a dict with the same keys, anything
else returns a list in input order. Sequences become new ``Protein`` objects;
``Protein`` objects you pass in are populated **in place** and handed back, so you
can keep whatever references you already had.

.. code-block:: python

   nardini_proteins({"IDR1": "MEEEKKK...", "IDR2": "QQQQNNN..."})   # dict of sequences
   nardini_proteins(["MEEEKKK...", "QQQQNNN..."])                   # list of sequences
   nardini_proteins(my_proteins)                                    # Protein objects
   nardini_proteins({"a": "MEEE...", "b": Protein("QQQQ...")})      # or a mix

Measured scaling on 160 IDRs at 1,000 scrambles: 18.7 s serially, 9.8 s on two
processes, 5.1 s on four, 2.7 s on eight (6.9x). ``n_processes`` defaults to
``os.cpu_count()``.

.. important::
   **Parallelism does not change the numbers.** Per-sequence seeds derive from the
   sequence itself rather than its position in a batch, so a vector is unaffected
   by how the input was chunked, how many processes ran, or what order they
   finished in. The parallel result is identical to the serial one and identical to
   what each Protein would compute alone -- asserted directly in
   ``test_nardini_batch.py``.

Malformed sequences
^^^^^^^^^^^^^^^^^^^

Real proteomes contain entries NARDINI cannot score -- a non-standard residue, or
something shorter than the window. By default the run raises, naming the offender.
For a job measured in hours that is rarely what you want, so ``on_error='skip'``
drops them, keeps everything else, and warns with a summary:

.. code-block:: python

   proteins = nardini_proteins(idrs, on_error="skip", seed=42)

The returned collection is then smaller than the input. One bad sequence never
costs more than its own chunk: a chunk that fails is retried one sequence at a
time to isolate the offender.

.. note::
   **On ``spawn`` platforms, call this from inside a**
   ``if __name__ == "__main__":`` **block.**

   sparrow defaults to the ``fork`` start method where it exists (Linux, macOS),
   which sidesteps the issue entirely. Under ``spawn`` -- Windows, or if you pass
   ``start_method='spawn'`` -- every worker re-imports the calling ``__main__``
   module, so a script that calls ``nardini_proteins`` at module level calls it
   again in each worker, recursively. sparrow detects that it is running inside a
   worker and falls back to computing serially rather than spawning further, so
   the worst case is wasted work rather than a runaway process, but the guard is
   still the right thing to write.

   ``fork`` is the default because a Jupyter notebook or interactive session has no
   ``__main__`` guard available at all. The tradeoff is that forking a process
   holding threaded state can deadlock; the NARDINI path is NumPy and SciPy only
   and never touches the predictor networks, so this is safe in practice. Pass
   ``start_method='spawn'`` if you would rather not rely on that.

Worker start-up costs about a second each, since sparrow has to be imported in
every process -- so this is worth it for hundreds of sequences, not for five. Below
16 sequences, or with ``n_processes=1``, it runs in-process without a pool.


.. _nardini-own-clusters:

Building your own clusters
--------------------------

Everything above *assigns* sequences to the 30 published human grammars.
:func:`~sparrow.nardini.clustering.build_gin_reference` does the other half: it
runs the same procedure over your own vectors and hands back a
:class:`~sparrow.nardini.gin.GINReference` that ``assign_gin_clusters`` accepts in
place of the bundled one.

.. code-block:: python

   from sparrow.nardini import (
       build_background, nardini_vectors, build_gin_reference,
       assign_gin_clusters, compare_gin_references,
   )

   background = build_background("my_idrome.fasta")
   vectors = nardini_vectors("my_idrome.fasta", background=background,
                             num_scrambles=10000, seed=42)

   reference = build_gin_reference(vectors, k=30, seed=0)
   reference.save("my_idrome_grammars.npz")        # the vectors cost hours; save them

   assign_gin_clusters(vectors, reference=reference)

This is what you want for a non-human proteome, a filtered sub-proteome, or a
designed library - anywhere forcing your sequences onto human grammars would be
the wrong move. ``k`` defaults to 30 to match the published analysis, but there is
nothing special about 30 for your data.

Clusters are numbered largest-first, and each gets a description derived from its
centroid's most extreme features:

.. code-block:: text

   cluster 0 (n=15): low Frac Aliphatic, high Frac D, high Frac E
   cluster 1 (n=15): S patches, high Frac Polar, high Frac S
   cluster 2 (n=15): high Frac K, high Frac K+R, low Frac Aliphatic
   cluster 3 (n=15): high Frac Polar, high Frac Q, Q patches

Those are generated, not published prose - pass ``descriptions={0: '...'}`` to
override any of them. Note that cluster *numbers* mean nothing across runs: k-means
numbering is arbitrary, and sparrow's largest-first ordering only makes it stable
within a run.

.. _nardini-cluster-contents:

How big is each cluster, and what is in it?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Cluster sizes** are recorded on the reference as
:attr:`~sparrow.nardini.gin.GINReference.num_idrs`, a ``{cluster: count}`` dict.
Clusters are numbered largest-first, so cluster 0 is always the most populated:

.. code-block:: python

   for cluster in reference.clusters:
       print(cluster, reference.num_idrs[cluster], reference.describe(cluster))

.. code-block:: text

   0 812 blocky neg-neg, high Frac D, high Frac E
   1 693 well-mixed pol-gly, high Frac G
   2 455 high Frac Q, Q patches

:meth:`~sparrow.nardini.gin.GINReference.definitions` gives the same information as
a list of dicts, which is handier for sorting or writing out::

   reference.definitions()
   # [{'cluster': 0, 'description': 'blocky neg-neg, ...', 'num_idrs': 812}, ...]

**Which IDRs are in each cluster** is a property of an *assignment*, not of the
reference -- the reference stores only the centroids and how many sequences formed
each one. Assign your sequences, then group the records by cluster:

.. code-block:: python

   import collections

   assignment = assign_gin_clusters(vectors, reference=reference)

   members = collections.defaultdict(list)
   for record in assignment.records():
       members[record["cluster"]].append(record["name"])

Now iterate the clusters, and the IDRs within each:

.. code-block:: python

   for cluster in reference.clusters:
       names = members[cluster]
       print(f"cluster {cluster}: {reference.describe(cluster)}  ({len(names)} IDRs)")
       for name in names:
           print("   ", name)

.. code-block:: text

   cluster 0: low Frac Aliphatic, high Frac E, high Frac D  (8 IDRs)
       DE_0
       DE_1
       DE_2

Each record carries the diagnostics alongside the label, so you can filter while
iterating -- for instance to list only the IDRs whose assignment is trustworthy:

.. code-block:: python

   for record in assignment.records():
       if record["is_typical"] and record["separation"] > 5:
           print(record["cluster"], record["name"], round(record["distance"], 1))

Printing the sequences in one cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The records carry names, not sequences -- but the result you scored already holds
both, in the same order, so one ``zip`` gives you a lookup:

.. code-block:: python

   sequence_of = dict(zip(vectors.names, vectors.sequences))

   cluster = 1
   print(f"cluster {cluster}: {reference.describe(cluster)}")

   for record in assignment.records():
       if record["cluster"] == cluster:
           print(f">{record['name']}")
           print(sequence_of[record["name"]])

.. code-block:: text

   cluster 1: high Frac S, S patches, high Disorder Promoting
   >Srich_0
   SGPSSPTSSPPSSSSSSSSSTSTPSSSPSGSSSPSTSTSSGSSGSTSGSSTTSSPPSGSS...
   >Srich_1
   SGSSSSGGSPSSGSPTSSSSGSTGSGSSTTPGSTSSSSSSSPSSSPTGSSTSSSSSSSSP...

That output is already FASTA, so writing a cluster out for use elsewhere -- an
alignment, a motif search, a figure -- is the same loop pointed at a file:

.. code-block:: python

   with open(f"cluster_{cluster}.fasta", "w") as handle:
       for record in assignment.records():
           if record["cluster"] == cluster:
               handle.write(f">{record['name']}\n{sequence_of[record['name']]}\n")

To do this for every cluster at once, reuse the ``members`` grouping from above:

.. code-block:: python

   for cluster in reference.clusters:
       with open(f"cluster_{cluster}.fasta", "w") as handle:
           for name in members[cluster]:
               handle.write(f">{name}\n{sequence_of[name]}\n")

If you started from a plain ``{name: sequence}`` dict you can of course index that
directly instead of building ``sequence_of`` -- the names are unchanged all the way
through.

If you used :func:`~sparrow.nardini.batch.nardini_proteins` you have ``Protein``
objects under the same names, which is more useful again, since you can go
straight from a cluster to any other sparrow property:

.. code-block:: python

   proteins = nardini_proteins(idr_sequences, background=background, seed=42)

   for name in members[cluster]:
       protein = proteins[name]
       print(name, len(protein), round(protein.FCR, 3), round(protein.kappa, 3))

.. important::
   ``reference.num_idrs`` and the sizes you get from an assignment answer different
   questions, and coincide only when you assign back exactly the set you clustered.
   ``num_idrs`` is a fixed property of the reference: how many sequences formed each
   centroid. Assigning a *different* set gives that set's distribution across the
   same centroids -- which is the point of saving a reference and reusing it.

Any backend works
^^^^^^^^^^^^^^^^^

Centroids live in the space of whatever statistic produced them, so a reference
records its backend and ``assign_gin_clusters`` refuses to mix spaces:

.. code-block:: python

   kappa = nardini_vectors(seqs, background=bg, backend="kappa")

   assign_gin_clusters(kappa)                      # refused: published centroids are 'nardini'
   reference = build_gin_reference(kappa, k=10, seed=0)
   assign_gin_clusters(kappa, reference=reference)  # fine: both in 'kappa' space

Did I recover GIN?
^^^^^^^^^^^^^^^^^^

Cluster numbering is arbitrary, so comparing labels is meaningless.
:func:`~sparrow.nardini.clustering.compare_gin_references` pairs your centroids
one-to-one against another reference's by solving the optimal assignment (the
Hungarian algorithm) and reports the distances:

.. code-block:: python

   report = compare_gin_references(reference)      # against the published human GIN

   report["mean_distance"]                # typical squared distance to its match
   report["separation_scale"]             # spacing of the published centroids
   report["reference_separation_scale"]   # spacing of yours
   report["pairs"][0]["matched_description"]

Read those three together. A mean distance well below ``separation_scale`` means
the two clusterings largely agree; comparable or above means they do not. But check
your own separation scale as well: **a clustering that found no structure puts all
its centroids near the origin, which makes them close to everything and produces a
flatteringly small match distance.** If your spacing is far below the published
one, your clusters are not comparable structure however well they appear to match.

Note that a poor distance does not mean a meaningless pairing. Clustering four
designed grammars (D/E tracts, poly-Q, S-rich, K blocks) and matching them against
the published reference pairs them with ``'D/E-tracts'``, ``'Q-tracts'`` and
``'K blocks'`` respectively -- purely from centroid geometry, with nothing told to
it -- while the distances stay well above the published spacing, because designed
sequences are compositionally extreme and sit far from anything in the human
IDRome. The grammars are recognisably analogous; the coordinates are not the same.
That worked example is in ``examples/nardini_module.ipynb``.

All distances here are squared Euclidean, matching ``assign_gin_clusters`` and
:attr:`~sparrow.nardini.gin.GINReference.separation_scale`.

.. warning::
   You can reproduce the *procedure*; reproducing the published 30 centroids
   exactly is a different proposition.

   * sparrow ships the centroids and the compositional priors, **not the IDRome
     sequences**. You supply those, and "the human IDRome" is itself a definition -
     which disorder predictor, which thresholds, which boundary rules. Different
     IDR boundaries give different vectors and so different clusters.
   * k-means depends on its initialization, so two runs agree only up to a
     permutation of labels at best.
   * Not every hyperparameter is documented upstream.
     ``sparrow/data/nardini/SOURCES.md`` records what could be verified: k=30,
     human IDRs of 100-300 residues, z clipped to +/-3. The restart count,
     iteration limit, any feature weighting and the ``num_scrambles`` behind the
     published vectors are not.
   * The published descriptions are human-authored and cannot be regenerated.

   The realistic goal is "build my own clusters and measure how far they are from
   the published ones", which is what ``compare_gin_references`` reports.

.. note::
   Cost is dominated by the vectors, not the clustering. Roughly 110 ms per IDR at
   1,000 scrambles and 560 ms at 10,000 (200-residue IDR), so a 25,000-IDR
   proteome is about 45 minutes or 4 hours single-core. The k-means itself takes
   seconds. ``build_gin_reference`` warns if you ask for fewer than
   :data:`~sparrow.nardini.clustering.MIN_SEQUENCES_PER_CLUSTER` sequences per
   cluster on average, since k-means returns ``k`` centroids whether or not the
   data supports that many.


.. _nardini-containers:

Working with the results
------------------------

Everything NARDINI returns is a plain NumPy array, or a thin container that keeps
the feature names attached to one. Nothing hands you a DataFrame unless you ask
for one.

The containers - :class:`~sparrow.nardini.table.FeatureVector` for one sequence,
:class:`~sparrow.nardini.table.FeatureTable` for several - do exactly one useful
thing: let you address a feature by name instead of by index.

.. code-block:: python

   v = p.nardini.vector(seed=42)

   v["Frac K+R"]        # one value, by name
   v.values             # the underlying (90,) ndarray
   v.names              # the matching labels
   v.to_dict()          # {feature: value}
   dict(v)              # same - iterating yields (name, value) pairs

   table = result.to_table()

   table["pol-neg"]                 # one feature across all sequences -> ndarray
   table.row("IDR1")                # one sequence -> FeatureVector
   table.row("IDR1")["Frac K+R"]    # one value
   table["IDR1", "Frac K+R"]        # the same thing
   table.values                     # the underlying (n, 90) ndarray
   table.column_names               # the matching labels
   table.to_dict()                  # {row: {feature: value}}

That is deliberately all of it. Filtering, sorting, reshaping, plotting and
export are things NumPy and pandas already do well, so sparrow does not
reimplement them - reach for ``.values`` and carry on.

If you would rather have a DataFrame, ``.to_dataframe()`` is on both containers
and on :class:`~sparrow.nardini.gin.GINAssignment`:

.. code-block:: python

   result.to_table().to_dataframe()      # or result.to_dataframe()
   result.gin_clusters().to_dataframe()

GIN assignments are plain dictionaries by default:

.. code-block:: python

   result.gin_clusters().records()       # [{'name': ..., 'cluster': 7, ...}, ...]
   p.nardini.gin_cluster()               # a single dict

Reading the numbers
-------------------

Both halves are z-scores, but they are answering different questions, and it is
easy to over-read them if you forget which is which.

**Patterning z-scores** compare a sequence to scrambles of *itself*:

* **positive** -- that residue type is more segregated (blockier) than its own
  composition would predict;
* **negative** -- more evenly interspersed than expected;
* **zero** -- either the feature was not evaluated (see ``min_fraction``), or the
  arrangement is exactly as expected.

Because the null is built from the sequence itself, these are directly comparable
between sequences of different length and composition.

**Compositional z-scores** compare a sequence to your background set. Positive
means enriched relative to that reference. These are only as meaningful as the
background -- a z-score of +40 usually means the feature barely varies in the
background, not that the sequence is remarkable.

.. note::
   If a compositional feature has zero variance across the background (every
   reference sequence had the identical value, which happens for rare patch
   features on small reference sets), its z-score is reported as ``0.0`` rather
   than infinity.

The features
------------

Patterning features (36)
^^^^^^^^^^^^^^^^^^^^^^^^

The upper triangle, including the diagonal, of the type-by-type matrix, named
``'<type1>-<type2>'``. For the default 8-type grouping:

.. code-block:: text

   pol   S T N Q C H        polar
   hyd   I L M V            aliphatic / hydrophobic
   pos   R K                positively charged
   neg   E D                negatively charged
   aro   F W Y              aromatic
   ala   A
   pro   P
   gly   G

giving ``pol-pol``, ``pol-hyd``, ... ``gly-gly``. Diagonal entries measure one
type against everything else; off-diagonal entries measure two types against each
other. The names are the same for every backend -- what changes is the statistic
behind them, so always record ``result.backend`` alongside the numbers.

Pass ``typeset='9'`` to break histidine out as its own type, which gives 45
features. This matches ``TYPEALL_9x9`` in the reference ``nardini`` package.

A feature is only evaluated when the type is abundant enough for the statistic to
mean anything: its whole-sequence fraction must *exceed* ``min_fraction``
(default 0.10), and for a pair both fractions must. Features that fail this test
are reported as exactly ``0.0``. This is why a typical vector has many zeros --
that is expected, not an error.

Compositional features (54)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* the 20 amino acid fractions (``'Frac A'`` ... ``'Frac Y'``);
* five grouped fractions: ``'Frac K+R'``, ``'Frac D+E'``, ``'Frac Polar'``,
  ``'Frac Aliphatic'``, ``'Frac Aromatic'``;
* two log-ratios: ``'R/K Ratio'``, ``'E/D Ratio'`` (log10, with a +1
  pseudo-count so they stay finite when a residue is absent);
* seven whole-sequence properties: ``'Frac Chain Expanding'``, ``'FCR'``,
  ``'NCPR'``, ``'Hydrophobicity'``, ``'Disorder Promoting'``, ``'Iso point'``,
  ``'PPII'``;
* 19 single-residue patch fractions (``'A Patch'`` ... ``'Y Patch'``); and
* ``'RG Frac'``, the fraction of the sequence in RG-motif patches.

Almost all of these are available directly on a ``Protein``, without computing a
NARDINI+ vector, and are the *same* numbers -- one implementation, read by both:

.. code-block:: text

   'Frac A' ... 'Frac Y'     p.amino_acid_fractions['A'] ...
   'Frac K+R' / 'Frac D+E'   p.fraction_positive / p.fraction_negative
   'Frac Aliphatic'          p.fraction_aliphatic
   'Frac Aromatic'           p.fraction_aromatic
   'FCR' / 'NCPR'            p.FCR / p.NCPR
   'Hydrophobicity'          p.hydrophobicity
   'R/K Ratio' / 'E/D Ratio' p.RK_ratio / p.ED_ratio
   'Frac Chain Expanding'    p.fraction_chain_expanding
   'Disorder Promoting'      p.fraction_disorder_promoting
   'Iso point'               p.isoelectric_point
   'PPII'                    p.PPII_propensity
   'A Patch' ... 'Y Patch'   p.compute_patch_fraction(residue_selector=...)
   'RG Frac'                 p.compute_rg_patch_fraction()

   'Frac Polar'              p.fraction_polar  -- DIFFERS, see below

The exception is ``'Frac Polar'``: NARDINI+ counts cysteine as polar and
``Protein.fraction_polar`` does not, so the two differ by the cysteine content.
See :ref:`protein-shared-with-nardini`.

What NARDINI+ adds over reading those properties one by one is the *z-scoring* --
each value expressed relative to a reference set of IDRs, which is what makes them
comparable across sequences.

.. note::
   There is deliberately no ``'W Patch'`` feature. NARDINI+ omits tryptophan
   because W patches essentially do not occur in the proteome, so the feature
   carries no signal. The ``'Frac W'`` composition feature *is* present.

Patches are contiguous runs of the target residue tolerating up to 2
interrupting residues, requiring at least 4 target residues; ``'RG Frac'``
requires at least 2 adjacent ``RG`` pairs. These are computed with sparrow's
:func:`~sparrow.sequence_analysis.patching.patch_fraction` primitives.

Relationship to the reference implementations
---------------------------------------------

This module reimplements the analysis performed by the NARDINI+ notebook natively
in sparrow. Differences worth knowing about:

* **No localCIDER dependency.** The four features that the notebook obtains from
  localCIDER (``'Frac Chain Expanding'``, ``'Disorder Promoting'``,
  ``'Iso point'``, ``'PPII'``) are implemented natively and agree with localCIDER
  to within floating-point noise. They are general sequence properties rather than
  NARDINI-specific ones, so they live in
  :mod:`sparrow.sequence_analysis.physical_properties` and are also available
  directly on a ``Protein``:

  .. code-block:: python

     p.fraction_chain_expanding      # 'Frac Chain Expanding'
     p.fraction_disorder_promoting   # 'Disorder Promoting'
     p.isoelectric_point             # 'Iso point'
     p.PPII_propensity               # 'PPII'

  NARDINI reads those same functions, so a ``Protein`` property and its
  corresponding compositional feature are bit-identical by construction -- pinned
  by a test, because the bundled backgrounds and GIN centroids are defined against
  these exact values.
* **Nothing compositional is computed twice.** The same applies to every other
  feature that has a ``Protein`` counterpart: the amino acid fractions, the grouped
  fractions, FCR, NCPR, hydrophobicity, the R/K and E/D ratios and all 20 patch
  features come from :mod:`sparrow.sequence_analysis.physical_properties`,
  :mod:`sparrow.calculate_parameters` and
  :mod:`sparrow.sequence_analysis.patching` -- the same code paths ``Protein``
  uses. The patterning half likewise: the ``'kappa'`` and ``'iwd'`` backends call
  :mod:`sparrow.patterning.vectorized`, and only the NARDINI omega/kappa window
  asymmetry itself is implemented in this module, because nothing else in sparrow
  computes it.

  .. warning::
     One feature deliberately differs from its ``Protein`` counterpart.
     ``'Frac Polar'`` counts cysteine (QNSTGCH); ``Protein.fraction_polar`` does
     not (GHNQST). The published reference data is defined against the former, so
     the two definitions are kept side by side rather than reconciled -- see
     :ref:`protein-shared-with-nardini`. They share an implementation and differ
     only in the residue set.
* **Much faster.** The patterning calculation is vectorized over scrambles, and
  skips the features and residue types the abundance gate has already excluded.
  Measured against a direct transcription of the reference loop, it is about
  **100x** faster at L=1000 and the margin grows with length. A 1000-residue IDR
  at the published 100,000 scrambles takes ~7 s rather than ~13 min, so the
  notebook's limits of 20 sequences and 1000
  residues do not apply.
* **Robust gamma fitting.** The scramble distribution for each feature is
  summarized by a maximum-likelihood 3-parameter gamma fit, as in the reference.
  ``scipy.stats.gamma.fit`` is not fully reliable on these distributions: it
  sometimes raises ``FitError`` (which aborts the reference implementation) and
  sometimes converges to a local optimum whose moments do not describe the data,
  silently producing a z-score wrong by more than an order of magnitude. sparrow
  sanity-checks each fit against the sample moments and falls back to them when
  the fit is bad. Where the reference fit is sound, sparrow's z-scores match it
  bit for bit.
* **Cluster assignments match.** The GIN assignment path reproduces the notebook's
  cluster calls and centroid distances exactly. The one deliberate difference is
  cosmetic: the notebook computes ``separation`` using ``100`` as a sentinel, so a
  true gap above 100 would be reported as exactly 100. sparrow reports the true
  gap; in practice no clipped z-vector reaches a separation that large, so the
  numbers agree.
* **GIN reference data is bundled**, rather than downloaded from Google Sheets at
  runtime as the notebook does, so analyses are reproducible offline and pinned to
  a known version. See ``sparrow/data/nardini/SOURCES.md`` for provenance, and
  ``devtools/scripts/build_nardini_reference_data.py`` to rebuild it -- that
  script verifies the upstream feature ordering before writing anything.

.. _nardini-migration:

Migrating from ``sparrow.sequence_analysis.grammar``
-----------------------------------------------------

sparrow's older grammar module -- reachable as
:meth:`~sparrow.protein.Protein.extract_feature_vector` and
:func:`~sparrow.sequence_analysis.grammar.compute_feature_vector` -- is
**deprecated in favour of this one** and will be removed in a future release.
Both now emit a ``DeprecationWarning``.

``sparrow.nardini`` computes a superset of what that module offered:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - ``sparrow.sequence_analysis.grammar``
     - ``sparrow.nardini``
   * - ``compute_feature_vector(seq, ...)``
     - ``nardini_vectors({"name": seq}, ...)``
   * - ``Protein.extract_feature_vector()``
     - ``nardini_vectors({"name": p.sequence}, ...)``
   * - ``backend='kappa_cython'``
     - ``backend='kappa'`` *(identical values)*
   * - ``backend='iwd_combined'``
     - ``backend='iwd'`` *(identical values)*
   * - *no equivalent*
     - ``backend='nardini'`` -- the published statistic, and the only one that
       supports GIN
   * - 50 composition features
     - the same 50, plus ``Frac Chain Expanding``, ``Disorder Promoting``,
       ``Iso point``, ``PPII``
   * - ``compute_composition_background_stats(...)``
     - ``build_background(...)``
   * - ``build_grammar_background_from_fasta(...)``
     - ``build_background(fasta)`` then ``.save(path)``
   * - one bundled sparrow human background
     - ``load_reference_background(species)`` for nine published IDRomes
   * - *no equivalent*
     - GIN cluster assignment

Two things to know when you migrate:

* **The patterning numbers do not change** if you keep the same statistic --
  ``backend='kappa'`` and ``backend='iwd'`` reproduce the old values exactly. Only
  switching to ``backend='nardini'`` changes them, because that is a different
  statistic.
* **The compositional z-scores will shift.** The two modules bundle *different*
  human IDR datasets (for example mean ``Hydrophobicity`` 3.81 vs. the published
  3.41, ``FCR`` 0.24 vs. 0.27). If you need continuity with previous results,
  rebuild your own background with :func:`~sparrow.nardini.build_background`
  rather than switching to ``load_reference_background('human')``.

The old module's feature vector also merged raw and z-score blocks with
``raw::``/``z::`` name prefixes. ``sparrow.nardini`` keeps them separate --
``result.zscores`` and ``result.raw_patterning`` / ``result.raw_composition``, or
``result.to_table(raw=True)``.

.. _nardini-citations:

Citations
---------

If you use this module, please cite the original work -- not sparrow alone. This
applies with particular force to the bundled reference data (the per-species
compositional backgrounds and the GIN centroids), which are redistributed from
the GIN resource under the MIT licence and were not generated by sparrow:

* **NARDINI** -- M.C. Cohan, M.K. Shinn, J.M. Lalmansingh, R.V. Pappu (2022).
  Uncovering non-random binary patterns within sequences of intrinsically
  disordered proteins. *Journal of Molecular Biology* 434(2), 167373.
* **NARDINI+** -- M.R. King, K.M. Ruff, A.Z. Lin, *et al.*, R.V. Pappu (2024).
  Macromolecular condensation organizes nucleolar sub-phases and sets up a pH
  gradient. *Cell* 187, 1889-1906.e24.
* **GIN resource** -- K.M. Ruff, M.R. King, A.W. Ying, *et al.*, R.V. Pappu
  (2025). Molecular grammars of predicted intrinsically disordered regions that
  span the human proteome. *Cell*.

API Reference
-------------

.. _nardini-protein:

The ``Protein.nardini`` accessor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: sparrow.nardini.protein_interface.ProteinNardini
   :members:

Labelled containers
^^^^^^^^^^^^^^^^^^^

.. autoclass:: sparrow.nardini.table.FeatureVector
   :members:

.. autoclass:: sparrow.nardini.table.FeatureTable
   :members:

Background construction
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sparrow.nardini.background.build_background

.. autofunction:: sparrow.nardini.background.load_reference_background

.. autoclass:: sparrow.nardini.background.NardiniBackground
   :members:

GIN cluster assignment
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sparrow.nardini.gin.assign_gin_clusters

.. autofunction:: sparrow.nardini.gin.gin_cluster_definitions

.. autofunction:: sparrow.nardini.gin.load_gin_reference

.. autoclass:: sparrow.nardini.gin.GINAssignment
   :members:

.. autoclass:: sparrow.nardini.gin.GINReference
   :members:

Vector calculation
^^^^^^^^^^^^^^^^^^

.. autofunction:: sparrow.nardini.vectors.nardini_vectors

.. autoclass:: sparrow.nardini.vectors.NardiniResult
   :members:

Patterning
^^^^^^^^^^

.. automodule:: sparrow.nardini.patterning
   :members: patterning_zscores, patterning_values, resolve_backend,
             encode_types, type_fractions, PATTERNING_BACKENDS

Composition
^^^^^^^^^^^

.. automodule:: sparrow.nardini.composition
   :members: composition_values, composition_matrix, composition_feature_names,
             isoelectric_point, charge_at_pH

Clustering
^^^^^^^^^^

.. automodule:: sparrow.nardini.clustering
   :members: build_gin_reference, compare_gin_references, describe_centroid,
             MIN_SEQUENCES_PER_CLUSTER, DEFAULT_NUM_CLUSTERS

Parallel computation
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: sparrow.nardini.batch.nardini_proteins

Errors and warnings
^^^^^^^^^^^^^^^^^^^

Everything sparrow raises deliberately derives from
:class:`~sparrow.sparrow_exceptions.SparrowException`, so one ``except`` clause
covers it. The advisory warnings below are not errors: the call still returns a
usable result, and they flag a result you probably should not trust. They share a
base class so you can silence them as a group::

    import warnings
    from sparrow.sparrow_exceptions import SparrowWarning

    warnings.filterwarnings("ignore", category=SparrowWarning)

.. autoclass:: sparrow.sparrow_exceptions.SparrowWarning

.. autoclass:: sparrow.sparrow_exceptions.BackgroundQualityWarning

.. autoclass:: sparrow.sparrow_exceptions.ClusteringQualityWarning

Residue groupings
^^^^^^^^^^^^^^^^^

.. automodule:: sparrow.nardini.groups
   :members: get_typeset, type_names, patterning_feature_names
