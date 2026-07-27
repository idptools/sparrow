# Architecture of `sparrow.nardini`

This describes how the files in this module fit together, and how they relate to the `Protein` object. It is for people changing the code; for how to *use* it, see the module docstrings or the [NARDINI guide](https://idptools-sparrow.readthedocs.io/en/latest/api_guides/nardini.html).

## The shape of the thing

A NARDINI+ vector has two halves that are computed in completely different ways, and almost every design decision here follows from that.

The **patterning** half (36 features by default) asks how non-randomly the residues are *arranged*. Each feature is z-scored against scrambles of the same sequence, so it needs no external data — just the sequence and a random number generator. It is also where essentially all the compute time goes, because it means scoring thousands of shuffled copies.

The **compositional** half (54 features) asks what the sequence is *made of*. Each feature is z-scored against a reference set of IDRs — a *background* — so it needs external data but almost no compute.

Everything else follows: the background machinery exists only for the second half, the parallelism exists only for the first, and the two halves are stitched together at the very end.

## Dependency layers

Arrows point from a module to the modules it imports. Nothing points upwards, so the layering is acyclic.

```
                        Protein  (sparrow/protein.py)
                           |
                           v
                  protein_interface.py  ......  batch.py
                           \                      /
                            v                    v
                              -->  vectors.py  <--
                                  /     |     \
                    ______________/     |      \_______________
                   /                    |                      \
            patterning.py         composition.py           background.py
                   |                /        \                  |
              groups.py            /          \                 |
                                  v            v                |
              sparrow.patterning.vectorized    sparrow.sequence_analysis.*
                                                                |
                                               (background also uses composition)

              gin.py  <--  clustering.py          table.py
                 ^                                   ^
                 |                                   |
            (consumed by vectors.py, protein_interface.py, batch.py)
```

### The bottom layer: what actually computes things

**`groups.py`** defines the residue typesets — the 8-type default and the 9-type variant — and derives the patterning feature names from them. Both are exact partitions of the 20 standard amino acids, asserted at import. That property is load-bearing: it is what lets `patterning.py` reduce a sequence to a compact integer label array instead of working with strings.

**`patterning.py`** computes the patterning half. It implements the NARDINI omega/kappa window asymmetry directly, because nothing else in sparrow computes that statistic. The other two backends do *not* have their own implementations here — `'kappa'` and `'iwd'` call `sparrow.patterning.vectorized`, the same code behind `Protein.kappa` and the IWD methods. This module also owns the scramble machinery and the gamma fitting that turns raw asymmetries into z-scores.

**`composition.py`** assembles the compositional half but implements almost none of it. The amino acid fractions come from `sparrow.calculate_parameters`, the grouped fractions and whole-sequence properties from `sparrow.sequence_analysis.physical_properties`, and the patch features from `sparrow.sequence_analysis.patching` — in every case the same functions behind the corresponding `Protein` property. It is deliberately a thin assembler: a feature and its `Protein` counterpart are then the same number by construction rather than by coincidence.

**`background.py`** builds and applies the compositional reference statistics. It calls `composition.py` over a set of IDRs and accumulates per-feature means and standard deviations with Welford's algorithm, so memory does not grow with the size of the IDRome. It also loads the nine bundled per-species priors from `sparrow/data/nardini/`.

### The middle layer: putting a vector together

**`vectors.py`** is the entry point and the only place the two halves meet. `nardini_vectors()` calls `patterning_zscores()` per sequence, computes the compositional block, z-scores it against the background, and returns a `NardiniResult`.

Two things it owns are relied upon elsewhere. It derives each sequence's random seed from the master seed *and the sequence itself*, never from position in the batch — which is what makes results independent of batching, and therefore what makes `batch.py` safe. And it records the patterning backend on the result, which is what stops incomparable vectors being mixed downstream.

**`table.py`** holds `FeatureVector` and `FeatureTable`, the labelled containers nearly everything returns. They are deliberately minimal: they keep feature names attached to a NumPy array and otherwise get out of the way. They import nothing else from the module, so they sit outside the dependency chain entirely.

### The top layer: the two entry points

**`protein_interface.py`** is what `Protein.nardini` resolves to. It is an adapter: it resolves the `background` argument, calls `nardini_vectors()` on the single sequence, and slices out whichever piece was asked for. It also owns the per-`Protein` memoization, keyed on the *contents* of the background rather than its identity.

**`batch.py`** is the other entry point, for many sequences at once. It chunks the input and runs the ordinary `nardini_vectors()` in worker processes, then attaches each result to a `Protein`. It reimplements nothing, which is the point: correctness rests entirely on the seeding property `vectors.py` guarantees.

### Off to the side: clusters

**`gin.py`** assigns vectors to cluster centroids and holds `GINReference` (the centroids) and `GINAssignment` (the answer, with its typicality diagnostics). It consumes vectors; it does not produce them.

**`clustering.py`** goes the other way: it takes vectors and *derives* centroids, producing a `GINReference` that `gin.py` can then assign against. It is the only module that depends on `scipy.cluster` and `scipy.optimize`.

These two are a matched pair — one builds the object, the other consumes it — and both are optional. You can use the rest of the module without touching either.

## How this connects to `Protein`

Two directions, and it is worth keeping them straight.

**`Protein` reaching into nardini.** `Protein.nardini` is a lazily-created, cached `ProteinNardini`. `sparrow/protein.py` imports nothing from this module at module load — the import happens inside the property — so nardini's dependencies are not paid for by every `Protein`.

**nardini reaching into `Protein`.** `composition.py` calls the same functions that back `Protein.FCR`, `Protein.hydrophobicity`, `Protein.isoelectric_point` and the rest, and `patterning.py` calls the same vectorized code behind `Protein.kappa`. Neither constructs a `Protein` to do it — they call the underlying stateless functions in `sparrow.sequence_analysis.physical_properties`, `sparrow.calculate_parameters` and `sparrow.patterning.vectorized`. That keeps the batch path free of per-sequence object overhead while still guaranteeing there is one implementation of each quantity.

The practical consequence is that `Protein(seq).FCR` and `composition_values(seq)['FCR']` are bit-identical, and a test asserts this for every feature with a counterpart. There is exactly one deliberate exception: `'Frac Polar'` counts cysteine and `Protein.fraction_polar` does not.

`batch.py` closes the loop by writing results *into* `Protein` objects, using two private hooks on `ProteinNardini` (`_cache_key` and `_store`). Those exist so the batch path and the accessor cannot disagree about what a cache entry is keyed on.

## Invariants worth not breaking

**The compositional feature values are frozen.** The bundled per-species backgrounds and the GIN centroids in `sparrow/data/nardini/` were computed against these exact numbers. Changing how a feature is summed — even in a way that looks algebraically identical — shifts every compositional z-score and every centroid distance. `fraction_in_set()` in `physical_properties` documents one place where the obvious simplification would do exactly that. Verify changes against a before/after snapshot with `array_equal`, not `allclose`.

**Seeds derive from sequence content, not batch position.** Break this and `batch.py` silently returns different numbers depending on how work was divided, and `Protein.nardini` stops agreeing with `nardini_vectors`.

**Backends must not be mixed.** A vector and the centroids it is assigned against must come from the same patterning statistic. `gin.py` enforces this by comparing the backend recorded on the result against the one recorded on the reference.

**Patterning is background-independent.** The patterning half never touches the background. `ProteinNardini.patterning()` relies on this to reuse a cached full result rather than re-running the scrambling.

## Where the tests live

| Test file | Covers |
|---|---|
| `test_nardini.py` | patterning, composition, backgrounds, and a transcribed reference implementation used as an oracle |
| `test_nardini_gin.py` | cluster assignment and the typicality diagnostics |
| `test_nardini_table.py` | the containers and the `Protein.nardini` accessor, including its cache |
| `test_nardini_batch.py` | parallel computation, and that it changes no numbers |
| `test_nardini_clustering.py` | deriving clusters, and comparing them to the published ones |
| `test_regressions.py` | that `Protein` properties and NARDINI features agree bit for bit |

The oracle in `test_nardini.py` is worth knowing about: it is a deliberately naive transcription of the reference notebook's algorithm, kept so the vectorized production code can be checked against it directly rather than against stored numbers.
