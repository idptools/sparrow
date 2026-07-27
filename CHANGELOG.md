# Changelog

All notable changes to SPARROW are documented here.

## Unreleased

### Added — `sparrow.patterning.vectorized`

Pure-NumPy reimplementations of the compiled patterning routines, as a step towards dropping the Cython extensions. Every public Cython entry point except `patterning_percentile` now has a signature-compatible twin:

| Cython | NumPy |
|---|---|
| `kappa.calculate_sigma` / `calculate_delta` / `kappa_x` | same names in `vectorized` |
| `iwd.calculate_average_inverse_distance_from_sequence` | same name |
| `iwd.calculate_average_inverse_distance_charge` | same name |
| `iwd.calculate_average_bivariate_inverse_distance_charge` | same name |
| `scd.compute_scd_x` / `compute_shd` | same names |
| `patterning.patterning_percentile` | same name — but see the seed note below |

Plus batch entry points with no Cython equivalent — `kappa_x_batch`, `kappa_from_ternary_batch`, `iwd_batch`, `ternarize_winged_batch` — which score a whole stack of sequences per call.

**Faster, mostly through better algorithms rather than vectorization alone.** IWD's pairwise `1/|i-j|` sum is the autocorrelation of the occupancy mask (one FFT instead of an explicit pairwise loop); SCD reduces to a `sqrt(d)`-weighted autocorrelation of the charge sequence, `O(n log n)` instead of `O(n²)`; SHD has a sum rather than a product in its numerator, so it collapses to prefix sums, `O(n)` instead of `O(n²)`. Measured against the extensions: `compute_shd` at L=5000 is **1779x** faster (393 ms → 0.22 ms), `compute_scd_x` 53x, single-call `kappa_x` 3x, and batched scramble scoring 18–28x at L=1000.

**Accuracy.** IWD, SCD and SHD are double precision on both sides and agree to ~1e-13. The kappa routines carry their intermediates in C `float` in the Cython code and `float64` here, so they agree to ~1e-6 — and the NumPy value is the more accurate one. `test_patterning_vectorized.py` (36 tests) pins this, including a test that replays the Cython arithmetic in single precision to demonstrate the residual gap is the compiled code's precision, not an error in the port.

**`patterning_percentile` and its seed.** The compiled version draws shuffles from C's `rand()`, whose algorithm the C standard leaves implementation-defined — so its seed-to-result mapping was never portable across platforms or libc versions to begin with. The NumPy version uses NumPy's generator, so the same `seed` gives a different, statistically equivalent set of shuffles. The *statistic* is unchanged: on identical arrangements the two agree to 1.6e-6 (the compiled code's precision), and percentiles from the two agree to within Monte Carlo sampling error. Verified by a paired test on identical arrangements plus a KS comparison of the two null distributions — the latter only agrees once both are snapped to the true discrete lattice, because the compiled version accumulates in float32 and splits genuinely-tied values into slightly different floats. It is also ~5x faster.

### Changed — `Protein` and `scaling_parameters` now use the NumPy patterning metrics

`sparrow.protein.Protein` and `sparrow.polymer.scaling_parameters` now call `sparrow.patterning.vectorized` instead of the compiled extensions. Measured effect on user-facing values:

| value | shift | speed (L=3000) |
|---|---|---|
| `Protein.kappa`, `compute_kappa_x` | ≤ 6e-06 | 3.33 ms → 0.23 ms (15x) |
| `Protein.SCD` | ≤ 5e-14 | 2.82 ms → 0.08 ms (34x) |
| `Protein.SHD` | ≤ 2e-14 | 138.85 ms → 0.14 ms (**967x**) |
| `compute_iwd` and the charge-weighted IWD variants | ≤ 7e-16 | — |

The kappa shift is single-precision arithmetic being replaced by double, so the new values are the more accurate ones; expect agreement with previously-stored kappa values to about six significant figures rather than exactly. All existing stored-reference tests (`test_kappa.py`, `test_iwd.py`, `test_scd.py`, `test_scd_shd_formula.py`, `test_protein_api.py`, `test_sparrow_vs_localcider.py`, `test_polymeric.py`) pass unchanged at their existing tolerances.

### Removed — the Cython extensions; sparrow is now pure Python

The four `.pyx` extension modules are gone. `sparrow.patterning.kappa`, `iwd`, `scd` and `patterning` are now thin pure-Python modules re-exporting from `sparrow.patterning.vectorized`, so **every existing import keeps working unchanged**.

Consequences:

* **No compiler needed.** `pyproject.toml` no longer requires `cython` (at build time or runtime) or build-time `numpy`, and `setup.py` is reduced to a stub. The build now produces a `py3-none-any` wheel instead of a platform-specific one.
* **Upgrading an existing checkout requires deleting stale artifacts**: `rm -f sparrow/patterning/*.so sparrow/patterning/*.c`. Python resolves extension modules before source modules, so a leftover `kappa.cpython-*.so` would shadow the new `kappa.py` and silently keep running the old compiled code. Noted in the README and install guide.
* Editor autosave files (`#file#`, `.#file`, `*~`) are now excluded from the distributions via `[tool.setuptools.exclude-package-data]` — MANIFEST.in cannot express this, since `#` starts a comment there.

**Validation was preserved rather than dropped.** The cross-checks in `test_patterning_vectorized.py` compared against the extensions, which would have become self-comparisons once the shims were in place. Before deleting, the original `.pyx` sources were rebuilt in isolation and used to capture 400 randomly-generated golden cases — every metric, both kappa flatten modes, the charge-weighted IWD variants, custom SCD groups, and the deterministic half of `patterning_percentile` — into `sparrow/tests/test_data/cython_patterning_golden_v1.npz` (86 KB). The tests now assert against that fixture, so the agreement with the compiled implementation stays pinned permanently and independently of whether it exists.

### Added — `Protein.nardini`, and NARDINI results are NumPy by default

**`Protein.nardini`** is a first-class accessor for single-sequence NARDINI work, cached and memoized per option set:

```python
p.nardini.vector(seed=42)        # 90-feature NARDINI+ vector
p.nardini.patterning(seed=42)    # just the 36 patterning z-scores
p.nardini.composition()          # just the 54 compositional z-scores
p.nardini.gin_cluster(seed=42)   # nearest published human GIN grammar
print(p.nardini.gin_report(seed=42))
```

Compositional z-scores need a reference IDR set and there is nothing to build one from for a single protein, so these default to the bundled published human IDRome background — which is also the space the GIN centroids live in. Override with a species key, an `.npz` path, or a `NardiniBackground`. The patterning half never uses it.

**Nothing returns a DataFrame by default any more.** Results are plain NumPy arrays, optionally wrapped in one of two deliberately thin containers — `FeatureVector` (one sequence) and `FeatureTable` (several) — whose only job is to let you address a feature by name rather than by index: `v["Frac K+R"]`, `table["pol-neg"]`, `table.row("IDR1")["Frac K+R"]`, plus `.values`, `.names`/`.column_names`, `.to_numpy()` and `.to_dict()`. Filtering, sorting, reshaping and export are left to NumPy and pandas rather than reimplemented.

* `NardiniResult.to_table()` / `.vector(name)` replace `.to_dataframe()` as the default view.
* `GINAssignment.records()` returns plain dicts and replaces `.to_dataframe()` as the default.
* `gin_cluster_definitions()` now returns a list of dicts; pass `as_dataframe=True` for the old behaviour.

`.to_dataframe()` still exists everywhere for anyone who would rather have pandas — it is opt-in rather than gone.

**Fixed: pandas was an undeclared dependency.** `sparrow.sequence_analysis.elm` imported it at module level, so `import sparrow` required pandas despite it never appearing in `pyproject.toml` — sparrow only worked because pandas happened to be installed. It is now declared in `dependencies`, and imported lazily at the few points that actually build a DataFrame rather than at import time. The ELM class table is parsed with the standard library instead (new `read_elm_table()`, which returns a header and a list of rows); `generate_elm_df()` is unchanged for callers who want the DataFrame.

### Added — GIN typicality diagnostics

Nearest-centroid assignment never abstains, so every sequence gets a label including ones that look nothing like a human IDR. `GINAssignment` now reports:

* **`composition_rms_z`** — the direct answer to "is this anything like a human IDR?". The 54 compositional features are standardized against the reference IDRome, so a member of that population has z ~ N(0,1) per feature and an RMS near 1. Measured: real and simulated IDRs 0.6–0.9, uniformly random sequence 1.6, homopolymers and other extremes 3–13. Computed before clipping so it is not capped.
* **`n_saturated`** — how many of the 90 features were pinned at the ±3 clip. A clipped feature contributes identically to every centroid distance, so it cannot discriminate between clusters; all 30 centroids lie within −2.32 to 2.99.
* **`is_typical`** — a coarse triage screen combining the two, plus a prose explanation from `Protein.nardini.gin_report()`.

Note that *distance to the nearest centroid is not usable as an absolute measure of typicality*, and an earlier draft of this that used it was wrong: in 90 dimensions even a perfectly ordinary z-vector sits about five times further from its nearest centroid than the centroids sit from each other, because centroids are cluster means rather than members. Distances remain meaningful compared between sequences, and `GINReference.separation_scale` documents the pitfall.

### Added — `sparrow.nardini`

A new module implementing the [NARDINI](https://www.sciencedirect.com/science/article/pii/S0022283621006100) and NARDINI+ analyses natively in sparrow, reproducing the analysis performed by the NARDINI+ Colab notebook. Three entry points:

* `build_background(source, ...)` estimates the mean and standard deviation of the 54 NARDINI+ compositional features over a reference set of IDRs, supplied as a FASTA file, a dict, or any iterable of sequences / `Protein` objects. Statistics accumulate online (Welford), so memory does not scale with the size of the IDRome. Backgrounds carry provenance `metadata` and save/load as `.npz` (`NardiniBackground.save` / `.load`). Alternatively, `load_reference_background(species)` returns a published per-species IDRome background bundled with sparrow (`human`, `mouse`, `rat`, `xenopus`, `fly`, `zebrafish`, `yeast`, `worm`, `arabidopsis`).
* `nardini_vectors(sequences, background=..., ...)` returns a `NardiniResult` holding the 90-feature NARDINI+ z-score vector per sequence: 36 patterning features z-scored against per-sequence scrambles, plus 54 compositional features z-scored against the background. `background` is optional — omit it for the 36 patterning features alone, which need no reference set. `NardiniResult` exposes `.zscores`, `.to_dataframe(raw=False)` and `.top_features(name, n)`.
* `assign_gin_clusters(result)` — also reachable as `result.gin_clusters()` — assigns each sequence to one of the 30 published human GIN clusters of shared molecular grammar (`'D/E-tracts'`, `'S patches'`, `'Well-mixed P and G'`, ...) by nearest centroid in ±3-clipped z-score space. Returns a `GINAssignment` with the cluster, its published description, the centroid distance, and the `separation` from the runner-up cluster as a confidence measure, plus `.ranked_clusters(name, n)` for the alternatives when that separation is small. `gin_cluster_definitions()` returns the 30 published descriptions and IDR counts.

Supporting public API: `patterning_zscores()`, `patterning_values()`, `composition_values()`, `composition_matrix()`, `load_gin_reference()`, `resolve_backend()`, the `TYPES_8` / `TYPES_9` residue groupings, and the feature-name helpers.

**Selectable patterning statistics.** The 36 patterning features can be computed three ways via `backend`:

* `'nardini'` (default) — the published NARDINI omega/kappa window asymmetry. Required for GIN cluster assignment.
* `'kappa'` — sparrow's generalized kappa (`kappa_x`), normalized against the maximally-segregated sequence and bounded 0–1.
* `'iwd'` — sparrow's inverse-weighted distance, a clustering metric that ignores `blob_size`.

The latter two are ports of the `kappa_cython` and `iwd_combined` backends from `sparrow.sequence_analysis.grammar` and reproduce its values **exactly** (verified bit-for-bit across a range of sequences); the legacy names are accepted as aliases. Only the patterning block is affected — the compositional block is identical across backends. The choice is recorded as `result.backend`, and `assign_gin_clusters()` refuses anything but `'nardini'`, since the published centroids are defined in that statistic's space and another backend would yield plausible-looking but meaningless cluster labels.

All three backends are vectorized over scrambles and cost about the same: at L=1000 and 2,000 scrambles, 0.49 s / 0.43 s / 0.37 s for nardini / kappa / iwd. The `kappa` and `iwd` backends originally delegated to the Cython extensions and were 9–22x slower (11.05 s and 4.77 s respectively at L=1000); they now use `sparrow.patterning.vectorized`. As a consequence their values agree with `sparrow.sequence_analysis.grammar` to ~1e-6 rather than bit-for-bit, because grammar goes through the single-precision Cython path while these use double — the NumPy value being the more accurate of the two. The `test_ported_backends_match_grammar` assertion was relaxed from `array_equal` to a 1e-5 relative tolerance to match.

### Deprecated — `sparrow.sequence_analysis.grammar`

Superseded by `sparrow.nardini`, which computes a strict superset of its features. `grammar.compute_feature_vector()`, `grammar.compute_composition_background_stats()`, `Protein.extract_feature_vector()` and `sparrow.build_grammar_background_from_fasta()` now emit a `DeprecationWarning` and will be removed in a future release. They continue to work unchanged in the meantime.

Migration is documented in the NARDINI guide. Two notes: the `'kappa'` / `'iwd'` backends reproduce the old patterning values exactly so those numbers do not move, but the two modules bundle *different* human IDR backgrounds (mean `Hydrophobicity` 3.81 vs. the published 3.41, `FCR` 0.24 vs. 0.27), so compositional z-scores will shift unless you rebuild your own background with `build_background()`.

**Bundled reference data.** `sparrow/data/nardini/` now holds the published GIN centroids (30 × 90), the human cluster definitions, and the nine per-species compositional backgrounds. These are redistributed from the [GIN resource](https://github.com/kierstenruff/RUFF_KING_Grammars_of_IDRs_using_NARDINI-) under its MIT licence and **must be cited** (Ruff et al. 2025) — see `sparrow/data/nardini/SOURCES.md` for full provenance. Bundling rather than downloading at runtime (as the notebook does) means analyses are reproducible offline and pinned to a known version. `devtools/scripts/build_nardini_reference_data.py` rebuilds them from the canonical sources, and verifies the upstream feature ordering still matches sparrow's own definitions before writing anything.

Notes on the implementation:

* **No new dependencies.** The four features the reference notebook takes from localCIDER (`Frac Chain Expanding`, `Disorder Promoting`, `Iso point`, `PPII`) are implemented directly and agree with localCIDER to floating-point noise. The Hilser PPII scale, EMBOSS side-chain pKa values, and the disorder-promoting and chain-expanding residue sets were added to `sparrow.data.amino_acids`.
* **Verified against the reference.** The patterning statistics are bit-identical to a direct transcription of the notebook algorithm across hand-picked and randomly-generated sequences; that transcription is kept in `sparrow/tests/test_nardini.py` as a permanent oracle.
* **20-40x faster patterning.** The asymmetry calculation is vectorized over scrambles by reducing each sequence to an integer type-label array (the groupings are exact partitions of the 20 amino acids) and deriving all window counts from one cumulative sum. A 1000-residue IDR at the published 100,000 scrambles takes ~15 s rather than ~9 min, so the notebook's limits of 20 sequences and 1000 residues are unnecessary.
* **More robust gamma fitting.** Each scramble distribution is summarized with a maximum-likelihood 3-parameter gamma fit, as in the reference. `scipy.stats.gamma.fit` is unreliable on these distributions: it raises `FitError` on some inputs (which aborts the reference implementation outright) and on others converges to a local optimum whose moments do not describe the data — a fitted mean 6x and variance 170x the sample values was observed, silently collapsing one z-score from ~12.6 to ~0.4. Fits are now sanity-checked against the sample moments and rejected when they disagree, which makes the z-scores stable across seeds and scramble counts. Where the reference fit is sound, results match it bit for bit.
* **GIN assignments match the reference exactly**, including all 30 centroid distances per sequence. The only deliberate difference is cosmetic: the notebook computes the runner-up gap using `100` as a sentinel value, so a true gap above 100 would be reported as exactly 100; sparrow reports the true gap. No clipped z-vector reaches a separation that large, so in practice the numbers are identical.
* Sanity check on the assignment: designed test sequences land in the semantically matching published clusters — a D/E tract in `'D/E-tracts'`, a poly-Q in `'Q-tracts'`, an S-rich sequence in `'S patches'`, a K-block sequence in `'K blocks'`.

### Added — example notebooks for NARDINI

Two executed notebooks in `examples/`, both worked through on a real human IDR (BARD1, whose disordered regions are located with sparrow's own disorder predictor rather than hard-coded):

* `nardini_from_protein_objects.ipynb` — the single-sequence interface via `Protein.nardini`: reading a `FeatureVector`, the residue typesets, why some patterning features are exactly zero, the compositional half and the choice of background, GIN assignment, and when a GIN label should not be believed.
* `nardini_module.ipynb` — the batch interface: building and saving your own compositional background, `nardini_vectors()` over a set, the `NardiniResult` and `FeatureTable` surfaces, GIN assignment across a dataset with the typicality diagnostics, and reading from FASTA.

Writing them surfaced three bugs, below. All three were cases where the documented behaviour was right and the implementation was not.

### Changed — nardini no longer computes anything twice

Following on from the four properties below, every remaining compositional quantity NARDINI+ computes now comes from the same code `Protein` uses. Audited feature by feature rather than by inspection: of the 50 features with a `Protein` counterpart, **49 are now bit-identical** and the 50th differs deliberately (below).

* The 20 amino acid fractions come from `calculate_parameters.calculate_aa_fractions`; the grouped fractions, FCR, NCPR, hydrophobicity and the R/K and E/D ratios from `sequence_analysis.physical_properties`; the 20 patch features already came from `sequence_analysis.patching`. `Protein.FCR`, `fraction_positive/negative/polar/aromatic/aliphatic` and `NCPR` now delegate to the same functions rather than re-summing inline.
* New `p.RK_ratio` and `p.ED_ratio` (`log10` with a +1 pseudo-count), which previously existed only as NARDINI features. They split residue pairs that share a formal charge but behave differently.
* On the patterning side there was never any duplication to remove: the `'kappa'` and `'iwd'` backends already call `sparrow.patterning.vectorized`, and the NARDINI omega/kappa window asymmetry is implemented once, in `sparrow.nardini.patterning`, because nothing else in sparrow computes it.

**`Protein` values shift by at most 5.6e-17**, and nardini's do not move at all. That asymmetry is deliberate: the bundled backgrounds and GIN centroids are defined against the NARDINI values, so those are the ones that had to be preserved bit for bit — verified against a 305-sequence snapshot taken before the refactor, compared with `array_equal`. The shift affects `FCR`, `fraction_aliphatic`, `fraction_aromatic` and `fraction_polar` on roughly a fifth of sequences.

The cause is worth recording, because it looks like a bug and is not: CPython 3.12 gave `sum()` compensated (Neumaier) summation for floats, while `Protein` accumulated in a manual `for` loop. Same residue sets, same order, same inputs — the loop was simply the less accurate of the two. Delegating adopts the better value.

**One feature deliberately still differs.** NARDINI+ counts cysteine as polar (`'Frac Polar'` = QNSTGCH); `Protein.fraction_polar` does not (GHNQST). The gap equals the cysteine fraction in 400/400 test sequences. Both conventions are in use, and the published reference data is defined against the NARDINI one, so the definitions are kept side by side and documented in both guides rather than silently reconciled — they share an implementation and differ only in the residue set. A test asserts the gap is exactly the cysteine content, so if either drifts for another reason it fails.

Two further tests guard the arrangement: one checks every shared feature against its `Protein` property, the other asserts `composition_values()` contains no residue-set arithmetic of its own, since the realistic regression is someone re-inlining a formula and the two quietly diverging.

### Changed — four compositional features are now `Protein` properties

`'Frac Chain Expanding'`, `'Disorder Promoting'`, `'Iso point'` and `'PPII'` were implemented inside `sparrow.nardini.composition`, which made them reachable only through a NARDINI+ vector even though they are ordinary sequence properties. There is now one implementation, in `sparrow.sequence_analysis.physical_properties`, read by both:

```python
p.fraction_chain_expanding      # 'Frac Chain Expanding'
p.fraction_disorder_promoting   # 'Disorder Promoting'
p.isoelectric_point             # 'Iso point'
p.PPII_propensity               # 'PPII'
p.charge_at_pH(5.5)             # the charge isoelectric_point bisects to zero
```

`charge_at_pH()` comes along for the ride, since it was the machinery behind the pI and is generally useful — note it differs from `NCPR`, which counts formal charges and assumes full ionization, whereas this applies Henderson-Hasselbalch with the EMBOSS side-chain pKa values and so also sees H, Y and C.

**The values are bit-identical, not merely close**, which was the constraint that shaped the refactor: the bundled per-species backgrounds and GIN centroids are defined against these exact numbers, so a shift in the last decimal place would silently move every compositional z-score and every centroid distance. Verified across 305 sequences before and after — the full 54-feature matrix compares equal under `array_equal`, not just `allclose`. That required care in one place: the fraction features sum per-residue fractions (`a/n + b/n`) rather than dividing a summed count (`(a+b)/n`), and those are not always the same float, so `fraction_in_set()` preserves the original summation order and says why.

`charge_at_pH` and `isoelectric_point` remain importable from `sparrow.nardini.composition` as re-exports, so nothing that used them breaks. New tests pin the Protein-property/NARDINI-feature equality per feature, and assert that nardini still delegates rather than carrying its own copy.

### Documentation

The NARDINI guide, the examples page, the README and the notebooks now cover everything above:

* The compositional feature list in the NARDINI guide now maps each feature onto its `Protein` equivalent, and `examples/nardini_from_protein_objects.ipynb` demonstrates the equality directly — including the deliberate `'Frac Polar'` / cysteine exception, shown numerically. Both make the point that what NARDINI+ adds over reading those properties one by one is the z-scoring.
* `sequence_analysis/physical_properties.py` gained a real module docstring. It had only `##` comments, which autodoc cannot render, so the rendered page jumped straight from the heading to the first function — unhelpful for what is now the shared home for these calculations.
* **The AAindex reference table now cites its sources.** The auto-generated table of all 569 shipped indices (regenerated from the bundled database on every build, so it cannot drift) used only 3 of the 7 metadata fields available. It gained a **Source** column carrying the journal citation and a PubMed link — 452 of the 569 resolve to a PMID; one entry records a literal `PMID:` with no number and falls back to the journal rather than emitting a broken link. The introduction now explains each column, notes that several near-identical descriptions cover scales derived quite differently (so the source is worth checking), and shows how to filter the list programmatically instead of scrolling 569 rows.
* A programmatic audit of `Protein`'s public surface against the guide: all 45 members are documented, as are all 16 `physical_properties` functions.
* **New `sparrow/nardini/architecture.md`** describing how the module's files interoperate and how they connect to `Protein` — the dependency layering, which module owns which half of a vector, why `composition.py` is an assembler rather than an implementation, and the four invariants that are not obvious from any single file (frozen compositional values, content-derived seeds, backends that must not be mixed, background-independent patterning). Every structural claim in it was verified against the actual imports rather than written from memory. Shipped in the wheel so the pointer from the package docstring resolves for an installed copy.
* **Module docstrings brought back in line with the code.** Several had drifted: the package docstring still showed `to_dataframe()` as the default view and did not mention parallel computation or clustering; `gin.py` said it "does not re-cluster anything" without pointing at the module that does; `composition.py` described features it no longer implements itself; `background.py` did not mention the quality warning it now emits; `protein_interface.py` omitted `gin_report()` and the memoization; `vectors.py` documented neither the seeding guarantee nor the backend recording, both of which other modules depend on.

* `docs/examples.rst` gained a worked proteome walkthrough — FASTA → parallel vectors → cached reads → GIN — plus pointers to the two executed notebooks, which it had never mentioned.
* `examples/nardini_module.ipynb` gained sections on parallel computation and on deriving your own grammars. The clustering demo plants four grammars (D/E tracts, poly-Q, S-rich, K blocks) and recovers all four 10/10; matching them against the published reference pairs them with `'D/E-tracts'`, `'Q-tracts'` and `'K blocks'` from centroid geometry alone. The distances are nonetheless far above the published centroid spacing, because designed sequences sit far from anything in the human IDRome — the guide and the notebook both now make that distinction explicit, since "recognisably analogous grammar" and "same coordinates" are not the same claim.
* The warning hierarchy (`SparrowWarning`, `BackgroundQualityWarning`, `ClusteringQualityWarning`) is documented as a reference section with the filtering recipe; `ClusteringQualityWarning` had not been named in the docs at all.
* The feature summaries in `README.md` and `docs/index.rst` described only the original NARDINI surface, with no mention of GIN clustering or parallel computation.

### Documentation — working with the contents of your own clusters

The "building your own clusters" section now covers what you actually do once you have a reference: read cluster sizes off `reference.num_idrs` (or `definitions()`), group an assignment's records by cluster to get the members, iterate clusters and then the IDRs within each, filter on the diagnostics while iterating, and get back to the sequences or `Protein` objects behind the names.

It also flags a distinction that is easy to trip over: `reference.num_idrs` records how many sequences *formed* each centroid, whereas assigning a set gives that set's distribution across the same centroids. They coincide only when you assign back exactly what you clustered — verified with an example where clustering 8 per cluster and then assigning a different 3 per cluster gives, correctly, different numbers.

### Fixed — `nardini_vectors()` silently swallowed unknown keyword arguments

It forwards `**read_fasta_kwargs` to `protfasta.read_fasta`, but only when `sequences` is a filename; with any other input those keywords vanished without a word. That is how `nardini_vectors(..., num_workers=16)` — the argument is `n_processes` — came to run at exactly the speed of `num_workers=1`, with nothing to indicate the request had been ignored. Measured: `num_workers` 1 vs 16 gave 7.01 s vs 7.02 s, where `n_processes` 1 vs 8 gives 7.11 s vs 1.11 s.

Unknown keywords now raise, with a suggestion where there is an obvious one (`n_procs` → "Did you mean 'n_processes'?"), an explicit pointer to `n_processes` for anything worker- or thread-shaped, and a note that FASTA options only apply when reading from a file. Genuine `protfasta` options still work when `sequences` is a filename. `nardini_proteins()` was never affected — it has no catch-all.

### Added — `nardini_vectors(n_processes=...)`

`nardini_vectors()` can now compute across processes, not just `nardini_proteins()`. Default is 1, so nothing spawns unless asked — which matters because `Protein.nardini` calls this for a single sequence. On 64 IDRs at 10,000 scrambles: 22.1 s serial, 11.5 s on two processes, 5.9 s on four, 3.3 s on eight (6.8x), and **bit-identical to serial at every process count and chunk size**, because a sequence's seed comes from its own text rather than its position in the batch.

The pool handling, chunking and per-sequence error isolation were extracted from `nardini_proteins()` into a shared `score_in_parallel()`, so there is one implementation rather than two that can drift. Results are reassembled in the caller's order, not the order workers finished in; parallel mode requires unique sequence names, since results are matched back up by name, and says so if they collide.

Use `nardini_proteins()` when you want `Protein` objects back or need `on_error='skip'`; use `nardini_vectors(n_processes=...)` when you just want the matrix.

### Documentation — `'nardini'` and `'kappa'` measure the same thing

Clarified in the guide, the `patterning` module docstring and the README, because the naming invites exactly the wrong conclusion. `'nardini'` z-scores **delta**, the unnormalized mean squared deviation of the per-window asymmetry from the whole-sequence asymmetry. `'kappa'` z-scores delta divided by `delta_max`, the delta of the most segregated arrangement of the same composition.

**Those are the same quantity once z-scored.** `delta_max` depends only on composition, scrambling preserves composition exactly, and so `delta_max` is the same constant for a sequence and for every member of its null distribution — verified: it takes exactly one distinct value across a sequence and 200 scrambles of it. A constant cancels out of a z-score.

They are close but not bit-identical, for reasons unrelated to the normalization: `kappa_x` scores a *winged* (end-padded) array, changing which windows exist, and caps values at 1. Measured across five sequence types at 4,000 scrambles, per-feature z-scores correlate at 0.97–1.00, differ by at most ~0.2, and agree in sign for 31 of 32 evaluated features.

So `'kappa'` is documented as being **for completeness** — reproducing results from the deprecated `sequence_analysis.grammar`, and offering the bounded form — not because it reports anything `'nardini'` does not.

### Changed — patterning is 2-3x faster

Profiling the scramble path found real waste rather than a need for cleverness. Measured at 10,000 scrambles, best of three interleaved runs:

| backend | L=200 | L=500 | L=1000 |
|---|---|---|---|
| `nardini` (default) | 2.13x | 2.44x | 2.75x |
| `kappa` | 1.22x | 1.20x | 1.21x |
| `iwd` | 1.12x | 1.18x | 1.13x |

The gain grows with sequence length, because the wasted work scaled with the number of windows.

* **Unevaluated features were being computed and then discarded.** A feature is only scored when its residue types clear `min_fraction`, and a typical IDR clears about 15 of 36 — but all 36 were computed and the rest zeroed afterwards. `_asymmetries_from_counts` now takes the mask. This was the single biggest saving, in the stage that was two thirds of the runtime.
* **Window counts covered all 8 residue types** when only the ~5 appearing in an evaluated feature are ever read. They are now restricted, computed one type at a time into a single reused buffer rather than allocating an `(n, 8, L)` cube, and accumulated in `int16` rather than `int32`.
* **Permutation by `argsort` of uniform randoms** — `O(n log n)` plus two large temporaries — replaced with `rng.permuted`, an `O(n)` Fisher-Yates shuffle into a preallocated buffer. This was the largest single stage once the above landed.
* In `sparrow.patterning.vectorized`, `_delta_from_ternary` counted the `+1`s and `-1`s separately, materializing two boolean masks and two `int64` count arrays. The ternary encoding gives both quantities the asymmetry actually needs — `a - b` is the window sum of the ternary array, `a + b` the window sum of its non-zero indicator — so it now takes two cumulative sums over narrow types instead. Worth about 1.15x on `kappa_from_ternary_batch`.

**The first two produce bit-identical output**, verified against a worktree of the previous commit with `array_equal`. The permutation change does not: it consumes randomness differently, so a given `seed` now yields a different — statistically equivalent — set of scrambles. Checked rather than asserted: across 40 seeds, mean `|z|` moves from 1.2257 to 1.2262, the largest per-feature mean shift is 0.30 seed-to-seed standard deviations, and no feature's two-sample KS test falls below p = 0.165. As with `patterning_percentile`, do not compare a stored z-score against a freshly computed one; recompute instead.

**Fixed along the way:** `rng.permuted` returns an F-contiguous array when handed a broadcast view, and pocketfft takes a different path on non-C-contiguous input — enough that identical rows in one call came out differing in the last bit. That turned an IWD null which should be an exact point mass into one with ~1e-16 of spread. Both scramble loops now permute into C-ordered buffers, and `iwd_batch` calls `ascontiguousarray` so its results no longer depend on the caller's memory layout at all.

### Added — `nardini_proteins()`, a whole proteome in parallel

`nardini_proteins()` takes sequences or `Protein` objects, computes every NARDINI+ vector across multiple processes, and returns Protein objects with the results already attached — so `p.nardini.vector(...)` afterwards is free rather than recomputed. Input and output shapes match: a dict returns a dict with the same keys, anything else a list in input order. Sequences become new Protein objects; Protein objects passed in are populated **in place** and handed back. Mixed input works.

Measured on 160 IDRs at 1,000 scrambles: 18.7 s serial → 9.8 s (2 processes) → 5.1 s (4) → **2.7 s (8), a 6.9x speed-up**.

**Parallelism does not change the numbers**, and that is not an accident — it rests on the seeding fix above. Because per-sequence seeds derive from the sequence rather than its batch position, results are independent of chunking, process count and completion order. The parallel result is bit-identical to the serial one and to what each Protein computes alone; the tests assert exactly that across process counts, chunk sizes and both start methods. Nothing about the calculation is reimplemented — the work is chunked through the ordinary `nardini_vectors`, so there is no second code path to drift.

`on_error='skip'` drops sequences NARDINI cannot score (non-standard residues, too short) and warns with a summary, rather than losing an hours-long run to one malformed entry. A failing chunk is retried one sequence at a time, so a bad sequence never costs more than itself.

**Start method.** sparrow defaults to `fork` where it exists (Linux, macOS) rather than the platform default. Under `spawn` every worker re-imports the calling `__main__`, so a script calling this at module level calls it again in each worker — recursively spawning, which is a fork bomb rather than an error message, and a Jupyter notebook has no `__main__` guard available at all. `fork` re-imports nothing. As a backstop on `spawn` platforms, sparrow detects that it is running inside a worker and computes serially instead of building a nested pool, so the worst case is wasted work rather than a runaway process.

### Fixed — `patterning()` recomputed work it already had

`ProteinNardini.patterning()` keys its cache on `background=None`, since the patterning half needs no background. But that made it miss whenever the cached result had been computed *with* a background — so it re-ran the scrambling, which is essentially the entire cost. Anyone calling `patterning()` after `vector()`, or after `nardini_proteins()` had populated a proteome, paid for it twice. The patterning block is genuinely background-independent (verified: identical against the human and yeast backgrounds and against `patterning()` alone), so any cached result with matching patterning options now satisfies the call.

### Added — build your own GIN-style clusters

`sparrow.nardini` could only *assign* sequences to the 30 published human grammars. `build_gin_reference()` now does the other half — it runs the same procedure (clip to ±3, then k-means) over your own vectors and returns a `GINReference` that `assign_gin_clusters()` accepts in place of the bundled one. That is what you want for a non-human proteome, a filtered sub-proteome, or a designed library, where forcing sequences onto human grammars is the wrong move.

* **Any backend works.** Centroids live in the space of whatever statistic produced them, so a reference records its backend and assignment refuses to mix spaces. `'kappa'` vectors against `'kappa'` centroids is now fine; `'kappa'` vectors against the published `'nardini'` centroids is still refused. Previously the guard hard-coded `'nardini'` and so blocked the legitimate case along with the dangerous one.
* **`GINReference.save()` / `.load()`**, because the vectors behind a real IDRome cost hours and re-clustering with a different seed gives different (equally valid) centroids.
* **Derived descriptions.** Clustering cannot regenerate the published human-authored labels, so each cluster gets one built from its centroid's most extreme features — `'blocky neg-neg, high Frac E, D patches'`. Pass `descriptions=` to override. Clusters are numbered largest-first, which makes the numbering stable within a run but still meaningless across runs.
* **`compare_gin_references()`** answers "did I recover GIN?". Cluster numbering is arbitrary, so it pairs centroids one-to-one by solving the optimal assignment (Hungarian algorithm) and reports per-cluster distances plus *both* references' `separation_scale`. Both matter: a clustering that found no structure leaves its centroids near the origin, which makes them close to everything and produces a flatteringly small match distance — the giveaway is your own separation scale sitting far below the published one.
* `build_gin_reference()` warns via `ClusteringQualityWarning` below `MIN_SEQUENCES_PER_CLUSTER` (10) sequences per cluster on average, since k-means returns k centroids whether or not the data supports that many and will happily assign every query to one of them. The published analysis had ~800 IDRs per cluster.

No new dependency — k-means comes from `scipy.cluster.vq.kmeans2` with k-means++ init, run with restarts (lowest inertia wins) and with empty-cluster or non-finite solutions discarded as failed restarts rather than silently accepted.

Validated by planting four deliberately different grammars (D/E tracts, poly-Q, S-rich, K-blocks) and checking the clustering puts them back into four groups: 48/48 sequences recovered, with the derived descriptions correctly naming each planted grammar.

**Fixed while building this:** `GINReference.separation_scale` returns a *squared* distance (matching the module's convention throughout, including the distances on `GINAssignment`) but its docstring described it as a distance. `compare_gin_references` initially used true Euclidean, which made its `mean_distance` silently incomparable to the `separation_scale` it is meant to be read against — a degenerate clustering looked like close agreement (6.8 against a scale of 17.8) when the correct comparison shows the opposite (47.1 against 17.8). Both now use squared Euclidean and the docstring says so.

### Added — `build_background` warns about a background you should not trust

A NARDINI+ compositional background fails quietly rather than loudly. If a feature never varies across your reference set there is no scale to divide by, so its z-score comes back as exactly `0.0` for every sequence you ever score — which reads as "perfectly average" but means "unmeasurable". `build_background()` now emits a `BackgroundQualityWarning` for the two ways this happens:

* **Fewer than 50 sequences** (`MIN_RECOMMENDED_SEQUENCES`). The relative standard error on an estimated standard deviation is about `1 / sqrt(2(n-1))`, so 50 puts the per-feature scale within ~10%; 13 sequences gives ~20%.
* **More than 10% of the 54 features constant** (`MAX_DEGENERATE_FRACTION`). Calibrated against real data: of the nine published IDRomes, seven have zero constant features and two have one, so a genuine reference set sits at 0–2%. Sixty random IDR-like sequences give about 8 of 54; one sequence gives all 54.

The count is recorded in `background.metadata["num_degenerate_features"]` regardless, and the background is still returned and still usable — this is advice, not an error. New `SparrowWarning` base class (mirroring `SparrowException`) so sparrow's advisory warnings can be filtered as a group; `BackgroundQualityWarning` derives from it. The bundled backgrounds are loaded rather than built, so they never warn.

### Fixed — NARDINI edge cases

* **A degenerate null distribution produced a confident-looking z-score of 2.0 instead of 0.** Some statistics genuinely cannot vary under scrambling for some sequences — IWD on a sequence whose target group covers every position, for instance, where the merged `{D,E,K,R}` group spans an E/K-only sequence. The null is then a point mass and `z = 0` is the correct report, which `_fit_gamma_moments` handled with an `if sample_var == 0` guard. But `np.var` over N identical values does not reliably return exactly zero — the mean subtraction rounds, and whether the result is `0.0` or `3e-30` depends on N, because NumPy blocks pairwise summation differently. At 500, 1000 and 2000 scrambles it returned `3e-30`, so the guard missed and the caller divided a ~1e-15 numerator by a ~1e-15 denominator, reporting `z = 2.0`. The guard now tests the spread directly. A test asserting exactly this behaviour already existed but only ran at 200 scrambles, which happens to land in a lucky bucket; it is now parametrized across both.
* **Per-sequence seeds were derived from batch position, so the documented batch-independence guarantee did not hold.** `nardini_vectors()` derived each sequence's seed as `seed + index`. That is reproducible for an identical set in an identical order and nothing more: re-scoring one sequence out of a batch, or inserting a sequence ahead of it, silently changed its patterning z-scores. Seeds are now derived from the master seed and the sequence itself (via `hashlib.blake2b` — Python salts `hash()` per process, so it is not reproducible across runs), which makes the guarantee true: a sequence gets the same scramble stream scored alone, in a set, or in a differently-ordered set, and `Protein.nardini` now agrees exactly with `nardini_vectors()` on the same sequence. Again, a test for the property existed but only ever compared index 0 — precisely where the old derivation happened to agree.
* **`Protein.nardini` memoized on `id(background)`, which could return another background's answer.** The cache stored only the key and held no reference to the background, so once the caller dropped theirs the object was collected and CPython was free to allocate the next background at the same address — at which point `vector()`, `gin_cluster()` and `gin_report()` silently returned the *previous* background's result. It reproduces within a handful of allocations (4 address reuses in 300 in one run) and was observed reporting a compositional z-score of **+19.8 where the correct answer was −4.7**. Backgrounds are now keyed by a hash of their feature names and mean/std arrays, which is both sound and lets two separately-built but identical backgrounds share a cache entry. Same bug class as the `Protein.polymeric` caching fixed above, just with object identity standing in for the missing parameters.
* **`pathlib.Path` was rejected wherever a path was accepted.** `build_background(Path(...))` and `nardini_vectors(Path(...))` fell through the `isinstance(source, str)` check to the "iterable of sequences" branch and failed with `'PosixPath' object is not iterable`. Every path argument now accepts `os.PathLike`, including the `background=` argument on both `nardini_vectors()` and the `Protein.nardini` methods.

Per-sequence seed derivation runs before residue validation, so it encodes as UTF-8 rather than ASCII — otherwise a non-standard character raised `UnicodeEncodeError` instead of sparrow's "no NARDINI type" message.

Also corrected: the README claimed the `'kappa'` and `'iwd'` backends run 9–22x slower than `'nardini'`. That was true before those statistics were vectorized and is not any more — all three cost about the same, as the NARDINI guide already said.

### Fixed / Documentation

The rest of this entry is a correctness and documentation pass. The headline items are a class of memoization bugs in which cached results were keyed on too little information and were therefore returned for calls made with *different* parameters, and a validation helper that silently did nothing when given a custom error message.

### Fixed — incorrect results

* **`Protein.polymeric`: cached polymer values ignored their parameters.** `get_mean_nudep_saw_end_to_end_distance()`, `get_mean_nudep_saw_radius_of_gyration()` and `get_nudep_saw_end_to_end_distribution()` cached on a fixed key that omitted `nu` and `prefactor`; `get_saw_end_to_end_distribution()`, `get_mean_saw_end_to_end_distance()` and `get_mean_saw_radius_of_gyration()` omitted `prefactor`; and `get_mean_afrc_end_to_end_distance()`, `get_mean_afrc_radius_of_gyration()` and `get_afrc_internal_scaling()` omitted `mode`. The second and all subsequent calls on the same `Protein` returned the *first* call's value regardless of the parameters passed. All of these now key their cache on every argument that affects the result.
* **`sparrow.tools.utilities.validate_keyword_option()` did nothing when passed a custom `error_message`.** The invalid-keyword branch only raised when `error_message` was `None`, so supplying one disabled the validation entirely. It now always raises, using the custom message when given.
* **`Predictor.binary_disorder()` ignored its own arguments after the first call.** It shared the plain `metapredict-disorder` cache key with `disorder()` and `disorder_domains()`, so a result computed with metapredict's defaults was reused even when non-default `disorder_threshold`, `minimum_IDR_size`, `minimum_folded_domain` or `gap_closure` values were requested. The cache key now includes those parameters, and the previous "we recommend `recompute=True`" caveat no longer applies.
* **`calculate_molecular_weight("")` returned 18.0 instead of 0.0.** The peptide-bond correction subtracted `18 * (len(seq) - 1)`, which *added* a water for an empty sequence. Also affected `Protein("").molecular_weight`.
* **`Polymeric.empirical_nu()` and `Polymeric.empirical_radius_of_gyration()` silently ignored `mode`.** Any value was accepted and the `zheng2020` parameterization used regardless. Unrecognized modes now raise.
* **`show_sequence(newline=-1)` did not do what it documented.** `-1` is documented to mean "use the sequence length", but the `-1` sentinel was only resolved for `blocksize`, so `newline=-1` was silently clamped up to `blocksize` and lines still broke every 10 residues. Both sentinels are now resolved before the block/line consistency check.
* `grammar.PATCH_FEATURES` now documents *why* `"W Patch"` is absent (it matches NARDINI+, which drops tryptophan because W patches essentially do not occur in the proteome) and derives the list from the amino-acid alphabet so the intent is explicit rather than looking like an off-by-one. The feature schema is unchanged.

### Fixed — errors and robustness

* **All sparrow exceptions now derive from `SparrowException`.** `ProteinException`, `PatterningException` and `CalculationException` previously inherited straight from `Exception`, so `except SparrowException` did not catch them despite the docstrings implying it would.
* `Protein.show_sequence()` used mutable default arguments (`colors={}`, `bold_positions=[]`, `bold_residues=[]`, `opaque_positions=[]`), which are shared across every call for the lifetime of the process. They now default to `None`.
* `Protein.compute_kappa_x()` performed its group-ordering normalization *inside* the residue-validation loop, so an empty `group1` skipped it entirely and produced a `TypeError` from the cache-key construction. Empty groups now raise a clear `ProteinException`.
* `Predictor.binary_disorder()`'s `override_folded_domain_minsize` argument has no effect — current versions of metapredict's `predict_disorder_domains()` do not accept it. Setting it now raises a `DeprecationWarning` instead of being silently dropped; the argument will be removed in a future release.
* `phospho_isoforms`: `phospho_rate` was unvalidated, so a value above 1 quietly returned the full isoform set and a negative value quietly returned an empty list. Out-of-range rates and out-of-range phosphosite indices now raise, and the bare `Exception`s raised for a bad `mode` are now `SparrowException`s.
* `elm.generate_elm_df()` raised `UnboundLocalError` on a class file with no recognizable header row; it now raises a `SparrowException` explaining what was expected.
* `elm.ELM.probability` is annotated `float` but held the raw string parsed from the TSV; it is now coerced to `float`.
* `elm.ELM.__eq__` raised `AttributeError` when compared against a non-`ELM` object instead of returning `NotImplemented`.
* The ELM class table is now parsed once and cached rather than re-read from disk on every `Protein.elms` access — the lost/gained/retained helpers build a second `Protein`, so they were paying for it twice per call.

### Fixed — tests

* **The test suite could not be collected from the repository root.** Four test modules resolved their reference data through `os.getcwd()`, so `pytest` only worked from inside `sparrow/tests`. Paths are now resolved from `__file__` via a new `sparrow.tests.reference_data_path()` helper, and `pytest sparrow/tests` works from anywhere.
* **`test_albatross.py` was intermittently failing.** Its ~30 batch-vs-single prediction assertions use `atol=1e-05` on sequences generated by `build_seq()`, which draws from numpy's *global* unseeded RNG — so each run tested different sequences. Padding and batching legitimately change the order of float32 accumulation in the LSTM, and roughly 2 in 360 comparisons land just over that tolerance (worst observed delta 1.34e-05), so the module failed a meaningful fraction of runs. An autouse fixture now seeds the RNG per test, keeping the coverage while making any failure reproducible instead of a coin flip.
* Added `sparrow/tests/test_regressions.py` pinning the behaviour of every fix above.

### Documentation

* `Predictor.radius_of_gyration()` and `Predictor.end_to_end_distance()`: the primary `use_scaled` option was entirely absent from the `Parameters` sections, and the automatic short-sequence fallback was documented as 25 residues when the actual threshold (`MIN_LENGTH_ALBATROSS_RE_RG`) is 35. Return units (Angstroms) are now stated.
* `Predictor.low_complexity_domains()` documented a `return domains : bool` parameter that does not exist and gave no return description; `Predictor.disorder_domains()` described `return_boundaries` only in passing.
* Corrected return types: `pLDDT()` and `pscore()` return arrays, not lists; `track_tools`' `predefined_linear_track()`, `linear_track_composition()` and `build_track()` return `np.ndarray`, not `list`; `Protein.elms` returns a `set` of `ELM` dataclasses, not a list of NamedTuples.
* `Predictor.transactivation_domains()` described its return value as a nuclear-export-signal score (copy-paste error).
* `sequence_complexity`: the `permissive` flag was documented as "max_interruption is ignored", which is not what it does — `max_interruption` applies in both modes; `permissive` controls whether bridged interruption residues count toward the domain's residue fraction.
* `Polymeric.get_mean_afrc_radius_of_gyration()` documented its default `mode` as `'scaling law'`; it is `'distribution'`. `recompute` is now documented on every method that accepts it, the AFRC/SAW distribution getters' return type is described as the tuple it actually is, and `Polymeric.__init__`'s truncated docstring is completed.
* `plaac_prion_like_domains()` / `plaac.score_sequence()`: supplying `bg_freqs` while leaving `alpha` at its default of `1.0` raises a `ValueError`. This constraint is now documented on both.
* `batch_predict`: list inputs map to 0-based keys (`0..n-1`), not `1..n`; `prepare_model`'s `gpuid` is now documented; the batch guide now covers `version`, `gpuid`, `return_seq2prediction`, `show_progress_bar`, the `pad-n-pack` PyTorch requirement, and the all-strings-or-all-`Protein` input constraint.
* `docs/predictors.rst` referred to a `dssp()` method that does not exist (it is `dssp_helicity()`) and gave the weights directory as a literal `sparrow/data/networks/predictor`. Added guidance on documenting options and on keying memoization correctly.
* `docs/api_guides/protein.rst`: added the previously-undocumented `Protein.plugin` accessor, `Predictor.low_complexity_domains`, and the full `Polymeric` surface (SAW / nu-dependent SAW / empirical methods) to the capability index.
* **Python requirement corrected to 3.9+.** `pyproject.toml`, the README and the install guide all claimed 3.7+, but `sparrow.data.properties.aaindex_loader` subscripts builtin generics (`dict[str, object]`) at runtime, which requires 3.9.
* The install guide advertised `pip install idptools-sparrow`, which does not match the `sparrow` distribution name in `pyproject.toml`; it now documents the GitHub install the README describes, plus a development-install section.
* The README's batch-prediction section claimed MPS acceleration; `batch_predict` selects between CUDA and CPU only.
* `docs/index.rst`'s feature list omitted prefactor prediction, SHD, molecular weight, the domain/motif tools and the plugin system.

## 1.0.0 — first major release

The first major (1.0) release. This release centers on a large correctness,
robustness, and usability pass: numerous bug fixes (several producing
incorrect numbers), a full overhaul of the predictor backend, a corrected and
much faster batch-prediction path, a new amino-acid-property profiling feature,
greatly expanded documentation, and roughly doubled test coverage. The
user-facing `Protein` API is unchanged except for additions and the fixes noted
below.

### Added
* **AAindex property profiles.** New `Protein.linear_property_profile(mode, window_size, end_mode, smooth)`
  maps a sequence onto any of the 500+ published AAindex1 amino-acid scales as a
  windowed per-residue profile. It takes the same arguments as
  `linear_sequence_profile`; `mode` selects an index by a readable
  `<meaning>-<first-author>-<year>[-N]` identifier (e.g. `hydropathy-kyte-1982`)
  or by raw AAindex accession (e.g. `KYTJ820101`).
* New `sparrow.data.aaindex` module exposing the property database:
  `list_property_indices()`, `resolve_identifier()`, `get_property_values()`,
  and `get_property_metadata()`.
* New `sparrow.tools.track_tools.linear_track_property()` helper.

### Changed / Refactored
* **Predictor backend overhaul.** All `.pt` network loading, hyper-parameter
  inference, and `BRNN` construction are centralized in a single shared loader
  (`sparrow.predictors.network_loader.load_parrot_network`) plus a
  `BaseNetworkPredictor` base class and reusable output helpers
  (`sparrow.predictors.outputs`). The 17 predictor classes are now thin
  subclasses that declare only what differs (weights path, default version,
  architecture, output post-processing). This removed roughly 3,000 lines of
  duplicated boilerplate. **Public API, import paths, instance attributes, and
  numerical outputs are unchanged** (verified byte-for-byte against a
  pre-refactor baseline).
* The shared loader strips the legacy `module.` state-dict prefix only when it
  is actually present, replacing the previous hand-toggled per-file blocks.
* **`batch_predict`: the `pad-n-pack` algorithm now works and is the default.**
  The previously non-functional `pad-n-pack` path (it decoded the wrong network
  head and was hard-disabled) was reimplemented correctly using
  `pack_padded_sequence`, faithfully replicating `BRNN_MtO.forward`. It batches
  mixed-length sequences and is **3.5–5.5× faster** than `size-collect` on
  length-diverse datasets (with identical values), and is selected by default on
  PyTorch ≥ 1.11. Inference now runs under `torch.no_grad()` in evaluation mode.
* `predict_transmebrane_regions()` was renamed to the correctly-spelled
  `predict_transmembrane_regions()`; the old name remains as a backward-compatible
  alias.

### Fixed
* **SCD_x was computed incorrectly.** `compute_scd_x` omitted every
  nearest-neighbor (|i−j| = 1) charge pair from the Sawle–Ghosh sum; it now sums
  over all pairs. The Cython extension was recompiled and the test fixture
  regenerated from an independent canonical reference.
* **`Protein.SHD` returned `None`.** The property had no body; it now computes
  the sequence hydropathy decoration.
* **`linear_sequence_profile(end_mode='zero-ends')`** returned a track shorter
  than the sequence; the C-terminus is now padded correctly.
* **`Protein.generate_phosphoisoforms`** raised `AttributeError` for every call
  (it passed a sequence string where a `Protein` object was expected); fixed for
  all modes.
* **`Protein.compute_iwd`** passed a `str` to a `list`-typed Cython function;
  it now passes a list.
* **`low_complexity_domains(mode='holt-permissive')`** silently ignored
  `fractional_threshold`; it is now forwarded, and the mode is reachable.
* **`show_sequence`** had a bare `raise` (raising `RuntimeError`), produced a
  malformed `<p>` tag, and used mutable default arguments — all fixed.
* **`elm.parse_hgvs`** used `assert ..., SparrowException(...)`, which never
  raised the intended exception (and is stripped entirely under `python -O`);
  replaced with a real `raise`.
* **`Alignment.display_msa`** was a `@property` that declared arguments (and so
  could never receive them) and printed the literal string `"None"`; it is now a
  regular method with correct output.
* **`io.uniprot_fetch`** parsed the HTTP response in a way that always produced
  an empty sequence; FASTA parsing was rewritten to decode and parse correctly.
* Numerous docstring and error-message corrections, including the predictor
  error messages (`raw_valus` → `raw_values` and correct residue names), the
  `fraction_polar` description, the `kappa` and `hydrophobicity` documentation,
  and `Protein.__repr__` formatting.

### Documentation
* The Sphinx documentation was reorganized and substantially expanded:
  a "What is sparrow" overview; a detailed installation guide (virtual
  environments with `pip` or `uv`, installing from PyPI or GitHub); worked,
  runnable examples (including plotting a linear NCPR profile); and a single,
  capability-organized **"The Protein Object"** reference that documents
  everything reachable from a `Protein` — direct members plus the `predictor`
  and `polymeric` accessors — in one place. An auto-generated reference table
  catalogues all AAindex property indices.

### Tests
* The test suite was roughly doubled to **288 tests**, adding 14 new modules
  covering previously-untested code (general tools, parameter calculations,
  track tools, sequence complexity, patching, physical properties, sequence
  compression, phosphoisoforms, grammar features, sequence visualization, the
  AAindex registry, predictor internals, and a comprehensive `Protein` API
  suite). Every fix above is guarded by a regression test.

---

## Earlier releases

#### June 2025 (version 0.2.3 release)
* Fixed an indexing bug in Sequence Hydropathy Decoration calculations
* Removed numpy < 2.0 requirement in main; numpy now requires `>1.14.0` only. We may move to force `>2.0` in the future but this will be homogenized across the HHL computational infrastructure should it happen.

#### November 2024 (version 0.2.3 release)
* Updated to using pyproject.toml for package data
* Fixed tests to work with metapredict V3

#### June 2024 (version 0.2.3 release)
* Cythonized SHD/SCD and IWD clustering sequence parameters

#### Sept 2023 (version 0.2.2 release)
* Updated low complexity domain identification code.
* Added ability to grey out sequences in `show_sequence()` [h/t Garrett!].

#### May 2023 (version 0.2.1 release)
* Update to ALBATROSS v2 networks (all networks use the `v2` by default both in individual predictors and batch predictions
* Re-wrote much of `batch_predict()` code. Changes here include
	* Implementation of the `size-collect` algorithm to ensure the passed batchsize does not impact the accuracy of predictions. Batch prediction can now use larger batch sizes, providing better performance on both GPUs and CPUs
	*  Set default batch size to 32
	*  Improved robustness of input types `batch_predict()` can accept. Can now take dictionaries and lists of sparrow.protein.Protein objects or dictionaries/lists of sequences.
	*  Changed order of input parameters for `batch_predict()`, such that now the only required options are [0] Input list/dictionary and [2] name of the network to be used.
	*  Updated return type for `batch_predict()` such that now the return type by default is a dictionary that maps input IDs (or list positions) to sequence and prediction. The original return behavior (a dictionary that maps sequence to prediction) can be obtained if the `return_seq2prediction` flag is set to True.
	*  Wrote much more extensive tests for all `batch_predict()` code
	*  Ensure `batch_predict()` guarantees the same return order as the input order if possible. The only exception is if return_seq2prediction=True and duplicate sequences are found in the input data, in which case only the first occurrence of a given sequence is included. Also order here refers to the order if the return dictionary had dict.values() called.
*  **Added scaled-network for small sequences**: In the course of testing the networks we noticed that in both V1 and V2, when sequences are short (<30-40 amino acids) the non-scaled Re and Rg predictors can return non-sensical results. In contrast, the `scaled_rg` and `scaled_re` networks show reasonable and reproducible polymeric behavior for these smaller sequences. To address this, in both single sequence predictions and batch predict, by default, even if an `rg` or `re` network is requested, if the sequence is less than 35 residues long, we force the `scaled_rg` or `scaled_re` networks. This can be over-ridden by setting the 'safe' keyword in either `batch_predict()` or the single sequence `radius_of_gyration()` or `end_to_end_distance()`
*  Technical change: the end-to-end distance predictor module found under sparrow/predictors was renamed from `re` to `e2e` to prevent clashing with Python's regular expression (`re`) package. This does not introduce any errors, but makes debugging predictors challenging. The actual network name is retained as `re`.
*  Default predictor for `Protein.predictor.radius_of_gyration()` and `Protein.predictor.end_to_end_distance()` use `use_scaled=True` as a default, based on accuracy of the V2 networks.

#### May 2023 (version 0.2 release)
* First major 'alpha' release to coincide with ALBATROSS preprint

#### Feb 2023
* Building early version of ALBATROSS

#### Nov 2022
* Major set of updates to fix errors in how kappa is calculated
* Added SCD and SHD functions
* Added some tests
* Moved `scd.py` into `sparrow.patterning`
* Updated IWD for bivariate charge clustering (h/t Garrett Ginell)
* Fixed some tests (h/t Jeff Lotthammer)
* Improved docstrings

#### July 2022
* Moved to idptools! If you had previously cloned sparrow, you can update your git remote location using:

		git remote set-url origin git@github.com:idptools/sparrow.git

* Updated requirement for Python 3.7 or 3.8
