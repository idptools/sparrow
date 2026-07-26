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
