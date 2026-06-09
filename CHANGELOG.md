# Changelog

All notable changes to SPARROW are documented here.

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
