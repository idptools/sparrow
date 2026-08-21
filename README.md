# SPARROW

**S**equence **PAR**ameters for **R**egi**O**ns in **W**indows — a lightweight, object-oriented toolkit for analyzing and predicting features of protein sequences, with a particular focus on intrinsically disordered regions (IDRs).

**Current version is 1.0.x (June 2026)** 

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

### Full documentation

SPARROW'S [official documentation is here](https://idptools-sparrow.readthedocs.io/en/latest/), this readme provides a very quick intro.

### Brief overview

Everything in SPARROW hangs off a single `Protein` object: create one from a sequence, then read parameters, properties, and predictions directly off it. Calculations are lazy and cached, so a `Protein` is cheap to make and you only compute what you ask for.

```python
from sparrow import Protein

p = Protein("MEEEKKKKSSSTTTDDDQQQQNNNN")
p.FCR                            # fraction of charged residues
p.kappa                          # charge patterning
p.predictor.disorder()           # per-residue disorder prediction
p.predictor.radius_of_gyration() # ALBATROSS radius of gyration
```

SPARROW integrates direct sequence calculations with per-residue predictions from deep-learning models trained using [PARROT](https://elifesciences.org/articles/70576), and is designed to be easy to extend with new analyses.

---

## Features

- **Sequence parameters** — composition, charge (FCR, NCPR, κ, SCD, SHD, isoelectric point, charge at any pH), hydrophobicity, conformational propensity (chain-expanding and disorder-promoting fractions, polyproline II), R/K and E/D ratios, sequence complexity, residue clustering (IWD) and patches.
- **Linear profiles** — per-residue windowed tracks for any of the above, plus **500+ published amino-acid property scales** (AAindex).
- **Deep-learning predictions** via `Protein.predictor` — disorder and pLDDT, DSSP secondary structure, polymer dimensions (Rg, Re, scaling exponent, asphericity, prefactor), phosphorylation, localization signals, transactivation domains, transmembrane regions, and phase-separation propensity. Built on the [ALBATROSS](https://www.nature.com/articles/s41592-023-02159-5) networks.
- **Polymer-model properties** via `Protein.polymeric` — analytical and simulation-derived dimensions and distance distributions.
- **Molecular grammars** — a native NARDINI / NARDINI+ implementation (`sparrow.nardini`) for detecting non-random residue patterning, with user-buildable compositional backgrounds, assignment to the 30 published human GIN grammars, clustering to derive your own, and parallel computation across a whole proteome.
- **Scale-up tools** — fast batch prediction across whole sequence sets, and fixed-length feature vectors for machine learning.
- **Extensible** — a simple plugin system for adding new sequence analyses.

## Installation

SPARROW runs on **Python 3.10+** and is **pure Python** — nothing to compile, no C compiler needed. Installing into an isolated environment is recommended.

```bash
# create and activate an environment (venv shown; uv also works)
python -m venv sparrow-env
source sparrow-env/bin/activate          # Windows: sparrow-env\Scripts\activate

# install the latest version from GitHub
pip install git+https://github.com/idptools/sparrow.git
```

> Earlier releases shipped Cython extensions and required a C compiler. Those have been replaced by vectorized NumPy implementations, so installation is now a plain pure-Python wheel with no build step.

Verify the installation:

```bash
python -c "import sparrow; print(sparrow.__version__)"
```

## Quick start

SPARROW is written in a **Protein-centric** way — almost everything is reached through the `Protein` object.

```python
from sparrow import Protein

p = Protein("MGSQSSRSSSQQQQQQQ")

# composition & charge
p.FCR, p.NCPR, p.amino_acid_fractions["Q"]

# patterning & complexity
p.kappa, p.SCD, p.complexity

# per-residue profiles (great for plotting)
p.linear_sequence_profile(mode="NCPR", window_size=8)
```

### Reading FASTA files

```python
from sparrow import read_fasta

proteins = read_fasta("my_fasta_file.fasta")   # {header: Protein}
```

`read_fasta()` accepts the same keyword arguments as [`protfasta.read_fasta`](https://protfasta.readthedocs.io/en/latest/read_fasta.html).

## ALBATROSS predictions

ALBATROSS conformational-property predictions are available two ways: per-protein predictions, and batch prediction for large sequence sets.

### Single-protein predictions

```python
from sparrow import Protein

P = Protein("MKYLAAYLLLNAAGNTPDATKIKAILESVGIEIEDEKVSSVLSALEGKSVDELITEGNEKLAAVPAAGPASAGGAAAASGDAAAEEEKEEEAAEESDDDMGFGLFD")

P.predictor.asphericity()
P.predictor.radius_of_gyration()                 # use_scaled=True by default
P.predictor.end_to_end_distance(use_scaled=True)
P.predictor.scaling_exponent()
P.predictor.prefactor()
```

Rg and Re can be predicted with the `use_scaled` flag, which uses networks trained on `Rg/√N` and `Re/√N`. We **recommend leaving `use_scaled=True`** (the default): it is much more accurate for short sequences and is the mode used in the main-text ALBATROSS figures.

### Batch predictions

Batch mode predicts thousands of sequences in seconds; this parallelizes particularly well on CUDA GPUs but is also a large speed-up on CPUs.

```python
from sparrow.predictors import batch_predict

sequences = {"p1": "MKYLAAYLLL...", "p2": "GRGRGGYGG...", "p3": "QQQQAASS..."}

# {key: [sequence, prediction]}
results = batch_predict.batch_predict(sequences, network="re")
```

Networks available  for batch predict are: `rg`, `scaled_rg`, `re`, `scaled_re`, `prefactor`,
`scaling_exponent`, `asphericity`. As with single-sequence predictions, we
**strongly recommend the `scaled_rg` and `scaled_re` networks**. Note that with the
default `safe=True`, the `rg` and `re` networks automatically hand sequences shorter
than 35 residues to their scaled counterparts.

See [the batch prediction guide](https://idptools-sparrow.readthedocs.io/en/latest/api_guides/batch.html)
for all options.

## NARDINI / NARDINI+ molecular grammars

`sparrow.nardini` is a native implementation of the [NARDINI](https://www.sciencedirect.com/science/article/pii/S0022283621006100) analysis:
it asks whether the residues in an IDR are arranged non-randomly by comparing each
sequence against scrambles of itself. NARDINI+ adds a block of compositional
features scored against a reference set of IDRs that you build yourself.

For a single sequence, everything hangs off `Protein.nardini`:

```python
from sparrow import Protein

p = Protein("MENDPRVRKFASDGAQWAIKWQKK...")

p.nardini.vector(seed=42)        # 90-feature NARDINI+ vector
p.nardini.patterning(seed=42)    # just the 36 patterning z-scores
p.nardini.composition()          # just the 54 compositional z-scores
p.nardini.gin_cluster(seed=42)   # nearest published human GIN grammar
print(p.nardini.gin_report(seed=42))
```

For a set of sequences:

```python
from sparrow.nardini import load_reference_background, nardini_vectors

# 1. a compositional background: published per-species, or built from your own IDRs
background = load_reference_background("human")     # bundled GIN prior
# background = build_background("my_idrome.fasta")  # or your own

# 2. NARDINI+ vectors (36 patterning + 54 compositional z-scores)
result = nardini_vectors("my_idrs.fasta", background=background,
                         num_scrambles=10000, seed=42)

result.to_table()                   # 90 columns, one row per IDR
result.top_features("IDR1", n=5)    # the most non-random features

# 3. assign each IDR to one of the 30 published human GIN clusters
result.gin_clusters().records()
```

For a whole proteome, `nardini_proteins()` does the same work across multiple processes and hands back `Protein` objects with the vectors already attached, so nothing is recomputed afterwards:

```python
from sparrow import read_fasta
from sparrow.nardini import nardini_proteins

proteins = nardini_proteins(read_fasta("my_idrs.fasta"), num_scrambles=10000,
                            seed=42, show_progress=True, on_error="skip")

proteins["IDR1"].nardini.vector(seed=42, num_scrambles=10000)   # cached, instant
proteins["IDR1"].nardini.gin_cluster(seed=42, num_scrambles=10000)
```

It takes sequences or `Protein` objects, as a dict or a list, and returns the same shape back — `Protein` objects are populated in place. On 160 IDRs at 1,000 scrambles: 18.7 s on one core, 2.7 s on eight. Parallelism does not change the numbers; per-sequence seeds derive from the sequence rather than its position in the batch, so the result is identical however it was chunked.

You can also **derive your own grammars** instead of assigning to the human ones, which is what you want for a non-human proteome or a designed library:

```python
from sparrow.nardini import build_gin_reference, compare_gin_references

reference = build_gin_reference(result, k=30, seed=0)   # k-means, as GIN does
reference.save("my_grammars.npz")                      # the vectors cost hours
compare_gin_references(reference)                      # how close is this to GIN?
```

**NumPy, not DataFrames.** Results are plain NumPy arrays, optionally with the feature names attached so you can say `v["Frac K+R"]` instead of remembering an index. The raw array is always `.values`, so filtering/sorting/plotting happen with NumPy as usual. `.to_dataframe()` is there if you would rather have pandas.

Patterning z-scores are computed against each sequence's own scrambles, so they
need no background and are comparable across sequences: **positive means blockier
than composition alone predicts, negative means more evenly interspersed**. Only
the compositional half uses the background — omit it to get just the 36 patterning
features.

Three patterning statistics are selectable via `backend`:

| `backend` | statistic | notes |
|---|---|---|
| `'nardini'` (default) | published NARDINI omega/kappa window asymmetry | **required for GIN** |
| `'kappa'` | the same asymmetry, normalized by `delta_max` | bounded 0–1; equivalent once z-scored |
| `'iwd'` | sparrow's inverse-weighted distance | ignores `blob_size` |

`'nardini'` z-scores the raw asymmetry (**delta**); `'kappa'` z-scores delta divided by `delta_max`. Since `delta_max` depends only on composition and every scramble preserves it, that divisor cancels out of the z-score — the two measure the same thing (correlation 0.97–1.00 in practice, differing only because `kappa_x` pads the ends and caps at 1). **`'kappa'` is provided for completeness**, and for continuity with the deprecated `sparrow.sequence_analysis.grammar`, whose values it reproduces to ~1e-6 (sparrow computes them in double precision where `grammar` used single). All three are vectorized over scrambles and all are fast enough to choose between on scientific grounds — at L=1000 and 10,000 scrambles, 0.81 s / 1.50 s / 1.45 s for `nardini` / `kappa` / `iwd`. Values are **not** comparable across backends, though; `result.backend` records which one produced them.

GIN cluster assignment maps sequences onto the published human grammars
(`'D/E-tracts'`, `'S patches'`, ...). Nearest-centroid assignment never abstains,
so three diagnostics tell you when not to believe a label: `separation` (gap to the
runner-up — small means the IDR sits *between* grammars), `composition_rms_z` (how
far the sequence is from the reference IDRome — ~1 is typical, >2 is not), and
`n_saturated`. `is_typical` combines the latter two, and `gin_report()` spells it
out in prose.

**Because the clusters describe the human IDRome, a label on something very unlike
a human IDR is an extrapolation** — sparrow flags this rather than hiding it. The
patterning half of the vector is unaffected and stays meaningful for any sequence,
since it is scored against the sequence's own scrambles.

The patterning calculation is vectorized over scrambles and skips the features
the abundance gate has already excluded, making it ~100x faster than the
reference implementation; a 1000-residue IDR at the published 100,000 scrambles
takes ~7 s rather than ~13 min.

Two worked example notebooks are in [`examples/`](examples): `nardini_from_protein_objects.ipynb` for the single-sequence interface and `nardini_module.ipynb` for the batch one. Both are executed, so you can read the outputs without running anything.

The bundled per-species backgrounds and GIN centroids are redistributed from the
[GIN resource](https://github.com/kierstenruff/RUFF_KING_Grammars_of_IDRs_using_NARDINI-)
under the MIT licence. **Please cite the original work** (Cohan et al. 2022 for
NARDINI; Ruff et al. 2025 for GIN) — see
[the NARDINI guide](https://idptools-sparrow.readthedocs.io/en/latest/api_guides/nardini.html)
for the full citation list.

## Amino-acid property profiles

Map a sequence onto any of the 500+ published AAindex scales as a windowed
per-residue profile:

```python
p = Protein("MEEEKKKKSSSTTTDDDQQQQNNNN")
p.linear_property_profile("hydropathy-kyte-1982", window_size=9)
```

Indices are addressable by a readable `<meaning>-<author>-<year>` identifier or
by AAindex accession; see `sparrow.data.aaindex.list_property_indices()`.

## Documentation

Full documentation is built with [Sphinx](https://www.sphinx-doc.org/) and lives
in [`docs/`](docs). It is available online at (https://idptools-sparrow.readthedocs.io/en/latest/). 



The documentation  includes an installation guide, worked examples (including plotting a linear NCPR profile), and a complete, capability-organized reference for everything reachable from a `Protein`. 

To build it locally:

```bash
pip install -r docs/requirements.txt
python -m sphinx -b html docs docs/_build/html
```

## Citation

If you use the ALBATROSS predictors, please cite:

> Lotthammer, J. M.; Ginell, G. M.; Griffith, D.; Emenecker, R. J.; Holehouse, A. S.
> *Direct Prediction of Intrinsically Disordered Protein Conformational
> Properties from Sequence.* **Nat. Methods** (2024).

Otherwise, please for now cite as "sparrow: **S**equence **PAR**ameters for **R**egi**O**ns in **W**indows (https://github.com/idptools/sparrow)". A preprint will be forthcoming, at some point...

IMPORTANTLY: If you cite specific tools, metrics, parameters etc you MUST cite the original papers. Ignore sparrow citations if you want, but please ensure you cite the original authors of the key foundational work that sparrow provides access to.

## Development

```bash
git clone https://github.com/idptools/sparrow.git
cd sparrow
pip install -e .

# run the test suite (works from anywhere in the repo)
pytest sparrow/tests
```

There is no build step. If you are updating a checkout that predates the removal
of the Cython extensions, delete any stale compiled artifacts first —
`rm -f sparrow/patterning/*.so sparrow/patterning/*.c` — since a leftover `.so`
would shadow the new pure-Python module of the same name.

Contributions are welcome — `sparrow` is designed so that new predictors and
plugins are easy to add (see the developer guides in the documentation).

## Changelog

The full version history is in [CHANGELOG.md](CHANGELOG.md). The latest release
is **1.0.0**.

## License

Released under the MIT License — see [LICENSE](LICENSE).

## Authors & acknowledgements

Built in the [Holehouse Lab](https://www.holehouselab.com/) by Alex Holehouse, Ryan Emenecker, Jeff Lotthammer, Nick Razo, Garrett Ginell, and Dan Griffith.

Copyright © 2020–2026 the SPARROW authors.
