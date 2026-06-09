# SPARROW

**S**equence **PAR**ameters for **R**egi**O**ns in **W**indows — a lightweight, object-oriented toolkit for analyzing and predicting features of protein sequences, with a particular focus on intrinsically disordered regions (IDRs).

**Current version is 1.0.x (June 2026)** 

[![Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/)
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

- **Sequence parameters** — composition, charge (FCR, NCPR, κ, SCD, SHD), hydrophobicity, sequence complexity, residue clustering (IWD) and patches.
- **Linear profiles** — per-residue windowed tracks for any of the above, plus **500+ published amino-acid property scales** (AAindex).
- **Deep-learning predictions** via `Protein.predictor` — disorder and pLDDT, DSSP secondary structure, polymer dimensions (Rg, Re, scaling exponent, asphericity, prefactor), phosphorylation, localization signals, transactivation domains, transmembrane regions, and phase-separation propensity. Built on the [ALBATROSS](https://www.nature.com/articles/s41592-023-02159-5) networks.
- **Polymer-model properties** via `Protein.polymeric` — analytical and simulation-derived dimensions and distance distributions.
- **Scale-up tools** — fast batch prediction across whole sequence sets, and fixed-length feature vectors for machine learning.
- **Extensible** — a simple plugin system for adding new sequence analyses.

## Installation

SPARROW runs on **Python 3.7+** and includes compiled (Cython) extensions. Installing into an isolated environment is recommended.

```bash
# create and activate an environment (venv shown; uv also works)
python -m venv sparrow-env
source sparrow-env/bin/activate          # Windows: sparrow-env\Scripts\activate

# install the latest version from GitHub
pip install git+https://github.com/idptools/sparrow.git
```

> Installing from GitHub compiles the Cython extensions, so a C compiler is required (Xcode command-line tools on macOS, `build-essential` on Debian/Ubuntu, MSVC build tools on Windows). NumPy is installed automatically.

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

Batch mode predicts thousands of sequences in seconds; this parallelizes particularly well on GPUs and MPS but also on x86 CPUs.

```python
from sparrow.predictors import batch_predict

sequences = {"p1": "MKYLAAYLLL...", "p2": "GRGRGGYGG...", "p3": "QQQQAASS..."}

# {key: [sequence, prediction]}
results = batch_predict.batch_predict(sequences, network="re")
```

Networks available  for batch predict are: `rg`, `scaled_rg`, `re`, `scaled_re`, `prefactor`,
`scaling_exponent`, `asphericity`. As with single-sequence predictions, we
**strongly recommend the `scaled_rg` and `scaled_re` networks**.

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

# build the Cython extensions in place and run the test suite
python setup.py build_ext --inplace
cd sparrow/tests && pytest
```

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
