# sparrow's bundled data (`sparrow/data`)

The data sparrow ships with: the trained networks behind its predictors, amino-acid property databases, NARDINI+ reference data, and the definitions behind its composition properties. It is also the Python package `sparrow.data`. Everything here is installed with sparrow (`MANIFEST.in` includes the whole directory), and code finds a file in it with `sparrow.data_path`, which takes the file's path relative to this directory - `sparrow.data_path("elm_classes.tsv")`, for example.

## Files

| File | What it holds | Used by |
|---|---|---|
| `__init__.py` | Makes this directory the package `sparrow.data`, and imports `configs` and `amino_acids` | Everything that imports `sparrow.data` |
| `amino_acids.py` | The definitions behind sparrow's composition properties, by one-letter code: the 20 valid amino acids, residue groups (aromatic, aliphatic, polar, charged, positive, negative, disorder-promoting, chain-expanding and so on), three-to-one letter codes, molecular weights, plotting colours, hydropathy scales (Kyte-Doolittle, raw and normalised, and HPS), PPII propensities (Hilser) and side-chain pKa values | `sparrow.properties`, `Protein`, `sparrow.nardini`, the ELM annotations and the plotting functions in `sparrow.visualize` |
| `aaindex.py` | Readable names for the AAindex1 amino-acid indices in `properties/aaindex1.json`, as `<meaning>-<first author>-<year>` (for example `hydropathy-kyte-1982`), with `property_values` and `list_property_indices` | `Protein.linear_property_profile`; the profile functions in `sparrow.properties.profiles` take its `property_values` as a scale |
| `configs.py` | Settings shared by several modules. `MIN_LENGTH_ALBATROSS_RE_RG` (35 residues) is the length below which the radius of gyration and end-to-end distance always come from the scaled networks, which are much more accurate for short sequences. `DISORDER_THRESHOLD` (0.7) is not read anywhere in sparrow at present | `sparrow.predictors` and `sparrow.predictors.batch_predictor` |
| `elm_classes.tsv` | The 349 Eukaryotic Linear Motif (ELM) classes, each with its regular expression and the probability of matching by chance, as downloaded from the ELM resource (download version 1.4, 2 April 2024). Cite the ELM resource if you use them: Kumar M, et al. ELM - the Eukaryotic Linear Motif resource - 2024 update. *Nucleic Acids Res.* 2024;52(D1):D442-D455 | `sparrow.sequence_analysis.elm`, and so `Protein.elms` |

## Directories

| Directory | What it holds |
|---|---|
| [`networks`](networks/README.md) | The trained networks behind `Protein.predictor` and `batch_predict`, one directory per predictor, with every version of each |
| [`nardini`](nardini/README.md) | NARDINI+ and GIN reference data: the GIN cluster centroids and definitions, and per-species compositional backgrounds, for `sparrow.nardini` |
| [`properties`](properties/README.md) | The AAindex amino-acid property databases (single-residue indices and pairwise matrices), with a stand-alone reader |

Each directory has a README describing its files, where they come from, and how sparrow uses them.
