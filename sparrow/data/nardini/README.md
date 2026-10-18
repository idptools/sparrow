# NARDINI+ reference data (`nardini`)

The reference data `sparrow.nardini` scores sequences against: the GIN cluster centroids that sort an IDR into one of 30 molecular grammars, and the per-species compositional backgrounds that turn its compositional features into z-scores. Almost all of it is redistributed unchanged from the GIN resource of Ruff et al. (*Cell* 2025), under its MIT licence, and must be cited if you use it. [`SOURCES.md`](SOURCES.md) gives the full provenance, licence, citations and how to regenerate each file, and [`LICENSE-GIN`](LICENSE-GIN) is the licence.

## Files

| File | What it holds | Loaded by |
|---|---|---|
| `gin_centroids_human_v1.npz` | The 30 published GIN cluster centroids, a 30 x 90 matrix of NARDINI+ z-scores (36 patterning features, then 54 compositional ones) | `load_gin_reference("published")` |
| `gin_cluster_definitions_human_v1.csv` | The description of each published cluster, and how many of the 24,508 human IDRs are in it | `load_gin_reference("published")` |
| `gin_centroids_human_corrected_v1.npz` | The published clusters refit to NARDINI+ vectors computed with sparrow's patterning windows, which leave out an extra, partial window the published vectors include (see `SOURCES.md`) | `load_gin_reference()`, the default |
| `gin_cluster_definitions_human_corrected_v1.csv` | The cluster descriptions, with the IDRs recounted against the refit centroids | `load_gin_reference()`, the default |
| `gin_composition_prior_<species>_v1.npz` | The mean and standard deviation of each of the 54 compositional features across a species' IDRs (at least 30 residues), for `human`, `mouse`, `rat`, `xenopus`, `fly`, `zebrafish`, `yeast`, `worm` and `arabidopsis` | `load_reference_background(species)`, or `NardiniBackground.load()` |
| `SOURCES.md` | Where every file comes from, its licence and citations, and how to regenerate it | |
| `LICENSE-GIN` | The GIN resource's MIT licence, which covers the files redistributed from it | |

## How sparrow uses them

`sparrow.nardini.load_gin_reference` (in `sparrow/nardini/gin.py`) reads a centroid file and its cluster definitions into a `GINReference`, and `assign_gin_clusters` puts each NARDINI+ vector in the cluster with the nearest centroid. `sparrow.nardini.load_reference_background` (in `sparrow/nardini/background.py`) reads a species' compositional background into a `NardiniBackground`, against which a sequence's compositional features are scored. `Protein.nardini` and the batch NARDINI+ functions use the human background and the corrected centroids by default; the API guide on NARDINI+ (`docs/api_guides/nardini.rst`) covers them.

`devtools/scripts/build_nardini_reference_data.py` rebuilds the published files from their canonical sources, and checks the upstream feature order still matches sparrow's before writing anything. `devtools/scripts/build_corrected_gin_reference.py` rebuilds the corrected centroids.
