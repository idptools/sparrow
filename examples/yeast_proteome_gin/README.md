# Molecular grammars of the yeast IDRome

A worked, end-to-end example: deriving GIN-style molecular grammars for *Saccharomyces cerevisiae* from nothing but a proteome FASTA.

The published [GIN resource](https://github.com/kierstenruff/RUFF_KING_Grammars_of_IDRs_using_NARDINI-) partitions the **human** IDRome into 30 clusters of shared molecular grammar. This example does the equivalent for yeast, which is the case sparrow's clustering support exists for: the published centroids live in human NARDINI+ z-score space, so applying them to a non-human proteome gives labels that describe which *human* grammar an IDR most resembles, scored against a population it does not belong to.

## Contents

| file | what it is |
|---|---|
| `yeast_idrome_gin_construction.ipynb` | the analysis, executed, with outputs saved |
| `uniprotkb_proteome_UP000002311_2026_07_27.fasta` | *S. cerevisiae* S288c reference proteome (UniProt UP000002311), 6,067 proteins |
| `yeast_gin_clusters.npz` | the derived reference — 30 yeast grammars, reusable |

## What the notebook does

1. **Predicts disordered regions** for every protein with [metapredict](https://github.com/idptools/metapredict), giving ~8,200 IDRs.
2. **Builds a yeast compositional background** — the 54 compositional features' mean and standard deviation across the yeast IDRome, so a z-score means "unusual *for yeast*".
3. **Computes a NARDINI+ vector per IDR**: 36 patterning z-scores against 10,000 scrambles of each sequence, plus 54 compositional z-scores against that background.
4. **Clusters them** into 30 yeast-native grammars with `build_gin_reference()`, following the published procedure (clip to ±3, then k-means).
5. **Looks inside the clusters** — which genes are in each grammar, and what their IDRs look like.
6. **Compares to the published human GIN**, pairing centroids one-to-one to see which grammars are analogous.

Runtime is around five minutes: about 40 s for disorder prediction and about 3 minutes for the vectors on 16 processes.

## What it finds

8,199 IDRs across 4,775 of the 6,067 proteins, sorted into 30 grammars. The largest are broad charge and polarity classes; the distinctive ones are the compositionally extreme tracts. A few spot-checks suggest the clustering is picking up real biology rather than noise:

| protein | IDR | assigned grammar |
|---|---|---|
| **SUP35** | N-terminal, 254 aa | `Q patches, high Frac Q, blocky pol-pol` |
| **PAB1** | second IDR, 79 aa | `Q patches, high Frac Q, blocky pol-pol` |
| **NPL3** | first IDR, 133 aa | `P patches, high Frac P, blocky pol-pro` |
| **NPL3** | second IDR, 145 aa | `high Frac G, high Frac Polar, low PPII` |

Sup35 is the canonical yeast prion and its N-terminal domain is Q/N-rich; Pab1's second IDR is its proline-rich P domain, the region implicated in its stress-induced condensation; Npl3 carries an RGG region, which lands in the glycine-rich grammar. None of that was given to the clustering — it works from the 90-dimensional vectors alone.

Comparing the yeast centroids against the published human ones (`compare_gin_references()`) pairs them sensibly: the yeast "low Iso point, well-mixed hyd-neg, low NCPR" grammar matches human `'Weak negative charge'` at a distance of 1.8, "high Frac K, high Frac K+R" matches `'Weak positive charge'`, and so on. Mean match distance is 16.4 against a published centroid spacing of 17.8, with our own spacing at 15.2 — comparable structure, described in comparable terms.

That comparison is suggestive rather than quantitative: the yeast vectors are z-scored against a yeast background and the human centroids live in human z-score space, so the axes are not strictly commensurate.

As a sanity check on the whole pipeline, `composition_rms_z` has a median of 0.76 — close to the 1.0 expected when sequences are scored against a background drawn from their own population — and only 4% of IDRs are flagged atypical.

## Reusing the derived reference

The saved reference is the useful artifact. Load it and assign new yeast IDRs — a mutant, a designed construct, a set of screen hits — into the same grammars without recomputing anything:

```python
from sparrow.nardini import nardini_vectors, build_background, assign_gin_clusters
from sparrow.nardini.gin import GINReference

reference = GINReference.load("yeast_gin_clusters.npz")

# score new IDRs against the same background the reference was built in
vectors = nardini_vectors(my_idrs, background=background, num_scrambles=10000, seed=42)
assignment = assign_gin_clusters(vectors, reference=reference)

for record in assignment.records():
    print(record["name"], record["cluster"], record["description"])
```

Two things to keep consistent, or the labels are meaningless:

- **the same background.** Rebuild it from the same yeast IDR set (the notebook shows how), or save it alongside with `background.save(...)`. Scoring against a different background moves the compositional half of every vector.
- **the same backend.** The reference records that it was built with `'nardini'`, and `assign_gin_clusters` refuses vectors computed with a different patterning statistic, since those are different quantities on different scales.

## Choices worth revisiting

This is an example, not a canonical yeast IDRome analysis. Three decisions are deliberately simple and worth changing for real work:

- **Every predicted IDR is kept**, including very short ones. Published IDRome analyses typically impose a minimum length (30 residues is common), and short IDRs give noisy compositional features. One line filters them.
- **`k=30`** mirrors the human analysis. There is nothing special about 30 for yeast — cluster at several values and look at where the structure actually is.
- **Cluster numbers are arbitrary.** k-means has no canonical ordering; sparrow sorts largest-first so numbering is stable within a run, but it means nothing across runs or against the human clusters. Compare grammars with `compare_gin_references()`, never by cluster number.

The cluster descriptions (`'high Frac Q, Q patches'`) are generated from each centroid's most extreme features. They are not the published human-authored labels, which clustering cannot regenerate.

## Documentation

The full guide is in the sparrow docs under [NARDINI / NARDINI+](https://idptools-sparrow.readthedocs.io/en/latest/api_guides/nardini.html); see "Building your own clusters" for the API this notebook uses, and `sparrow/nardini/architecture.md` if you are working on the module itself.

## Citation

sparrow is an independent implementation, not the source of the method. Please cite:

- **NARDINI** — Cohan, M. C., Shinn, M. K., Lalmansingh, J. M. & Pappu, R. V. Uncovering non-random binary patterns within sequences of intrinsically disordered proteins. *J. Mol. Biol.* (2022).
- **GIN / NARDINI+** — Ruff, K. M. *et al.* Molecular grammars of predicted intrinsically disordered regions that span the human proteome. *Cell* (2025).
- **metapredict** — Emenecker, R. J., Griffith, D. & Holehouse, A. S. Metapredict: a fast, accurate, and easy-to-use predictor of consensus disorder and structure. *Biophys. J.* (2021).
