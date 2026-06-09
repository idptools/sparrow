# AAindex property databases

This directory bundles three JSON databases of amino-acid properties, together
with a lightweight reader (`aaindex_loader.py`). They are derived from the
[AAindex database](https://www.genome.jp/aaindex/) (Kawashima & Kanehisa, *Nucleic
Acids Res.* 2000) plus a handful of additional, more recent scales/matrices that
have been folded into the same format (see [Added entries](#added-entries)).

| File | Kind | Entries | Look-up |
|------|------|--------:|---------|
| `aaindex1.json` | single-residue indices (one value per amino acid) | 569 | `values[aa]` |
| `aaindex2.json` | pairwise substitution / mutation matrices | 105 | `matrix[i][j]` |
| `aaindex3.json` | pairwise contact / statistical potentials | 50 | `matrix[i][j]` |

Every entry retains its `description` and literature `reference` so that
downstream code (e.g. sparrow linear-profile docstrings) can cite the source of
each scale.

---

## Quick start

```python
from sparrow.data.properties import load        # or: aaindex_loader as ai

# --- single-residue indices (aaindex1) ---
db = load("aaindex1")
db.value("KYTJ820101", "W")          # -0.9   (Kyte-Doolittle hydropathy of Trp)
db.values("KYTJ820101")              # {"A": 1.8, "R": -4.5, ...}
db.metadata("KYTJ820101")            # {description, reference, authors, title, journal}
db.list("stickiness")                # [(accession, description), ...] substring search

# --- pairwise matrices (aaindex2 / aaindex3) ---
mats = load("aaindex3")
mats.pair("VILL220102", "W", "W")    # 3.6096   (unified potential, Trp-Trp)
mats.matrix("EDSSMat62")             # nested {i: {j: value}}  (from load("aaindex2"))
mats.is_symmetric("VILL220103")      # False   (this entry is asymmetric -> order matters)
```

Accession look-ups are case-insensitive (`"edssmat62"` == `"EDSSMat62"`). The
loader is standard-library only and resolves the JSON files relative to its own
location, so it also works as a stand-alone helper.

> There is also a higher-level, slug-based accessor for the single-residue
> database in `sparrow/data/aaindex.py` (e.g. `get_property_values("hydropathy-kyte-1982")`).
> Use that when you want human-readable identifiers; use this loader for direct
> access by accession and for the pairwise matrices.

---

## Entry schema

### `aaindex1.json` — single-residue indices

Keyed by accession; each entry:

```jsonc
{
  "accession":    "ANDN920101",
  "description":  "alpha-CH chemical shifts (Andersen et al., 1992)",
  "reference":    "PMID:1575719",
  "authors":      "Andersen, N.H., Cao, B. and Chen, C.",
  "title":        "...",
  "journal":      "Biochem. and Biophys. Res. Comm. 184, 1008-1014 (1992)",
  "correlations": {"BUNA790102": 0.949},   // correlated AAindex1 entries
  "values":       {"A": 4.35, "R": 4.38, ..., "V": 3.95}   // 20 residues
}
```

* `values` — maps each of the 20 standard amino acids to a number, or `null`
  where the source database leaves it undefined.
* `values_sd` — *(optional)* per-residue standard deviations, present on entries
  that ship with uncertainties (currently `VILL220101`).

### `aaindex2.json` / `aaindex3.json` — pairwise matrices

Same metadata fields, plus the matrix block:

```jsonc
{
  "accession":   "TANS760101",
  "description": "Statistical contact potential ...",
  "reference":   "PMID:1004017",
  "authors":     "...", "title": "...", "journal": "...",
  "comment":     null,                     // free-text "*" note, or null
  "rows":        "ARNDCQEGHILKMFPSTWYV",    // row residue labels
  "cols":        "ARNDCQEGHILKMFPSTWYV",    // column residue labels
  "symmetric":   true,                      // see note below
  "matrix":      {"A": {"A": -2.6, "R": -3.4, ...}, ...}   // matrix[i][j]
}
```

* `matrix` is a nested mapping; look up an ordered pair as `matrix[i][j]`
  (row `i`, column `j`). `null` marks values undefined in the source.
* `symmetric` records whether the source stored the matrix in symmetric form.
  When `true`, `matrix[i][j] == matrix[j][i]` (both directions are populated).
  When `false` the matrix is **asymmetric** and order matters (e.g.
  `VILL220103`).
* `matrix_sd` — *(optional)* a companion standard-deviation matrix (currently on
  `VILL220102`).

**Amino-acid order.** Unless an entry's `rows`/`cols` say otherwise, residues use
the canonical AAindex order `ARNDCQEGHILKMFPSTWYV`. A few AAindex2 entries carry
extended labels (e.g. a `-` gap column); always read `rows`/`cols` if you iterate
positionally rather than by residue letter.

---

## Added entries

Beyond the standard AAindex release, the following scales/matrices have been
added in the same format:

### Villegas–Levy unified statistical potential (2022)
*"A unified statistical potential reveals that amino acid stickiness governs
nonspecific recruitment of client proteins into condensates", Protein Science
31:e4361. PMID:35762716.*

| Accession | File | What |
|-----------|------|------|
| `VILL220101` | aaindex1 | 1D "stickiness" (Voronoi interface propensity); has `values_sd` |
| `VILL220102` | aaindex3 | pairwise interaction incl. desolvation of **both** residues (symmetric); has `matrix_sd` |
| `VILL220103` | aaindex3 | pairwise interaction incl. desolvation of **one** residue (**asymmetric**) |
| `VILL220104` | aaindex3 | pairwise contact propensity, **no** desolvation (symmetric) |

### Cao–Tesei–Lindorff-Larsen stickiness scales (2026)
*"A stickiness scale for disordered proteins", J. Phys. Chem. B.
DOI:10.1021/acs.jpcb.6c00592.* CALVADOS λ values, ~0 (Glu) → ~1 (Trp).

| Accession | File | What |
|-----------|------|------|
| `CAOF260101` | aaindex1 | λ with residue-specific diameters (σ_AA) |
| `CAOF260102` | aaindex1 | λ with a shared diameter ⟨σ⟩ = 0.56 nm |

### IDR-specific substitution matrices
Integer log-odds substitution matrices for intrinsically disordered regions
(in `aaindex2.json`):

| Accessions | Reference |
|------------|-----------|
| `EDSSMat50/60/62/70/75/80/90` | Trivedi & Nagarajaram (2019), *Sci. Rep.* 9:16380. PMID:31705028 |
| `DUNMat` | Radivojac et al. (2002), *Pac. Symp. Biocomput.* 589–600. PMID:11928508 |
| `Disorder40/60/85` | Brown et al. (2010), *Mol. Biol. Evol.* 27:609–621. PMID:19923193 |

---

## Reading the files directly

The JSON is plain and can be read without sparrow:

```python
import json
with open("aaindex1.json") as fh:
    aaindex1 = json.load(fh)

aaindex1["KYTJ820101"]["values"]["W"]     # -0.9
```

---

## Provenance / regeneration

The JSON files are generated from the raw AAindex flat files and the added
sources by the scripts under `tools/sparrow/aaindex/` (`parse_aaindex.py`,
`add_stickiness.py`, `add_pairwise.py`, `add_cao_stickiness.py`,
`add_idr_matrices.py`). Those scripts are the source of truth; regenerate and
copy the resulting `aaindex{1,2,3}.json` here if the underlying data changes.
