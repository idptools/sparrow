"""Regenerate the bundled NARDINI/GIN reference data in sparrow/data/nardini.

The reference data are published alongside the GIN resource (Ruff et al. 2025)
and are distributed under the MIT licence from
https://github.com/kierstenruff/RUFF_KING_Grammars_of_IDRs_using_NARDINI-

This script downloads them from their canonical locations, checks that the
feature ordering matches sparrow's own NARDINI feature definitions, and writes
the compact archives that ``sparrow.nardini`` loads at runtime. It is not needed
to *use* sparrow - it exists so the bundled data can be audited and rebuilt.

Usage
-----
    python devtools/scripts/build_nardini_reference_data.py

Requires network access, pandas and requests-free ``urllib``.
"""

import csv
import io
import json
import os
import sys
import urllib.request

import numpy as np

# make sure we import the in-tree sparrow
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sparrow.nardini.composition import COMPOSITION_FEATURE_NAMES  # noqa: E402
from sparrow.nardini.groups import patterning_feature_names  # noqa: E402

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "sparrow", "data", "nardini"
)

# Google Sheet holding the per-species compositional priors and GIN centroids
PRIOR_SHEET_ID = "1yxt0R1G0gdI2bGpjYXgk_h7-1qA6EY6J"
# Google Sheet holding the human GIN cluster definitions
CLUSTER_DEF_SHEET_ID = "1_ujBtGWrxEtWHV3UmMDOXXzOx04wk0aReSPk9t28yCE"

# worksheet name -> canonical species key used by sparrow
SPECIES_WORKSHEETS = {
    "Sapiens": ("human", "Homo sapiens"),
    "Musculus": ("mouse", "Mus musculus"),
    "norvegicus": ("rat", "Rattus norvegicus"),
    "tropicalis": ("xenopus", "Xenopus tropicalis"),
    "melanogaster": ("fly", "Drosophila melanogaster"),
    "rerio": ("zebrafish", "Danio rerio"),
    "cerevisiae": ("yeast", "Saccharomyces cerevisiae"),
    "elegans": ("worm", "Caenorhabditis elegans"),
    "thaliana": ("arabidopsis", "Arabidopsis thaliana"),
}

CITATION = (
    "Ruff, K. M., King, M. R., Ying, A. W., Liu, V., Pant, A., Lieberman, W. E., "
    "Shinn, M. K., Su, X., Kadoch, C. & Pappu, R. V. Molecular grammars of "
    "predicted intrinsically disordered regions that span the human proteome. "
    "Cell (2025)."
)

SOURCE_REPO = (
    "https://github.com/kierstenruff/RUFF_KING_Grammars_of_IDRs_using_NARDINI-"
)


def _sheet_csv(sheet_id, worksheet):
    url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        f"/gviz/tq?tqx=out:csv&sheet={worksheet}"
    )
    with urllib.request.urlopen(url, timeout=120) as handle:
        text = handle.read().decode("utf-8")
    return list(csv.reader(io.StringIO(text))), url


def build_priors():
    """Write one NardiniBackground-format archive per species."""
    expected = list(COMPOSITION_FEATURE_NAMES)

    for worksheet, (key, latin) in SPECIES_WORKSHEETS.items():
        rows, url = _sheet_csv(PRIOR_SHEET_ID, worksheet)
        header = rows[0]
        feature_col = header.index("Feature")
        mean_col = header.index("Mean")
        std_col = header.index("Std")

        names, means, stds = [], [], []
        for row in rows[1:]:
            if not row or not row[feature_col]:
                continue
            names.append(row[feature_col])
            means.append(float(row[mean_col]))
            stds.append(float(row[std_col]))

        if names != expected:
            raise SystemExit(
                f"{worksheet}: feature order does not match sparrow's "
                f"COMPOSITION_FEATURE_NAMES ({len(names)} vs {len(expected)})"
            )

        metadata = {
            "species": latin,
            "species_key": key,
            "source": url,
            "source_repository": SOURCE_REPO,
            "citation": CITATION,
            "licence": "MIT",
            "description": (
                f"Published NARDINI+ compositional background for the {latin} "
                "IDRome (IDRs >= 30 residues)."
            ),
            "num_sequences_known": False,
        }

        out = os.path.join(OUTPUT_DIR, f"gin_composition_prior_{key}_v1.npz")
        max_len = max(len(n) for n in names)
        np.savez_compressed(
            out,
            feature_names=np.asarray(names, dtype=f"<U{max_len}"),
            mean=np.asarray(means, dtype=np.float64),
            std=np.asarray(stds, dtype=np.float64),
            # the published tables do not report how many IDRs went into them
            num_sequences=np.asarray(-1),
            format_version=np.asarray(1),
            metadata_json=np.asarray(json.dumps(metadata)),
        )
        print(f"wrote {os.path.basename(out)}  ({len(names)} features)")


def build_centroids():
    """Write the 30 x 90 human GIN centroid matrix."""
    rows, url = _sheet_csv(PRIOR_SHEET_ID, "GIN_centroids")
    header = rows[0]

    # first column is the (unnamed) cluster index
    feature_names = [h for h in header[1:] if h != ""]
    expected = list(patterning_feature_names("8")) + list(COMPOSITION_FEATURE_NAMES)
    if feature_names != expected:
        raise SystemExit(
            "GIN_centroids: feature order does not match sparrow's NARDINI+ "
            f"feature order ({len(feature_names)} vs {len(expected)})"
        )

    clusters, matrix = [], []
    for row in rows[1:]:
        if not row or row[0] == "":
            continue
        clusters.append(int(row[0]))
        matrix.append([float(x) for x in row[1 : 1 + len(feature_names)]])

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (30, 90):
        raise SystemExit(f"expected a 30 x 90 centroid matrix, got {matrix.shape}")

    metadata = {
        "source": url,
        "source_repository": SOURCE_REPO,
        "citation": CITATION,
        "licence": "MIT",
        "species": "Homo sapiens",
        "description": (
            "k-means centroids (k=30) of NARDINI+ z-score vectors for human IDRs "
            "of length 100-300, clipped to +/-3. Defines the GIN clusters."
        ),
        "clip": 3.0,
    }

    out = os.path.join(OUTPUT_DIR, "gin_centroids_human_v1.npz")
    max_len = max(len(n) for n in feature_names)
    np.savez_compressed(
        out,
        feature_names=np.asarray(feature_names, dtype=f"<U{max_len}"),
        clusters=np.asarray(clusters, dtype=np.int32),
        centroids=matrix,
        format_version=np.asarray(1),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    print(f"wrote {os.path.basename(out)}  ({matrix.shape[0]} clusters)")


def build_cluster_definitions():
    """Write the human GIN cluster descriptions as a small CSV."""
    rows, url = _sheet_csv(CLUSTER_DEF_SHEET_ID, "Homo_sapiens_cluster_definitions")
    header = rows[0]

    cluster_col = header.index("Cluster")
    desc_col = header.index("Description")
    count_col = header.index("Number of IDRs")

    out = os.path.join(OUTPUT_DIR, "gin_cluster_definitions_human_v1.csv")
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cluster", "description", "num_idrs"])
        for row in rows[1:]:
            if not row or row[cluster_col] == "":
                continue
            writer.writerow([int(row[cluster_col]), row[desc_col], int(row[count_col])])

    print(f"wrote {os.path.basename(out)}  (from {url})")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    build_priors()
    build_centroids()
    build_cluster_definitions()
    print("\nDone. Remember to update sparrow/data/nardini/SOURCES.md if the "
          "upstream data changed.")
