"""Refit the GIN cluster centroids to sparrow's corrected NARDINI windows.

The published GIN centroids (Ruff et al. 2025) were fit to NARDINI+ vectors whose
patterning block came from the ``nardini`` package, which scores ``n - b + 2``
windows per sequence: the ``n - b + 1`` complete windows plus one holding only the
last ``b - 1`` residues, still divided by ``b``. sparrow scores the complete windows
only (see ``sparrow/nardini/patterning.py``), so its vectors sit slightly off the
published ones and belong with centroids fit in the same space. This script makes
those centroids, and writes them next to the published ones in
``sparrow/data/nardini``.

The clusters are refit rather than re-derived. The GIN k-means landscape is flat:
rerunning the published k-means call on the published vectors, with the published
random state or another, lands on partitions whose inertia is within 0.2% of the
published one but which agree with it at an adjusted Rand index of only 0.37-0.51.
A fresh clustering would therefore scramble the cluster identities, and with them
the published descriptions. Instead, k-means starts from the published centroids and runs to
convergence on corrected vectors of the same IDRs, so every cluster keeps its
identity and moves only as far as the correction moves its members.

Steps
-----
1. Read the human IDRs from the GIN repository, filtered exactly as there (at
   least 30 residues, none of X, U, Z, J, B, O), in file order.
2. Compute corrected NARDINI+ vectors with sparrow against the published human
   background: 100,000 scrambles for the IDRs of 100-300 residues that define the
   clusters, as in the published protocol, and 10,000 for the rest, which only
   feed the cluster sizes.
3. From the published centroids, alternate nearest-centroid assignment and
   centroid update on the clipped 100-300 residue vectors until no IDR changes
   cluster. That is Lloyd's algorithm, the one the published k-means ran.
4. Assign every IDR to the refit centroids to recount ``num_idrs``.
5. Write the centroids and cluster definitions, and report how many IDRs keep the
   cluster the published vectors and centroids give them.

Usage
-----
    python devtools/scripts/build_corrected_gin_reference.py PATH_TO_GIN_REPOSITORY \\
        --vectors corrected_vectors.npz --processes 14

``PATH_TO_GIN_REPOSITORY`` is a checkout of
https://github.com/kierstenruff/RUFF_KING_Grammars_of_IDRs_using_NARDINI-. The
vectors take about an hour on 14 cores, almost all of it the 100,000-scramble
clustering set; ``--vectors`` saves them on the first run and reuses them after.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

# make sure we import the in-tree sparrow
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import sparrow  # noqa: E402
from sparrow.nardini import (  # noqa: E402
    GIN_CLIP,
    load_gin_reference,
    load_reference_background,
    nardini_vectors,
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "sparrow" / "data" / "nardini"
CENTROID_FILE = OUTPUT_DIR / "gin_centroids_human_corrected_v1.npz"
DEFINITION_FILE = OUTPUT_DIR / "gin_cluster_definitions_human_corrected_v1.csv"

#: Locations inside the GIN repository: the human IDRs, and the published 90-feature
#: z-score vectors (row i belongs to the i-th IDR kept by the filter below).
IDR_FILE = Path("0_Creating_GIN_Clusters/data/idrs_min_length_30/sapiens")
PUBLISHED_VECTOR_FILE = Path("0_Creating_GIN_Clusters/data/sapiens_full_idrome_gte_30_nardini_compositional_RtoK.npy")

#: The GIN repository keeps an IDR with at least this many residues and none of the
#: non-standard ones (0_extract_GIN_clusters_IDRs_100_300, cell 4).
MIN_IDR_LENGTH = 30
NONSTANDARD_RESIDUES = ("X", "U", "Z", "J", "B", "O")

#: The published clusters were fit to the human IDRs of 100-300 residues, inclusive.
CLUSTERING_LENGTH_RANGE = (100, 300)

#: Scrambles per IDR. The published vectors used 100,000. The IDRs outside the
#: clustering set only feed the cluster sizes, for which 10,000 is ample.
CLUSTERING_NUM_SCRAMBLES = 100000
OTHER_NUM_SCRAMBLES = 10000

#: Master seed for nardini_vectors; each IDR's own seed derives from it and the sequence.
SEED = 42

#: Lloyd iterations allowed before giving up. The refit converges in a handful.
MAX_ITERATIONS = 300

SOURCE_REPO = "https://github.com/kierstenruff/RUFF_KING_Grammars_of_IDRs_using_NARDINI-"
CITATION = (
    "Ruff, K. M., King, M. R., Ying, A. W., Liu, V., Pant, A., Lieberman, W. E., "
    "Shinn, M. K., Su, X., Kadoch, C. & Pappu, R. V. Molecular grammars of "
    "predicted intrinsically disordered regions that span the human proteome. "
    "Cell (2025)."
)


def load_idrs(repository: Path) -> tuple[list[str], list[str]]:
    """Read the human IDRs, filtered and ordered as in the GIN repository.

    Parameters
    ----------
    repository : pathlib.Path
        Root of the GIN repository checkout.

    Returns
    -------
    tuple of list of str
        ``(names, sequences)``, names being ``ACCESSION_IDRNUMBER``.

    Raises
    ------
    SystemExit
        If the names are not unique, since vectors are matched up by name.
    """
    names, sequences = [], []
    with open(repository / IDR_FILE) as handle:
        for line in handle:
            fields = line.split(" ")
            sequence = fields[3].strip("\n")
            if len(sequence) < MIN_IDR_LENGTH:
                continue
            if any(residue in sequence for residue in NONSTANDARD_RESIDUES):
                continue
            names.append(fields[0])
            sequences.append(sequence)

    if len(set(names)) != len(names):
        raise SystemExit(f"{IDR_FILE} has duplicate IDR names, which this script relies on being unique")
    return names, sequences


def clustering_mask(sequences: list[str]) -> np.ndarray:
    """Which IDRs fall in the length range the published clusters were fit to.

    Parameters
    ----------
    sequences : list of str
        All IDRs.

    Returns
    -------
    numpy.ndarray
        Boolean mask, one entry per IDR.
    """
    lengths = np.array([len(sequence) for sequence in sequences])
    low, high = CLUSTERING_LENGTH_RANGE
    return (lengths >= low) & (lengths <= high)


def planned_scrambles(in_clustering_set: np.ndarray) -> np.ndarray:
    """Scrambles to use for each IDR.

    Parameters
    ----------
    in_clustering_set : numpy.ndarray
        Boolean mask from :func:`clustering_mask`.

    Returns
    -------
    numpy.ndarray
        Integer scramble count per IDR.
    """
    return np.where(in_clustering_set, CLUSTERING_NUM_SCRAMBLES, OTHER_NUM_SCRAMBLES)


def compute_vectors(names: list[str], sequences: list[str], num_scrambles: np.ndarray, processes: int) -> np.ndarray:
    """Corrected NARDINI+ vectors against the published human background.

    Parameters
    ----------
    names, sequences : list of str
        All IDRs.
    num_scrambles : numpy.ndarray
        Scrambles per IDR, from :func:`planned_scrambles`.
    processes : int
        Worker processes for :func:`sparrow.nardini.nardini_vectors`.

    Returns
    -------
    numpy.ndarray
        ``(len(names), 90)`` z-score vectors, in input order.
    """
    background = load_reference_background("human")
    vectors = np.full((len(names), 90), np.nan)

    for count in np.unique(num_scrambles):
        rows = np.flatnonzero(num_scrambles == count)
        print(f"computing {rows.size} vectors at {count} scrambles", flush=True)
        result = nardini_vectors(
            {names[row]: sequences[row] for row in rows},
            background=background,
            num_scrambles=int(count),
            seed=SEED,
            n_processes=processes,
            show_progress=True,
        )
        vectors[rows] = result.zscores

    assert np.all(np.isfinite(vectors)), "every IDR must have a finite vector"
    return vectors


def load_or_compute_vectors(
    cache: Path | None,
    names: list[str],
    sequences: list[str],
    num_scrambles: np.ndarray,
    processes: int,
) -> np.ndarray:
    """Reuse saved vectors when they were made with exactly these settings.

    Parameters
    ----------
    cache : pathlib.Path or None
        Where vectors are saved. None computes them without saving.
    names, sequences : list of str
        All IDRs.
    num_scrambles : numpy.ndarray
        Scrambles per IDR.
    processes : int
        Worker processes, if the vectors have to be computed.

    Returns
    -------
    numpy.ndarray
        ``(len(names), 90)`` z-score vectors.

    Raises
    ------
    SystemExit
        If the cache was made for different IDRs, scrambles or seed.
    """
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if list(data["names"]) != names:
                raise SystemExit(f"{cache} holds vectors for different IDRs")
            if not np.array_equal(data["num_scrambles"], num_scrambles) or int(data["seed"]) != SEED:
                raise SystemExit(f"{cache} was computed with different scrambles or a different seed")
            print(f"reusing the vectors saved in {cache}")
            return np.asarray(data["zscores"], dtype=np.float64)

    vectors = compute_vectors(names, sequences, num_scrambles, processes)
    if cache is not None:
        np.savez_compressed(
            cache,
            names=np.asarray(names),
            zscores=vectors,
            num_scrambles=num_scrambles,
            seed=np.asarray(SEED),
        )
        print(f"saved the vectors to {cache}")
    return vectors


def nearest_centroid(clipped: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Index of the nearest centroid by squared Euclidean distance.

    Parameters
    ----------
    clipped : numpy.ndarray
        ``(n, 90)`` vectors, already clipped.
    centroids : numpy.ndarray
        ``(k, 90)`` centroids.

    Returns
    -------
    numpy.ndarray
        Cluster index per vector.
    """
    # one centroid at a time: the whole IDRome against all 30 at once would be a
    # half-gigabyte temporary
    distances = np.empty((clipped.shape[0], centroids.shape[0]))
    for index, centroid in enumerate(centroids):
        distances[:, index] = ((clipped - centroid) ** 2).sum(axis=1)
    return np.argmin(distances, axis=1)


def cluster_means(clipped: np.ndarray, labels: np.ndarray, num_clusters: int) -> np.ndarray:
    """Mean of each cluster's members.

    Parameters
    ----------
    clipped : numpy.ndarray
        ``(n, 90)`` clipped vectors.
    labels : numpy.ndarray
        Cluster index per vector.
    num_clusters : int
        Number of clusters.

    Returns
    -------
    numpy.ndarray
        ``(num_clusters, 90)`` means.

    Raises
    ------
    SystemExit
        If a cluster has emptied, which would leave its centroid undefined.
    """
    sizes = np.bincount(labels, minlength=num_clusters)
    if np.any(sizes == 0):
        raise SystemExit(f"clusters {np.flatnonzero(sizes == 0).tolist()} lost every member during the refit")
    return np.array([clipped[labels == cluster].mean(axis=0) for cluster in range(num_clusters)])


def refit(clipped: np.ndarray, initial_centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Lloyd's algorithm from given centroids, run until no vector changes cluster.

    Parameters
    ----------
    clipped : numpy.ndarray
        ``(n, 90)`` clipped vectors.
    initial_centroids : numpy.ndarray
        ``(k, 90)`` starting centroids.

    Returns
    -------
    tuple
        ``(labels, centroids, iterations)``. At convergence every vector is nearest
        its own centroid and every centroid is the mean of its members.

    Raises
    ------
    SystemExit
        If it has not converged after :data:`MAX_ITERATIONS` updates.
    """
    num_clusters = initial_centroids.shape[0]
    labels = nearest_centroid(clipped, initial_centroids)

    for iteration in range(1, MAX_ITERATIONS + 1):
        centroids = cluster_means(clipped, labels, num_clusters)
        updated = nearest_centroid(clipped, centroids)
        if np.array_equal(updated, labels):
            return labels, centroids, iteration
        labels = updated

    raise SystemExit(f"the refit did not converge in {MAX_ITERATIONS} iterations")


def retained_per_cluster(before: np.ndarray, after: np.ndarray, num_clusters: int) -> np.ndarray:
    """Fraction of each cluster's original members that are still in it.

    Parameters
    ----------
    before, after : numpy.ndarray
        Cluster index per IDR, before and after.
    num_clusters : int
        Number of clusters.

    Returns
    -------
    numpy.ndarray
        One fraction per cluster.
    """
    kept = np.zeros(num_clusters)
    for cluster in range(num_clusters):
        members = before == cluster
        kept[cluster] = np.mean(after[members] == cluster)
    return kept


def report(label: str, before: np.ndarray, after: np.ndarray, num_clusters: int) -> float:
    """Print how many IDRs keep their published cluster, overall and per cluster.

    Parameters
    ----------
    label : str
        What is being compared, for the printout.
    before, after : numpy.ndarray
        Published and corrected cluster index per IDR.
    num_clusters : int
        Number of clusters.

    Returns
    -------
    float
        Fraction of IDRs that keep their published cluster.
    """
    kept = float(np.mean(before == after))
    per_cluster = retained_per_cluster(before, after, num_clusters)
    print(
        f"{label}: {np.sum(before != after)} of {before.size} IDRs change cluster ({100 * (1 - kept):.2f}%); "
        f"per-cluster retention min {per_cluster.min():.3f} (cluster {int(np.argmin(per_cluster))}), "
        f"median {np.median(per_cluster):.3f}"
    )
    return kept


def write_reference(
    centroids: np.ndarray,
    feature_names: tuple[str, ...],
    descriptions: dict[int, str],
    num_idrs: np.ndarray,
    metadata: dict[str, object],
) -> None:
    """Write the centroids and definitions in the same format as the published ones.

    Parameters
    ----------
    centroids : numpy.ndarray
        ``(30, 90)`` refit centroids.
    feature_names : tuple of str
        The 90 feature names, in column order.
    descriptions : dict
        Published description of each cluster.
    num_idrs : numpy.ndarray
        Human IDRs assigned to each cluster.
    metadata : dict
        Provenance, stored as JSON alongside the centroids.
    """
    width = max(len(name) for name in feature_names)
    np.savez_compressed(
        CENTROID_FILE,
        feature_names=np.asarray(feature_names, dtype=f"<U{width}"),
        clusters=np.arange(centroids.shape[0], dtype=np.int32),
        centroids=centroids,
        format_version=np.asarray(1),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    print(f"wrote {CENTROID_FILE}")

    with open(DEFINITION_FILE, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cluster", "description", "num_idrs"])
        for cluster in range(centroids.shape[0]):
            writer.writerow([cluster, descriptions[cluster], int(num_idrs[cluster])])
    print(f"wrote {DEFINITION_FILE}")


def main() -> None:
    """Refit the centroids, recount the clusters, and write both files."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("repository", type=Path, help="checkout of the GIN repository")
    parser.add_argument("--vectors", type=Path, default=None, help="save (or reuse) the computed vectors here")
    parser.add_argument("--processes", type=int, default=os.cpu_count() or 1, help="worker processes")
    args = parser.parse_args()

    names, sequences = load_idrs(args.repository)
    in_clustering_set = clustering_mask(sequences)
    num_scrambles = planned_scrambles(in_clustering_set)
    print(f"{len(names)} human IDRs, {int(in_clustering_set.sum())} of them in the clustering set")

    published = load_gin_reference("published")
    num_clusters = len(published)

    vectors = load_or_compute_vectors(args.vectors, names, sequences, num_scrambles, args.processes)
    clipped = np.clip(vectors, -GIN_CLIP, GIN_CLIP)

    labels, centroids, iterations = refit(clipped[in_clustering_set], published.centroids)
    print(f"refit converged in {iterations} iterations")

    # what the published vectors and centroids say, to measure how much moved
    published_vectors = np.load(args.repository / PUBLISHED_VECTOR_FILE, allow_pickle=False)
    if published_vectors.shape != vectors.shape:
        raise SystemExit(f"{PUBLISHED_VECTOR_FILE} has shape {published_vectors.shape}, expected {vectors.shape}")
    published_calls = nearest_centroid(np.clip(published_vectors, -GIN_CLIP, GIN_CLIP), published.centroids)

    corrected_calls = nearest_centroid(clipped, centroids)
    assert np.array_equal(corrected_calls[in_clustering_set], labels), "assignment must reproduce the refit labels"

    clustering_kept = report("clustering set", published_calls[in_clustering_set], labels, num_clusters)
    idrome_kept = report("whole IDRome", published_calls, corrected_calls, num_clusters)

    shift = ((centroids - published.centroids) ** 2).sum(axis=1)
    print(
        f"centroid shift (squared distance): median {np.median(shift):.4f}, max {shift.max():.4f}; "
        f"published centroid spacing {published.separation_scale:.2f}"
    )

    num_idrs = np.bincount(corrected_calls, minlength=num_clusters)
    metadata = {
        "species": "Homo sapiens",
        "description": (
            "The published GIN clusters refit to NARDINI+ vectors computed with the "
            "n - b + 1 complete patterning windows. k-means was started from the "
            "published centroids and run to convergence, so cluster numbers and "
            "descriptions are the published ones."
        ),
        "window_convention": "n - b + 1 complete windows",
        "clip": float(GIN_CLIP),
        "initial_centroids": "published GIN centroids (gin_centroids_human_v1.npz)",
        "background": "published human compositional prior (gin_composition_prior_human_v1.npz)",
        "clustering_length_range": list(CLUSTERING_LENGTH_RANGE),
        "num_clustering_idrs": int(in_clustering_set.sum()),
        "num_idrs_total": len(names),
        "clustering_num_scrambles": CLUSTERING_NUM_SCRAMBLES,
        "other_num_scrambles": OTHER_NUM_SCRAMBLES,
        "seed": SEED,
        "iterations": iterations,
        "fraction_clustering_idrs_keeping_published_cluster": clustering_kept,
        "fraction_idrs_keeping_published_cluster": idrome_kept,
        "sparrow_version": sparrow.__version__,
        "build_script": "devtools/scripts/build_corrected_gin_reference.py",
        "source_repository": SOURCE_REPO,
        "citation": CITATION,
        "licence": "MIT",
    }

    write_reference(centroids, published.feature_names, published.descriptions, num_idrs, metadata)


if __name__ == "__main__":
    main()
