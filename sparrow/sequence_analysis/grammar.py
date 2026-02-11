"""Core grammar-style feature extraction for a single sequence.

This module implements sequence-to-feature-vector workflows using Sparrow-native
primitives and optional scramble/statistics-based z-scoring.
"""

import math
import random
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy import stats

from sparrow.data import amino_acids
from sparrow.protein import Protein
from sparrow.tools import general_tools

AMINO_ACIDS = tuple(amino_acids.VALID_AMINO_ACIDS)

PATTERN_GROUPS = OrderedDict(
    [
        ("pol", tuple("STNQCH")),
        ("hyd", tuple("ILMV")),
        ("pos", tuple("RK")),
        ("neg", tuple("ED")),
        ("aro", tuple("FWY")),
        ("ala", ("A",)),
        ("pro", ("P",)),
        ("gly", ("G",)),
    ]
)
PATTERN_NAMES = tuple(PATTERN_GROUPS.keys())

COMPOSITION_FEATURES = tuple(
    [f"Frac {aa}" for aa in AMINO_ACIDS]
    + [
        "Frac K+R",
        "Frac D+E",
        "Frac Polar",
        "Frac Aliphatic",
        "Frac Aromatic",
        "R/K Ratio",
        "E/D Ratio",
        "FCR",
        "NCPR",
        "Hydrophobicity",
    ]
)

PATCH_FEATURES = (
    "A Patch",
    "C Patch",
    "D Patch",
    "E Patch",
    "F Patch",
    "G Patch",
    "H Patch",
    "I Patch",
    "K Patch",
    "L Patch",
    "M Patch",
    "N Patch",
    "P Patch",
    "Q Patch",
    "R Patch",
    "S Patch",
    "T Patch",
    "V Patch",
    "Y Patch",
    "RG Frac",
)

PATTERN_FEATURES_KAPPA = tuple(
    f"kappa::{PATTERN_NAMES[i]}-{PATTERN_NAMES[j]}"
    for i in range(len(PATTERN_NAMES))
    for j in range(i, len(PATTERN_NAMES))
)
PATTERN_FEATURES_IWD = tuple(
    f"iwd::{PATTERN_NAMES[i]}-{PATTERN_NAMES[j]}"
    for i in range(len(PATTERN_NAMES))
    for j in range(i, len(PATTERN_NAMES))
)

DEFAULT_COMPOSITION_BACKGROUND_FILENAME = (
    "human_idrs_new_grammar_composition_background_f32.npz"
)
DEFAULT_COMPOSITION_BACKGROUND_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / DEFAULT_COMPOSITION_BACKGROUND_FILENAME
)
_DEFAULT_COMPOSITION_STATS_CACHE = None


class GrammarException(RuntimeError):
    """Raised when grammar feature computation fails."""


@dataclass(frozen=True)
class GrammarCompositionStats:
    feature_names: Sequence[str]
    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class GrammarPatterningConfig:
    backend: str = "kappa_cython"
    num_scrambles: int = 10000
    blob_size: int = 5
    min_fraction: float = 0.10
    seed: Optional[int] = None
    fit_method: str = "gamma_mle"  # "gamma_mle" or "moments"

    def __post_init__(self):
        if self.backend not in ("kappa_cython", "iwd_combined"):
            raise GrammarException(
                f"Unknown backend={self.backend!r}; expected 'kappa_cython' or 'iwd_combined'"
            )
        if self.num_scrambles < 1:
            raise GrammarException("num_scrambles must be >= 1")
        if self.blob_size < 2:
            raise GrammarException("blob_size must be >= 2")
        if self.min_fraction < 0 or self.min_fraction > 1:
            raise GrammarException("min_fraction must be between 0 and 1")
        if self.fit_method not in ("gamma_mle", "moments"):
            raise GrammarException(
                f"Unknown fit_method={self.fit_method!r}; expected 'gamma_mle' or 'moments'"
            )

    def rng(self):
        return random.Random(self.seed)


def _coerce_protein(sequence_or_protein):
    if isinstance(sequence_or_protein, Protein):
        protein = sequence_or_protein
    elif isinstance(sequence_or_protein, str):
        protein = Protein(sequence_or_protein)
    else:
        raise GrammarException("Input must be a sequence string or a sparrow.Protein")

    general_tools.validate_protein_sequence(
        protein.sequence,
        allow_empty=False,
        uppercase=False,
        exception_cls=GrammarException,
        sequence_name="sequence",
    )
    return protein


def _resolve_patterning_config(
    patterning_config=None,
    backend=None,
    num_scrambles=None,
    blob_size=None,
    min_fraction=None,
    seed=None,
    fit_method=None,
):
    """Build effective patterning config from defaults, config object, and overrides."""
    config = patterning_config or GrammarPatterningConfig()
    overrides = {}
    if backend is not None:
        overrides["backend"] = backend
    if num_scrambles is not None:
        overrides["num_scrambles"] = num_scrambles
    if blob_size is not None:
        overrides["blob_size"] = blob_size
    if min_fraction is not None:
        overrides["min_fraction"] = min_fraction
    if seed is not None:
        overrides["seed"] = seed
    if fit_method is not None:
        overrides["fit_method"] = fit_method
    if not overrides:
        return config
    return replace(config, **overrides)


def _pattern_feature_name(backend, name1, name2):
    if backend == "kappa_cython":
        return f"kappa::{name1}-{name2}"
    return f"iwd::{name1}-{name2}"


def pattern_feature_names(backend):
    """Return ordered patterning feature names for the selected backend."""
    if backend == "kappa_cython":
        return PATTERN_FEATURES_KAPPA
    if backend == "iwd_combined":
        return PATTERN_FEATURES_IWD
    raise GrammarException(f"Unknown backend={backend!r}")


def composition_feature_names():
    """Return ordered composition feature names."""
    return COMPOSITION_FEATURES


def patch_feature_names():
    """Return ordered patch feature names."""
    return PATCH_FEATURES


def compute_composition_raw(sequence_or_protein):
    """Compute Sparrow-native composition + patch features."""
    protein = _coerce_protein(sequence_or_protein)
    aa_fracs = protein.amino_acid_fractions
    seq_length = len(protein.sequence)

    features = OrderedDict()
    for aa in AMINO_ACIDS:
        features[f"Frac {aa}"] = float(aa_fracs[aa])

    features["Frac K+R"] = float(aa_fracs["K"] + aa_fracs["R"])
    features["Frac D+E"] = float(aa_fracs["D"] + aa_fracs["E"])
    features["Frac Polar"] = float(sum(aa_fracs[x] for x in "QNSTGCH"))
    features["Frac Aliphatic"] = float(sum(aa_fracs[x] for x in "ALMIV"))
    features["Frac Aromatic"] = float(sum(aa_fracs[x] for x in "FWY"))

    features["R/K Ratio"] = math.log10(
        ((seq_length * aa_fracs["R"]) + 1) / ((seq_length * aa_fracs["K"]) + 1)
    )
    features["E/D Ratio"] = math.log10(
        ((seq_length * aa_fracs["E"]) + 1) / ((seq_length * aa_fracs["D"]) + 1)
    )

    features["FCR"] = float(protein.FCR)
    features["NCPR"] = float(protein.NCPR)
    features["Hydrophobicity"] = float(protein.hydrophobicity)

    for feature_name in PATCH_FEATURES:
        if feature_name == "RG Frac":
            continue
        residue = feature_name.split()[0]
        features[feature_name] = float(protein.compute_patch_fraction(residue))

    features["RG Frac"] = float(
        protein.compute_patch_fraction(
            residue_selector="RG",
            min_target_count=None,
            adjacent_pair_pattern="RG",
            min_adjacent_pair_count=2,
        )
    )

    return features


def compute_composition_zscores(raw_composition, composition_stats):
    """Compute z-scores for selected composition/patch features."""
    if len(composition_stats.mean) != len(composition_stats.std) or len(
        composition_stats.mean
    ) != len(composition_stats.feature_names):
        raise GrammarException("GrammarCompositionStats mean/std length mismatch")

    out = OrderedDict()
    for name, mean, std in zip(
        composition_stats.feature_names, composition_stats.mean, composition_stats.std
    ):
        value = raw_composition.get(name)
        if value is None:
            out[name] = float("nan")
        elif std == 0:
            out[name] = 0.0
        else:
            out[name] = (value - mean) / std
    return out


def _compute_group_fractions(protein):
    fracs = OrderedDict()
    for group_name, residues in PATTERN_GROUPS.items():
        fracs[group_name] = protein.compute_residue_fractions(residues)
    return fracs


def _compute_pattern_value(protein, backend, name1, name2, blob_size):
    group1 = list(PATTERN_GROUPS[name1])
    group2 = list(PATTERN_GROUPS[name2])

    if backend == "kappa_cython":
        if name1 == name2:
            value = protein.compute_kappa_x(
                group1=group1, group2=None, window_size=blob_size, flatten=True
            )
        else:
            value = protein.compute_kappa_x(
                group1=group1, group2=group2, window_size=blob_size, flatten=True
            )
        if value < 0:
            return 0.0
        return float(value)

    if name1 == name2:
        return float(protein.compute_iwd(group1))

    merged_group = sorted(set(group1 + group2))
    return float(protein.compute_iwd(merged_group))


def compute_patterning_raw(sequence_or_protein, config):
    """Compute ordered patterning features for a sequence."""
    protein = _coerce_protein(sequence_or_protein)
    fractions = _compute_group_fractions(protein)

    values = OrderedDict()
    for i, name1 in enumerate(PATTERN_NAMES):
        for j in range(i, len(PATTERN_NAMES)):
            name2 = PATTERN_NAMES[j]
            feature_name = _pattern_feature_name(config.backend, name1, name2)

            if i == j:
                valid = fractions[name1] > config.min_fraction
            else:
                valid = (
                    fractions[name1] > config.min_fraction
                    and fractions[name2] > config.min_fraction
                )

            if not valid:
                values[feature_name] = 0.0
                continue

            values[feature_name] = _compute_pattern_value(
                protein=protein,
                backend=config.backend,
                name1=name1,
                name2=name2,
                blob_size=config.blob_size,
            )
    return values


def compute_patterning_scramble_distribution(sequence_or_protein, config):
    """Compute scramble distributions for each patterning feature."""
    protein = _coerce_protein(sequence_or_protein)
    base_sequence = protein.sequence
    rng = config.rng()

    raw_template = compute_patterning_raw(protein, config)
    distributions = OrderedDict(
        (name, np.zeros(config.num_scrambles, dtype=np.float64))
        for name in raw_template.keys()
    )

    seq_chars = list(base_sequence)
    for i in range(config.num_scrambles):
        shuffled_chars = seq_chars[:]
        rng.shuffle(shuffled_chars)
        shuffled_sequence = "".join(shuffled_chars)
        shuffled_raw = compute_patterning_raw(shuffled_sequence, config)
        for feature_name in distributions:
            distributions[feature_name][i] = shuffled_raw[feature_name]

    return distributions


def _fit_distribution(values, fit_method):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")

    mean = float(np.mean(values))
    var = float(np.var(values))

    if fit_method == "moments" or var == 0:
        return mean, var

    try:
        alpha, loc, beta = stats.gamma.fit(values)
        gamma_mean = float(stats.gamma.mean(alpha, loc, beta))
        gamma_var = float(stats.gamma.var(alpha, loc, beta))
        return gamma_mean, gamma_var
    except Exception:
        return mean, var


def compute_patterning_zscores(raw_patterning, scramble_distribution, config):
    """Compute patterning z-scores from raw values and scramble distributions."""
    out = OrderedDict()
    for feature_name, raw_value in raw_patterning.items():
        if feature_name not in scramble_distribution:
            raise GrammarException(
                f"Missing scramble distribution for feature {feature_name}"
            )
        mean, var = _fit_distribution(
            scramble_distribution[feature_name], config.fit_method
        )
        if raw_value == 0 or math.isnan(mean) or math.isnan(var) or var == 0:
            out[feature_name] = 0.0
        else:
            out[feature_name] = (raw_value - mean) / math.sqrt(var)
    return out


def merge_feature_blocks(raw_blocks=None, z_blocks=None):
    """Merge raw and z-score blocks into a single ordered feature vector."""
    raw_blocks = raw_blocks or []
    z_blocks = z_blocks or []

    out = OrderedDict()
    for block in raw_blocks:
        for feature_name, value in block.items():
            out[f"raw::{feature_name}"] = float(value)
    for block in z_blocks:
        for feature_name, value in block.items():
            out[f"z::{feature_name}"] = float(value)
    return out


def _finalize_feature_output(
    vector,
    return_array=True,
    return_feature_names=False,
):
    if not return_array:
        return vector

    arr = np.fromiter(vector.values(), dtype=np.float32, count=len(vector))
    if return_feature_names:
        return arr, tuple(vector.keys())
    return arr


def compute_feature_vector(
    sequence_or_protein,
    patterning_config=None,
    composition_stats=None,
    use_default_composition_stats=True,
    include_raw=False,
    return_array=True,
    return_feature_names=False,
    backend=None,
    num_scrambles=None,
    blob_size=None,
    min_fraction=None,
    seed=None,
    fit_method=None,
):
    """Compute an ordered grammar feature vector for one sequence.

    If ``use_default_composition_stats`` is True and ``composition_stats`` is
    None, composition z-scores use Sparrow's built-in human-IDR background.
    Z-score features are always included. Set ``include_raw=True`` to append
    the raw feature block. By default this returns a ``np.float32`` array.

    ``patterning_config`` is optional. Users can override config fields directly
    via keyword arguments like ``num_scrambles`` and ``backend``.
    """
    protein = _coerce_protein(sequence_or_protein)
    config = _resolve_patterning_config(
        patterning_config=patterning_config,
        backend=backend,
        num_scrambles=num_scrambles,
        blob_size=blob_size,
        min_fraction=min_fraction,
        seed=seed,
        fit_method=fit_method,
    )
    if composition_stats is None and use_default_composition_stats:
        composition_stats = load_default_composition_stats()

    raw_patterning = compute_patterning_raw(protein, config)
    raw_composition = None

    raw_blocks = []
    if include_raw:
        raw_composition = compute_composition_raw(protein)
        raw_blocks = [raw_patterning, raw_composition]

    scramble_distribution = compute_patterning_scramble_distribution(protein, config)
    z_blocks = [
        compute_patterning_zscores(raw_patterning, scramble_distribution, config)
    ]
    if composition_stats is not None:
        if raw_composition is None:
            raw_composition = compute_composition_raw(protein)
        z_blocks.append(compute_composition_zscores(raw_composition, composition_stats))

    vector = merge_feature_blocks(raw_blocks=raw_blocks, z_blocks=z_blocks)
    return _finalize_feature_output(
        vector,
        return_array=return_array,
        return_feature_names=return_feature_names,
    )


def compute_composition_background_stats(sequences_or_proteins, dtype=np.float32):
    """Compute composition/patch background stats with low memory use.

    Parameters
    ----------
    sequences_or_proteins : iterable, mapping, str, or Protein
        Sequence collection used to estimate background mean/std. If a mapping
        is passed, values are used.
    dtype : numpy dtype, optional
        Output dtype for stored means/stds. Default ``np.float32``.

    Returns
    -------
    GrammarCompositionStats
        Feature names plus background mean/std arrays.
    """
    if isinstance(sequences_or_proteins, Mapping):
        iterator = iter(sequences_or_proteins.values())
    elif isinstance(sequences_or_proteins, (str, Protein)):
        iterator = iter([sequences_or_proteins])
    else:
        iterator = iter(sequences_or_proteins)

    count = 0
    feature_names = None
    mean = None
    m2 = None

    for entry in iterator:
        raw = compute_composition_raw(entry)
        if feature_names is None:
            feature_names = tuple(raw.keys())
            mean = np.zeros(len(feature_names), dtype=np.float64)
            m2 = np.zeros(len(feature_names), dtype=np.float64)

        values = np.asarray([raw[name] for name in feature_names], dtype=np.float64)

        count += 1
        delta = values - mean
        mean += delta / count
        delta2 = values - mean
        m2 += delta * delta2

    if count == 0:
        raise GrammarException("Cannot compute composition background from empty input")

    # Population variance (ddof=0) matches np.std default semantics.
    variance = m2 / count
    std = np.sqrt(variance)

    out_dtype = np.dtype(dtype)
    return GrammarCompositionStats(
        feature_names=feature_names,
        mean=mean.astype(out_dtype, copy=False),
        std=std.astype(out_dtype, copy=False),
    )


def save_composition_stats_npz(
    output_filename, composition_stats, dtype=np.float32, compressed=True
):
    """Save grammar composition background stats to a compact NumPy archive."""
    if len(composition_stats.mean) != len(composition_stats.std) or len(
        composition_stats.mean
    ) != len(composition_stats.feature_names):
        raise GrammarException("GrammarCompositionStats mean/std length mismatch")

    out_dtype = np.dtype(dtype)
    feature_names = tuple(str(x) for x in composition_stats.feature_names)
    max_name_len = max(len(name) for name in feature_names) if feature_names else 1

    payload = {
        "feature_names": np.asarray(feature_names, dtype=f"<U{max_name_len}"),
        "mean": np.asarray(composition_stats.mean, dtype=out_dtype),
        "std": np.asarray(composition_stats.std, dtype=out_dtype),
    }

    if compressed:
        np.savez_compressed(output_filename, **payload)
    else:
        np.savez(output_filename, **payload)


def load_composition_stats_npz(input_filename):
    """Load grammar composition background stats from NumPy archive."""
    with np.load(input_filename, allow_pickle=False) as data:
        feature_names = tuple(str(x) for x in data["feature_names"].tolist())
        mean = np.asarray(data["mean"])
        std = np.asarray(data["std"])

    return GrammarCompositionStats(feature_names=feature_names, mean=mean, std=std)


def load_default_composition_stats():
    """Load built-in human-IDR composition background stats (cached)."""
    global _DEFAULT_COMPOSITION_STATS_CACHE
    if _DEFAULT_COMPOSITION_STATS_CACHE is None:
        if not DEFAULT_COMPOSITION_BACKGROUND_PATH.exists():
            raise GrammarException(
                f"Default composition background file not found at "
                f"{DEFAULT_COMPOSITION_BACKGROUND_PATH}. "
                "Pass composition_stats explicitly or set "
                "use_default_composition_stats=False."
            )
        _DEFAULT_COMPOSITION_STATS_CACHE = load_composition_stats_npz(
            DEFAULT_COMPOSITION_BACKGROUND_PATH
        )
    return _DEFAULT_COMPOSITION_STATS_CACHE


__all__ = [
    "AMINO_ACIDS",
    "PATTERN_GROUPS",
    "PATTERN_NAMES",
    "PATTERN_FEATURES_KAPPA",
    "PATTERN_FEATURES_IWD",
    "DEFAULT_COMPOSITION_BACKGROUND_FILENAME",
    "DEFAULT_COMPOSITION_BACKGROUND_PATH",
    "COMPOSITION_FEATURES",
    "PATCH_FEATURES",
    "GrammarException",
    "GrammarCompositionStats",
    "GrammarPatterningConfig",
    "pattern_feature_names",
    "composition_feature_names",
    "patch_feature_names",
    "compute_composition_raw",
    "compute_composition_zscores",
    "compute_patterning_raw",
    "compute_patterning_scramble_distribution",
    "compute_patterning_zscores",
    "merge_feature_blocks",
    "compute_feature_vector",
    "compute_composition_background_stats",
    "save_composition_stats_npz",
    "load_composition_stats_npz",
    "load_default_composition_stats",
]
