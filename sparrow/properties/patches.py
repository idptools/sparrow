"""Patch fractions: how much of a sequence lies in patches of a residue type.

A *patch* is a stretch rich in a set of target residues. Target residues
separated by no more than ``interruption`` other residues are bridged into one
candidate patch, and the candidate counts if it holds at least
``min_target_count`` target residues (and, optionally, enough copies of a
two-residue motif such as ``RG``). The patch fraction is the number of positions
covered by patches that count, divided by the sequence length. These are the
NARDINI+ patch features.

The whole batch is processed at once. Its sequences are laid end to end
(:class:`~sparrow.properties.encoding.EncodedBatch`), every candidate patch in
every sequence is found in one pass along that array - with sequence boundaries
acting as breaks a patch cannot cross - and each candidate's length, target
count and motif count come from a single ``bincount``. A patch fraction is a
whole number of positions over the length, so the values are exact.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from sparrow.properties import encoding
from sparrow.sparrow_exceptions import CalculationException

__all__ = [
    "DEFAULT_INTERRUPTION",
    "DEFAULT_MIN_TARGET_COUNT",
    "DEFAULT_MIN_ADJACENT_RG_PAIRS",
    "patch_fraction",
    "rg_patch_fraction",
]

#: Maximum number of non-target residues bridged inside a patch. NARDINI+ uses 2.
DEFAULT_INTERRUPTION = 2

#: Minimum number of target residues for a patch to count. NARDINI+ uses 4.
DEFAULT_MIN_TARGET_COUNT = 4

#: Minimum number of adjacent ``RG`` pairs for an RG patch to count. NARDINI+ uses 2.
DEFAULT_MIN_ADJACENT_RG_PAIRS = 2

#: What the functions in this module accept.
PatchBatch = str | Sequence[str] | encoding.EncodedBatch


def _covered_positions(batch: encoding.EncodedBatch, hit: NDArray[np.bool_], interruption: int) -> NDArray[np.bool_]:
    """Positions inside a candidate patch: target residues, plus bridged gaps.

    A non-target position is bridged when the nearest target residues on either
    side of it belong to its own sequence and are no more than
    ``interruption + 1`` positions apart - that is, the gap between them holds at
    most ``interruption`` residues.
    """
    n_positions = hit.size
    positions = np.arange(n_positions)

    # nearest target residue at or before, and at or after, every position
    previous_hit = np.maximum.accumulate(np.where(hit, positions, -1))
    next_hit = np.minimum.accumulate(np.where(hit, positions, n_positions)[::-1])[::-1]

    owner = batch.owner
    has_both = (previous_hit >= 0) & (next_hit < n_positions)
    previous_owner = owner[np.clip(previous_hit, 0, n_positions - 1)]
    next_owner = owner[np.clip(next_hit, 0, n_positions - 1)]
    within_sequence = has_both & (previous_owner == owner) & (next_owner == owner)

    bridged = ~hit & within_sequence & (next_hit - previous_hit <= interruption + 1)
    covered: NDArray[np.bool_] = hit | bridged
    return covered


def _patch_span(
    batch: encoding.EncodedBatch,
    targets: Sequence[int],
    interruption: int,
    min_target_count: int | None,
    motif: tuple[int, int] | None,
    min_adjacent_pair_count: int,
) -> NDArray[np.int64]:
    """Number of positions of each sequence covered by patches that count."""
    n_sequences = len(batch)
    codes = batch.codes
    if codes.size == 0:
        return np.zeros(n_sequences, dtype=np.int64)

    is_target = np.zeros(encoding.NUM_RESIDUES, dtype=bool)
    is_target[list(targets)] = True
    hit = is_target[codes]
    covered = _covered_positions(batch, hit, interruption)

    # A candidate patch is a run of covered positions. A run starts wherever a
    # covered position follows an uncovered one, or begins a sequence - so runs
    # never cross from one sequence into the next.
    starts_sequence = np.zeros(codes.size, dtype=bool)
    starts_sequence[batch.offsets[:-1][batch.lengths > 0]] = True
    follows_covered = np.concatenate([[False], covered[:-1]])
    run_start = covered & (~follows_covered | starts_sequence)
    run_id = np.cumsum(run_start) - 1
    n_runs = int(run_start.sum())
    if n_runs == 0:
        return np.zeros(n_sequences, dtype=np.int64)

    run_length = np.bincount(run_id[covered], minlength=n_runs)
    counts = np.ones(n_runs, dtype=bool)

    if min_target_count is not None:
        counts &= np.bincount(run_id[hit], minlength=n_runs) >= min_target_count

    if motif is not None:
        # an adjacent pair inside one run - the next position is in the same run
        # only if it is covered and does not start a new run
        same_run = covered[:-1] & covered[1:] & ~run_start[1:]
        is_motif = same_run & (codes[:-1] == motif[0]) & (codes[1:] == motif[1])
        motif_count = np.bincount(run_id[:-1][is_motif], minlength=n_runs)
        # a candidate shorter than two residues cannot hold a pair at all
        counts &= (run_length >= 2) & (motif_count >= min_adjacent_pair_count)

    run_owner = batch.owner[run_start]
    span = np.bincount(run_owner[counts], weights=run_length[counts], minlength=n_sequences)
    return np.rint(span).astype(np.int64)


def _residue_codes_of(selector: Iterable[str]) -> list[int]:
    return [encoding.RESIDUE_INDEX[residue] for residue in selector]


def patch_fraction(
    sequences: PatchBatch,
    residue_selector: str | Iterable[str],
    interruption: int = DEFAULT_INTERRUPTION,
    min_target_count: int | None = DEFAULT_MIN_TARGET_COUNT,
    adjacent_pair_pattern: str | Iterable[str] | None = None,
    min_adjacent_pair_count: int = 0,
) -> NDArray[np.float64]:
    """Fraction of each sequence covered by patches of a set of residues.

    Parameters
    ----------
    sequences : str, sequence of str, or EncodedBatch
        The batch.
    residue_selector : str or iterable of str
        One or more residues that define patch membership.
    interruption : int, optional
        Maximum number of non-target residues bridged inside a patch. Default 2.
    min_target_count : int or None, optional
        Minimum number of target residues for a candidate patch to count.
        Default 4. None disables this filter.
    adjacent_pair_pattern : str or iterable of str, optional
        A two-residue motif (for example ``"RG"``) that must occur inside each
        patch. Default None (no motif requirement).
    min_adjacent_pair_count : int, optional
        Minimum number of occurrences of ``adjacent_pair_pattern`` in a patch.
        Ignored without a pattern. Default 0. With a pattern, a candidate patch
        of a single residue never counts.

    Returns
    -------
    numpy.ndarray
        ``(n_sequences,)`` fractions between 0 and 1; 0.0 for an empty sequence.

    Raises
    ------
    sparrow.sparrow_exceptions.CalculationException
        If an argument is out of range, or the selector or motif is not made of
        standard residues.
    sparrow.sparrow_exceptions.SparrowException
        If a sequence contains a non-standard residue.

    Examples
    --------
    >>> patch_fraction(["AAAAGGGG", "AAGGAAGG", "GGAGG"], "A").tolist()
    [0.5, 0.75, 0.0]
    """
    if interruption < 0:
        raise CalculationException("interruption must be >= 0")
    if min_target_count is not None and min_target_count < 1:
        raise CalculationException("min_target_count must be >= 1 or None")
    if min_adjacent_pair_count < 0:
        raise CalculationException("min_adjacent_pair_count must be >= 0")

    selector = encoding.normalize_residue_selector(
        residue_selector,
        selector_name="residue_selector",
        exception_cls=CalculationException,
        uppercase=True,
        require_nonempty=True,
        unique=True,
        sort_unique=True,
        return_type="list",
    )
    motif = None
    if adjacent_pair_pattern is not None:
        pattern = encoding.normalize_residue_selector(
            adjacent_pair_pattern,
            selector_name="adjacent_pair_pattern",
            exception_cls=CalculationException,
            uppercase=True,
            require_nonempty=True,
            expected_length=2,
            return_type="list",
        )
        first, second = _residue_codes_of(pattern)
        motif = (first, second)

    batch = encoding.encode_batch(sequences)
    span = _patch_span(
        batch, _residue_codes_of(selector), interruption, min_target_count, motif, min_adjacent_pair_count
    )
    fraction: NDArray[np.float64] = np.divide(span, np.maximum(batch.lengths, 1), dtype=np.float64)
    return fraction


def rg_patch_fraction(
    sequences: PatchBatch,
    interruption: int = DEFAULT_INTERRUPTION,
    min_adjacent_rg_pairs: int = DEFAULT_MIN_ADJACENT_RG_PAIRS,
) -> NDArray[np.float64]:
    """Fraction of each sequence covered by RG-motif patches.

    A patch of R and G residues counts when it holds at least
    ``min_adjacent_rg_pairs`` adjacent ``RG`` pairs, however many R and G
    residues it has. This is the NARDINI+ ``'RG Frac'`` feature.

    Parameters
    ----------
    sequences : str, sequence of str, or EncodedBatch
        The batch.
    interruption : int, optional
        Maximum number of non-R/G residues bridged inside a patch. Default 2.
    min_adjacent_rg_pairs : int, optional
        Minimum number of adjacent ``RG`` pairs in a patch. Default 2.

    Returns
    -------
    numpy.ndarray
        ``(n_sequences,)`` fractions between 0 and 1.

    Raises
    ------
    sparrow.sparrow_exceptions.CalculationException
        If ``min_adjacent_rg_pairs`` is below 1 or ``interruption`` below 0.
    """
    if min_adjacent_rg_pairs < 1:
        raise CalculationException("min_adjacent_rg_pairs must be >= 1")
    return patch_fraction(
        sequences,
        residue_selector="RG",
        interruption=interruption,
        min_target_count=None,
        adjacent_pair_pattern="RG",
        min_adjacent_pair_count=min_adjacent_rg_pairs,
    )
