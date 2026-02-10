"""Sequence patching primitives.

This module contains reusable primitives for estimating patch coverage in
protein sequences using NARDINI-style semantics.
"""

import numpy as np

from sparrow import sparrow_exceptions
from sparrow.tools import general_tools


def _bridge_positions(positions, interruption):
    if positions.size == 0:
        return positions

    gaps = positions[1:] - positions[:-1]
    fill_idx = np.where((gaps > 1) & (gaps <= interruption + 1))[0]
    if fill_idx.size:
        extras = []
        for idx in fill_idx:
            extras.extend(range(positions[idx] + 1, positions[idx + 1]))
        if extras:
            positions = np.unique(
                np.concatenate([positions, np.array(extras, dtype=np.int32)])
            )
    return positions


def patch_fraction(
    sequence,
    residue_selector,
    interruption=2,
    min_target_count=4,
    adjacent_pair_pattern=None,
    min_adjacent_pair_count=0,
):
    """Compute the sequence fraction covered by residue patches.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    residue_selector : str or iterable[str]
        One or more residues that define patch membership.
    interruption : int, default 2
        Maximum number of non-target residues that can be bridged inside a patch.
    min_target_count : int or None, default 4
        Minimum number of target residues required for a bridged region to count.
        If ``None``, no minimum target-count filter is applied.
    adjacent_pair_pattern : str or iterable[str] or None, default None
        Optional adjacent two-residue motif required inside each bridged region
        (for example ``"RG"``).
    min_adjacent_pair_count : int, default 0
        Minimum number of occurrences of ``adjacent_pair_pattern`` required for
        a bridged region to count. Ignored when ``adjacent_pair_pattern`` is
        ``None``.

    Returns
    -------
    float
        Fraction of sequence positions covered by valid patch spans.
    """

    if interruption < 0:
        raise sparrow_exceptions.CalculationException("interruption must be >= 0")
    if min_target_count is not None and min_target_count < 1:
        raise sparrow_exceptions.CalculationException(
            "min_target_count must be >= 1 or None"
        )
    if min_adjacent_pair_count < 0:
        raise sparrow_exceptions.CalculationException(
            "min_adjacent_pair_count must be >= 0"
        )

    sequence = general_tools.validate_protein_sequence(
        sequence,
        allow_empty=True,
        uppercase=True,
        exception_cls=sparrow_exceptions.CalculationException,
        sequence_name="sequence",
    )
    if len(sequence) == 0:
        return 0.0

    selector = general_tools.normalize_residue_selector(
        residue_selector,
        selector_name="residue_selector",
        exception_cls=sparrow_exceptions.CalculationException,
        uppercase=True,
        require_nonempty=True,
        unique=True,
        sort_unique=True,
        return_type="list",
    )
    motif = None
    if adjacent_pair_pattern is not None:
        motif = general_tools.normalize_residue_selector(
            adjacent_pair_pattern,
            selector_name="adjacent_pair_pattern",
            exception_cls=sparrow_exceptions.CalculationException,
            uppercase=True,
            require_nonempty=True,
            unique=False,
            sort_unique=False,
            expected_length=2,
            return_type="str",
        )

    seq_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    selector_bytes = np.array([ord(residue) for residue in selector], dtype=np.uint8)
    hit_mask = np.isin(seq_bytes, selector_bytes)
    positions = np.where(hit_mask)[0]
    if positions.size == 0:
        return 0.0

    positions = _bridge_positions(positions, interruption)

    mark = np.zeros(seq_bytes.shape[0], dtype=np.int8)
    mark[positions] = 1
    diff = np.diff(np.concatenate([[0], mark, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    patch_span = 0
    for start, end in zip(starts, ends):
        if min_target_count is not None:
            if int(np.sum(hit_mask[start:end])) < min_target_count:
                continue

        if motif is not None:
            region = seq_bytes[start:end]
            if region.size < 2:
                continue
            motif_count = int(
                np.sum((region[:-1] == ord(motif[0])) & (region[1:] == ord(motif[1])))
            )
            if motif_count < min_adjacent_pair_count:
                continue

        patch_span += end - start

    return patch_span / len(sequence)


def rg_patch_fraction(sequence, interruption=2, min_adjacent_rg_pairs=2):
    """Compute NARDINI-style RG patch span fraction.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    interruption : int, default 2
        Maximum bridged interruption size inside a candidate RG patch.
    min_adjacent_rg_pairs : int, default 2
        Minimum number of adjacent ``RG`` pairs required inside a bridged region.

    Returns
    -------
    float
        Fraction of sequence positions covered by valid RG patch spans.
    """

    if min_adjacent_rg_pairs < 1:
        raise sparrow_exceptions.CalculationException(
            "min_adjacent_rg_pairs must be >= 1"
        )

    return patch_fraction(
        sequence=sequence,
        residue_selector="RG",
        interruption=interruption,
        min_target_count=None,
        adjacent_pair_pattern="RG",
        min_adjacent_pair_count=min_adjacent_rg_pairs,
    )
