"""Cutting a sequence into blocks of ``k`` residues.

The block shuffle and the background k-mer model both build a null from
blocks, and both take ``k`` as one block size or several. With one size,
``k=3``, the sequence is cut into blocks of 3, and whatever is left over -
fewer than 3 residues - is one shorter block. With several, ``k=[2, 3]``, it is
cut into blocks of each size, with each size covering as close to an equal
share of the residues as whole blocks allow, and a shorter block only if no
mix of the sizes adds up to the length.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence

import numpy as np

from sparrow.sparrow_exceptions import SparrowException

__all__ = ["check_block_sizes", "plan_blocks"]


def check_block_sizes(k: int | Sequence[int], name: str = "k", maximum: int | None = None) -> tuple[int, ...]:
    """The block sizes ``k`` names, checked.

    Parameters
    ----------
    k : int or sequence of int
        One block size, or several different ones.
    name : str, optional
        What the sizes are called in error messages. Default ``"k"``.
    maximum : int or None, optional
        The largest size allowed, or None for no limit. Default None.

    Returns
    -------
    tuple of int
        The sizes, smallest first.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If a size is not a positive integer or is larger than ``maximum``, no
        size is given, or a size is given twice.
    """
    if isinstance(k, (int, np.integer)):
        sizes = [k]
    elif isinstance(k, Iterable) and not isinstance(k, str):
        sizes = list(k)
    else:
        raise SparrowException(f"{name} must be a block size, or a list of them, not {k!r}")
    if not sizes:
        raise SparrowException(f"{name} must be a block size, or a list of them; an empty list was given")

    for size in sizes:
        # bool is an int, but a block of True residues is a mistake
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1:
            raise SparrowException(f"Each block size in {name} must be a positive integer, not {size!r}")
        if maximum is not None and size > maximum:
            raise SparrowException(f"Block sizes in {name} can be at most {maximum}, not {size}")

    if len(set(sizes)) != len(sizes):
        raise SparrowException(f"{name} names the same block size more than once: {sizes}")

    return tuple(sorted(int(size) for size in sizes))


def plan_blocks(length: int, block_sizes: tuple[int, ...]) -> list[int]:
    """The sizes of the blocks a sequence is cut into, in no particular order.

    Each size covers as close to an equal share of the residues as it can,
    with no residues left over if any mix of the sizes adds up to ``length``.
    Any that are left over make one shorter block, as when ``length`` is odd
    and the only size is 2.

    Parameters
    ----------
    length : int
        The number of residues, at least 1.
    block_sizes : tuple of int
        The sizes, smallest first, from :func:`check_block_sizes`.

    Returns
    -------
    list of int
        One size for each block, summing to ``length``: the largest blocks
        first, then any shorter block last.
    """
    smallest, larger = block_sizes[0], block_sizes[1:]
    share = length / len(block_sizes)

    # Every size but the smallest starts from the number of blocks nearest its
    # share, and the smallest takes whatever residues are left. Trying up to
    # (smallest - 1) blocks either side of each start lets those leftover
    # residues fall into any remainder modulo the smallest size, so the plan
    # with none left over is found whenever there is one. A plan of only the
    # smallest size always fits, should every other overshoot the length.
    nearest = [round(share / size) for size in larger]
    offsets = range(1 - smallest, smallest)
    candidates = [
        [count + offset for count, offset in zip(nearest, chosen)]
        for chosen in itertools.product(offsets, repeat=len(larger))
    ]
    candidates.append([0] * len(larger))

    best_counts: list[int] = []
    best_key: tuple[int, float] | None = None
    for larger_counts in candidates:
        used = sum(count * size for count, size in zip(larger_counts, larger))
        if min(larger_counts, default=0) < 0 or used > length:
            continue

        smallest_count, leftover = divmod(length - used, smallest)
        counts = [smallest_count, *larger_counts]
        unevenness = sum(abs(count * size - share) for count, size in zip(counts, block_sizes))

        # fewest residues left over first, then the most even shares
        key = (leftover, unevenness)
        if best_key is None or key < best_key:
            best_key = key
            best_counts = counts

    plan = [size for size, count in reversed(list(zip(block_sizes, best_counts))) for _ in range(count)]
    leftover = length - sum(plan)
    if leftover > 0:
        plan.append(leftover)
    return plan
