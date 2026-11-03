"""The block shuffle: a sequence cut into blocks of k residues, in a random order.

With ``k=1`` this is a plain shuffle, which keeps the composition and nothing
else. With larger blocks, every residue keeps the neighbours it had inside its
block, so ``k=2`` keeps half the sequence's pairs of neighbouring residues and
``k=3`` two thirds of them, while the order of the blocks is random. With
several sizes, ``k=[2, 3]``, the sequence is cut into blocks of 2 and 3 with
about half its residues in each (see :mod:`sparrow.null.blocks`).

The cuts start at a random point, so every pair of neighbouring residues is as
likely as any other to be kept together. With ``k=2`` that means a block of 1
residue at either end half the time, rather than the same pairs every time.

The shuffle needs no background: it is an empirical null, made from the
sequence alone.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from sparrow.null.base import NullModel
from sparrow.null.blocks import check_block_sizes, plan_blocks
from sparrow.sparrow_exceptions import SparrowException

__all__ = ["BlockShuffle"]


class BlockShuffle(NullModel):
    """Nulls made by cutting the sequence into blocks and shuffling the blocks.

    Parameters
    ----------
    k : int or sequence of int, optional
        The block size, or several, each with about an equal share of the
        residues. Default 1, a plain shuffle.

    Attributes
    ----------
    block_sizes : tuple of int
        The block sizes, smallest first.

    Examples
    --------
    >>> from sparrow.null import BlockShuffle
    >>> nulls = BlockShuffle(k=2).generate("MEEEKKKKSSSTTTDDDQQQQ", count=3, seed=1)
    >>> len(nulls), len(nulls[0])
    (3, 21)
    """

    needs_background = False

    def __init__(self, k: int | Sequence[int] = 1) -> None:
        super().__init__()
        self.block_sizes = check_block_sizes(k)

    def __repr__(self) -> str:
        k = self.block_sizes[0] if len(self.block_sizes) == 1 else list(self.block_sizes)
        return f"BlockShuffle(k={k})"

    def _generate(self, codes: NDArray[np.int8], count: int, rng: np.random.Generator) -> list[NDArray[np.int8]]:
        """Shuffle the blocks of one sequence, ``count`` times.

        Parameters
        ----------
        codes : numpy.ndarray
            The sequence, as residue codes.
        count : int
            How many nulls to make.
        rng : numpy.random.Generator

        Returns
        -------
        list of numpy.ndarray
            The nulls, as residue codes.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If the sequence is no longer than one block, so every null would be
            the sequence itself.
        """
        plan = plan_blocks(len(codes), self.block_sizes)
        if len(plan) == 1:
            raise SparrowException(
                f"A sequence of {len(codes)} residues is a single block with block sizes {self.block_sizes}, "
                "so every null would be the sequence itself."
            )

        nulls = []
        for _ in range(count):
            blocks = np.split(codes, np.cumsum(_cut(plan, rng))[:-1])
            order = rng.permutation(len(blocks))
            nulls.append(np.concatenate([blocks[index] for index in order]))
        return nulls


def _cut(plan: list[int], rng: np.random.Generator) -> list[int]:
    """The sizes of the blocks along the sequence, from its start, for one null.

    The planned blocks are put in a random order, and the cuts are then moved
    along by a random number of residues within the first block: what the
    move cuts off its start becomes a block at the end. That way the cuts
    start at a random point even when every block is the same size.

    Parameters
    ----------
    plan : list of int
        The block sizes, from :func:`sparrow.null.blocks.plan_blocks`.
    rng : numpy.random.Generator

    Returns
    -------
    list of int
        The block sizes, in order along the sequence, summing to its length.
    """
    sizes = [int(size) for size in rng.permutation(plan)]
    offset = int(rng.integers(sizes[0]))
    if offset == 0:
        return sizes
    return [sizes[0] - offset, *sizes[1:], offset]
