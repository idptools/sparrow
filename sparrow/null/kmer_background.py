"""Background k-mer reconstruction: a sequence rebuilt from a background's k-mers.

The model counts every k-mer in a background of natural sequences, and builds
a null of a sequence from blocks of ``k`` residues, each one a k-mer drawn
according to how often it occurs in the background, so the null has the
background's local grammar in place of the sequence's own. With several
sizes, ``k=[2, 3]``, about half the residues are in blocks of each size (see
:mod:`sparrow.null.blocks`).

A null always has exactly the sequence's composition. Drawn by background
frequency alone, the blocks would follow the background's composition until
the residues it favours ran out, and leave the rest to pile up together. So
the blocks are drawn from the background given that, together, they have
exactly the sequence's composition. For a k-mer ``b`` with background
frequency ``P(b)``, that makes its chance the product of

* ``P(b) t(b1) t(b2) ... t(bk)``: its background frequency, tilted towards the
  sequence's composition by a weight ``t`` for each of its residues. The
  weights are found for each sequence, as those with which k-mers drawn this
  way have, on average, the sequence's composition (see
  :mod:`sparrow.null.tilt`); and
* the chance of drawing its residues one after another from those of the
  sequence not yet in a block, ``n(b1)/N * (n(b2) - [b2 = b1])/(N - 1) * ...``,
  over the chance of drawing them from the sequence's composition,
  ``q(b1) q(b2) ...``. This is close to 1 until the residues left stray from
  the sequence's proportions, and pulls them back, which is what makes the
  composition exact.

With no grammar in the background - every k-mer as frequent as its residues
make it - ``t(a)`` is ``q(a) / f(a)``, for the residue's fraction ``q(a)`` of
the sequence and ``f(a)`` of the background, and this is a plain shuffle. The
blocks are put in a random order once drawn, so no part of the null is made
differently from any other.

Which residues a k-mer is made of decides its second factor and its tilt, and
their order along it only its background frequency. So the k-mers are grouped
by the residues they are made of, and a group is drawn first, then an order
within it: 8,855 groups of four residues, rather than 160,000 4-mers, for each
block.

A block of a size with no k-mer that both occurs in the background and is
made of residues still left - which happens near the end of a null, or with
a small background - is drawn as two
smaller ones instead, a block one residue shorter and a single residue, and
the model warns how many were.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sparrow.null.base import NullModel
from sparrow.null.blocks import check_block_sizes, plan_blocks
from sparrow.null.kmers import MAX_KMER_SIZE, count_background_kmers, kmer_residues, weighted_choice
from sparrow.null.tilt import fit_tilt
from sparrow.properties import encoding
from sparrow.sparrow_exceptions import NullModelWarning, SparrowException

__all__ = ["KmerBackground"]


@dataclass(frozen=True)
class _KmerGroups:
    """The k-mers of one size seen in the background, grouped by the residues they are made of.

    Attributes
    ----------
    residues : numpy.ndarray
        ``(groups, size)`` the residues of each group, smallest code first.
    repeats : numpy.ndarray
        ``(groups, size)`` how many times each group's residue at that place
        comes before it in the group, so that drawing the group's residues
        from what is left takes ``left[residue] - repeats`` of each.
    probability : numpy.ndarray
        ``(groups,)`` each group's background frequency: the sum of its
        k-mers' frequencies.
    first, stop : numpy.ndarray
        ``(groups,)`` where each group's k-mers start and end in ``orderings``.
    orderings : numpy.ndarray
        ``(kmers,)`` every k-mer seen, as residues along it, group by group.
    ordering_probability : numpy.ndarray
        ``(kmers,)`` each k-mer's background frequency, in the order of
        ``orderings``.
    """

    residues: NDArray[np.int64]
    repeats: NDArray[np.int64]
    probability: NDArray[np.float64]
    first: NDArray[np.int64]
    stop: NDArray[np.int64]
    orderings: NDArray[np.int64]
    ordering_probability: NDArray[np.float64]

    @classmethod
    def from_counts(cls, counts: NDArray[np.int64], size: int) -> _KmerGroups:
        """Group the k-mers of one size that the background has.

        Parameters
        ----------
        counts : numpy.ndarray
            The count of every k-mer of the size, indexed by k-mer index.
        size : int
            The k-mer length.

        Returns
        -------
        _KmerGroups
        """
        seen = np.flatnonzero(counts)
        residues = kmer_residues(seen, size)
        probability = counts[seen] / counts.sum()

        # the k-mers made of the same residues share the index of those residues sorted
        sorted_residues = np.sort(residues, axis=1)
        group_index = np.zeros(len(seen), dtype=np.int64)
        for position in range(size):
            group_index = group_index * encoding.NUM_RESIDUES + sorted_residues[:, position]
        group_ids, group_of = np.unique(group_index, return_inverse=True)

        order = np.argsort(group_of, kind="stable")
        group_sizes = np.bincount(group_of)
        stop = np.cumsum(group_sizes)

        group_residues = kmer_residues(group_ids, size)
        repeats = np.zeros_like(group_residues)
        for position in range(1, size):
            same_as_before = group_residues[:, position] == group_residues[:, position - 1]
            repeats[:, position] = np.where(same_as_before, repeats[:, position - 1] + 1, 0)

        return cls(
            residues=group_residues,
            repeats=repeats,
            probability=np.bincount(group_of, weights=probability),
            first=stop - group_sizes,
            stop=stop,
            orderings=residues[order],
            ordering_probability=probability[order],
        )

    def made_of(self, present: NDArray[np.bool_]) -> _KmerGroups:
        """Only the groups made entirely of some residues.

        Parameters
        ----------
        present : numpy.ndarray
            ``(20,)`` True for each residue a group may have.

        Returns
        -------
        _KmerGroups
            The same k-mers, with only those groups.
        """
        keep = np.all(present[self.residues], axis=1)
        return _KmerGroups(
            residues=self.residues[keep],
            repeats=self.repeats[keep],
            probability=self.probability[keep],
            first=self.first[keep],
            stop=self.stop[keep],
            orderings=self.orderings,
            ordering_probability=self.ordering_probability,
        )


class KmerBackground(NullModel):
    """Nulls built from blocks drawn from a background's k-mers, with the sequence's composition.

    Parameters
    ----------
    k : int or sequence of int, optional
        The block size, or several, each with about an equal share of the
        residues; each at most :data:`~sparrow.null.kmers.MAX_KMER_SIZE`.
        Default 3.

    Attributes
    ----------
    block_sizes : tuple of int
        The block sizes, smallest first.

    Examples
    --------
    >>> from sparrow.null import KmerBackground
    >>> model = KmerBackground(k=3).fit("human_idrs.fasta")  # doctest: +SKIP
    >>> nulls = model.generate(sequence, count=10, seed=1)  # doctest: +SKIP
    """

    needs_background = True

    def __init__(self, k: int | Sequence[int] = 3) -> None:
        super().__init__()
        self.block_sizes = check_block_sizes(k, maximum=MAX_KMER_SIZE)
        self._groups: dict[int, _KmerGroups] = {}
        self._frequencies = np.zeros(encoding.NUM_RESIDUES)

    def __repr__(self) -> str:
        k = self.block_sizes[0] if len(self.block_sizes) == 1 else list(self.block_sizes)
        return f"KmerBackground(k={k})"

    def _fit(self, sequences: list[str]) -> None:
        """Count and group the background's k-mers.

        Every length up to the largest block size is counted, since a block
        that cannot be drawn at its size is drawn as shorter ones.

        Parameters
        ----------
        sequences : list of str
            The background, upper case.
        """
        counts = count_background_kmers(sequences, max(self.block_sizes), type(self).__name__)
        self._frequencies = counts[1] / counts[1].sum()
        self._groups = {size: _KmerGroups.from_counts(counts[size], size) for size in counts if size > 1}

    def _generate(self, codes: NDArray[np.int8], count: int, rng: np.random.Generator) -> list[NDArray[np.int8]]:
        """Build ``count`` nulls of one sequence from the background's k-mers.

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
            If the sequence has a residue the background never has.
        """
        composition = np.bincount(codes, minlength=encoding.NUM_RESIDUES).astype(np.int64)
        missing = [encoding.RESIDUES[code] for code in np.flatnonzero((composition > 0) & (self._frequencies == 0))]
        if missing:
            raise SparrowException(
                f"The background has no {', '.join(missing)}, so {self} cannot place it; "
                "fit it to a background with every residue the sequence has."
            )
        plan = plan_blocks(len(codes), self.block_sizes)

        # only the groups made of the sequence's residues can ever be drawn, and
        # each is weighted by its tilted background frequency
        groups = {}
        tilted = {}
        for size, all_groups in self._groups.items():
            groups[size] = all_groups.made_of(composition > 0)
            tilted[size] = _tilted_weights(groups[size], composition, self._frequencies)

        nulls = []
        split_blocks = []
        for _ in range(count):
            null, split = _build_null(composition, plan, groups, tilted, rng)
            nulls.append(null)
            split_blocks.append(split)

        _warn_about_split_blocks(split_blocks, len(plan))
        return nulls


def _tilted_weights(
    kmers: _KmerGroups, composition: NDArray[np.int64], frequencies: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Each group's background frequency, tilted towards a composition and divided by its chance under it.

    That is ``P(b) t(b1) ... t(bk) / (q(b1) ... q(bk))`` for each group of
    k-mers ``b``, from the first factor of a block's chance and the denominator
    of its second (see the module's description). Only the residues the groups
    have are tilted to, in the proportions they have in the sequence: a residue
    of the sequence that no k-mer of this size in the background has is only
    ever drawn on its own.

    Parameters
    ----------
    kmers : _KmerGroups
        The groups, all made of the sequence's residues.
    composition : numpy.ndarray
        ``(20,)`` the count of each residue in the sequence.
    frequencies : numpy.ndarray
        ``(20,)`` each residue's frequency in the background.

    Returns
    -------
    numpy.ndarray
        ``(groups,)`` the weights, the largest 1.
    """
    if len(kmers.probability) == 0:
        return np.zeros(0)

    size = kmers.residues.shape[1]
    covered = np.unique(kmers.residues)
    target = composition[covered] / composition[covered].sum()

    # how many of each covered residue each group has
    place_of = np.full(encoding.NUM_RESIDUES, -1, dtype=np.int64)
    place_of[covered] = np.arange(len(covered))
    residue_counts = np.zeros((len(kmers.probability), len(covered)))
    for position in range(size):
        np.add.at(residue_counts, (np.arange(len(kmers.probability)), place_of[kmers.residues[:, position]]), 1)

    log_probability = np.log(kmers.probability)

    def log_normalizer(log_tilt: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        # A(theta) is the log of the tilted frequencies' sum, per residue, and its
        # gradient the tilted k-mers' average composition
        log_weight = log_probability + residue_counts @ log_tilt
        largest = float(log_weight.max())
        weight = np.exp(log_weight - largest)
        total = float(weight.sum())
        return (largest + np.log(total)) / size, (weight @ residue_counts) / (size * total)

    # with no grammar in the background, the tilt is q / f
    log_tilt = fit_tilt(log_normalizer, target, np.log(target / frequencies[covered]))

    log_weight = log_probability + residue_counts @ (log_tilt - np.log(target))
    return np.exp(log_weight - log_weight.max())


def _build_null(
    composition: NDArray[np.int64],
    plan: list[int],
    groups: dict[int, _KmerGroups],
    tilted: dict[int, NDArray[np.float64]],
    rng: np.random.Generator,
) -> tuple[NDArray[np.int8], int]:
    """One null, built block by block from the background's k-mers.

    Parameters
    ----------
    composition : numpy.ndarray
        ``(20,)`` the count of each residue in the sequence.
    plan : list of int
        The block sizes, from :func:`sparrow.null.blocks.plan_blocks`.
    groups : dict
        For each block size above 1, the background's k-mers of that size.
    tilted : dict
        For each block size above 1, each group's tilted weight, from
        :func:`_tilted_weights`.
    rng : numpy.random.Generator

    Returns
    -------
    null : numpy.ndarray
        The null, as residue codes.
    split : int
        How many blocks were drawn as two smaller ones.
    """
    left = composition.copy()
    to_draw = [int(size) for size in rng.permutation(plan)]
    blocks = []
    split = 0

    while to_draw:
        size = to_draw.pop()
        block = _draw_block(size, left, groups, tilted, rng)
        if block is None:
            # no k-mer of this size in the background is made of the residues
            # left, so the block is drawn as one a residue shorter and a single residue
            to_draw.extend([1, size - 1])
            split += 1
            continue
        left -= np.bincount(block, minlength=encoding.NUM_RESIDUES)
        blocks.append(block)

    order = rng.permutation(len(blocks))
    null = np.concatenate([blocks[index] for index in order]).astype(np.int8)
    return null, split


def _draw_block(
    size: int,
    left: NDArray[np.int64],
    groups: dict[int, _KmerGroups],
    tilted: dict[int, NDArray[np.float64]],
    rng: np.random.Generator,
) -> NDArray[np.int64] | None:
    """One block, drawn from the background's k-mers and the residues left.

    Parameters
    ----------
    size : int
        The block size.
    left : numpy.ndarray
        ``(20,)`` the count of each residue not yet in a block.
    groups : dict
        For each block size above 1, the background's k-mers of that size.
    tilted : dict
        For each block size above 1, each group's tilted weight.
    rng : numpy.random.Generator

    Returns
    -------
    numpy.ndarray or None
        The block's residue codes, or None if no k-mer of this size in the
        background is made of the residues left.
    """
    # a single residue has no grammar to draw on, so it is drawn from what is left
    if size == 1:
        return np.array([weighted_choice(left, rng.random())], dtype=np.int64)

    kmers = groups[size]

    # the number of ways of drawing each group's residues from what is left
    ways = np.prod(np.clip(left[kmers.residues] - kmers.repeats, 0, None), axis=1)
    weights = tilted[size] * ways
    if not np.any(weights > 0):
        return None

    group = weighted_choice(weights, rng.random())
    first, stop = kmers.first[group], kmers.stop[group]
    ordering = first + weighted_choice(kmers.ordering_probability[first:stop], rng.random())
    return kmers.orderings[ordering]


def _warn_about_split_blocks(split_blocks: list[int], planned: int) -> None:
    """Warn how many blocks were drawn as smaller ones, if any were.

    Parameters
    ----------
    split_blocks : list of int
        For each null, how many of its blocks were split.
    planned : int
        How many blocks each null was planned to have.
    """
    total = sum(split_blocks)
    if total == 0:
        return
    nulls_with_any = sum(1 for split in split_blocks if split > 0)
    warnings.warn(
        f"In {nulls_with_any} of {len(split_blocks)} nulls, {total} of the {planned * len(split_blocks)} blocks "
        "were drawn as smaller ones, because the background has no k-mer of their size made of the residues "
        "left. The nulls still have the sequence's composition.",
        NullModelWarning,
        stacklevel=4,
    )
