"""The k-let shuffle: a shuffle that keeps the count of every k-let.

A k-let is a run of ``k`` consecutive residues. A k-let shuffle makes a
sequence with exactly as many of each k-let as the original: with ``k=2``,
the same count of every pair of neighbouring residues (every dipeptide), and
so also the same composition. It is the protein form of the dinucleotide
shuffle of Altschul & Erickson (1985), the usual null for asking whether a
sequence's order means more than its local statistics. Unlike a block
shuffle, it carries no residues around as rigid blocks, and makes no pairs
that the sequence does not have.

The first and last ``k - 1`` residues stay where they are, as in Altschul &
Erickson and in uShuffle (Jiang et al. 2008). Every sequence with the same
k-lets and the same ends is equally likely.

How it works
------------
Write the sequence as a walk through a graph whose points are its (k-1)-lets,
with an edge for each k-let, from the (k-1)-let it starts with to the one it
ends with. Every walk that starts at the first (k-1)-let and uses each edge
once - an Eulerian trail - spells out a sequence with the same k-lets, and
every such sequence is one of these walks. A random one is made as Kandel et
al. (1996) describe:

1. For each point but the last, pick the edge by which the walk leaves it for
   the last time, so that these edges form a tree leading to the last point.
   Wilson's (1996) algorithm picks the tree uniformly at random.
2. Put each point's other edges in a random order, with its last edge last.
3. Walk from the first point, leaving each point by its next unused edge.

The tree in step 1 means the walk can never be stuck at a point with edges
left elsewhere, and choosing the tree and orders uniformly makes every trail
equally likely.

References
----------
Altschul SF, Erickson BW (1985) Significance of nucleotide sequence
alignments: a method for random sequence permutation that preserves
dinucleotide and codon usage. *Mol Biol Evol* 2:526-538.

Kandel D, Matias Y, Unger R, Winkler P (1996) Shuffling biological sequences.
*Discrete Appl Math* 71:171-185.

Wilson DB (1996) Generating random spanning trees more quickly than the cover
time. *Proc 28th ACM Symp Theory Comput* 296-303.

Jiang M, Anderson J, Gillespie J, Mayo M (2008) uShuffle: a useful tool for
shuffling biological sequences while preserving the k-let counts. *BMC
Bioinformatics* 9:192.
"""

from __future__ import annotations

import random

import numpy as np
from numpy.typing import NDArray

from sparrow.null.base import NullModel
from sparrow.sparrow_exceptions import SparrowException

__all__ = ["KletShuffle"]

# The largest seed drawn from the NumPy generator for the Python one, which
# makes the many single choices of a walk far faster than NumPy can
_PYTHON_SEED_LIMIT = 2**63


class KletShuffle(NullModel):
    """Nulls with exactly the k-let counts of the sequence.

    Parameters
    ----------
    k : int, optional
        The length of the runs of residues whose counts are kept. Default 2,
        every dipeptide; 1 is a plain shuffle.

    Attributes
    ----------
    k : int

    Notes
    -----
    A short sequence, or one with few distinct k-lets, may have few other
    sequences with the same k-lets, and then a null is often the sequence
    itself. ``KletShuffle(k=2)`` can only give ``"MKMKM"`` back unchanged,
    for instance.

    Examples
    --------
    >>> from sparrow.null import KletShuffle
    >>> null = KletShuffle(k=2).generate("MEEEKKKKSSSTTTDDDQQQQ", seed=1)[0]
    >>> null[0], null[-1]
    ('M', 'Q')
    """

    needs_background = False

    def __init__(self, k: int = 2) -> None:
        super().__init__()
        # bool is an int, but k=True is a mistake
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise SparrowException(f"k must be a positive integer, not {k!r}")
        self.k = int(k)

    def __repr__(self) -> str:
        return f"KletShuffle(k={self.k})"

    def _generate(self, codes: NDArray[np.int8], count: int, rng: np.random.Generator) -> list[NDArray[np.int8]]:
        """Make ``count`` k-let shuffles of one sequence.

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
            If ``k`` is not shorter than the sequence, so its only k-let shuffle
            is itself.
        """
        if self.k >= len(codes):
            raise SparrowException(
                f"A sequence of {len(codes)} residues has no k-let shuffle with k={self.k} but itself; "
                "k must be shorter than the sequence."
            )
        if self.k == 1:
            return [rng.permutation(codes) for _ in range(count)]

        out_edges = _out_edges(codes, self.k)
        chooser = random.Random(int(rng.integers(_PYTHON_SEED_LIMIT)))
        return [_random_trail(codes, self.k, out_edges, chooser) for _ in range(count)]


def _out_edges(codes: NDArray[np.int8], k: int) -> dict[bytes, list[int]]:
    """The graph of a sequence's k-lets.

    Parameters
    ----------
    codes : numpy.ndarray
        The sequence, as residue codes.
    k : int
        The k-let length, at least 2.

    Returns
    -------
    dict
        For each (k-1)-let that some k-let starts with, as bytes of residue
        codes, the residue that ends each such k-let, in the order they come
        in the sequence - so its edges, each going to the (k-1)-let that its
        k-let ends with.
    """
    sequence = codes.tobytes()
    out_edges: dict[bytes, list[int]] = {}
    for start in range(len(sequence) - k + 1):
        out_edges.setdefault(sequence[start : start + k - 1], []).append(sequence[start + k - 1])
    return out_edges


def _random_trail(
    codes: NDArray[np.int8], k: int, out_edges: dict[bytes, list[int]], chooser: random.Random
) -> NDArray[np.int8]:
    """One uniformly random sequence with the k-lets and ends of ``codes``.

    Parameters
    ----------
    codes : numpy.ndarray
        The sequence, as residue codes.
    k : int
        The k-let length, at least 2.
    out_edges : dict
        The graph of its k-lets, from :func:`_out_edges`.
    chooser : random.Random
        The generator every choice is made with.

    Returns
    -------
    numpy.ndarray
        The null, as residue codes.
    """
    sequence = codes.tobytes()
    first, last = sequence[: k - 1], sequence[len(sequence) - k + 1 :]

    # 1. the edge each point is left by for the last time
    last_exits = _last_exits(out_edges, last, chooser)

    # 2. each point's edges in a random order, with its last one last
    edge_orders: dict[bytes, list[int]] = {}
    for point, edges in out_edges.items():
        others = list(range(len(edges)))
        if point in last_exits:
            others.remove(last_exits[point])
        chooser.shuffle(others)
        if point in last_exits:
            others.append(last_exits[point])
        edge_orders[point] = [edges[index] for index in others]

    # 3. the walk from the first point, each time by the point's next unused edge
    null = bytearray(first)
    used = dict.fromkeys(edge_orders, 0)
    point = first
    for _ in range(len(sequence) - k + 1):
        residue = edge_orders[point][used[point]]
        used[point] += 1
        null.append(residue)
        point = point[1:] + bytes((residue,))

    return np.frombuffer(bytes(null), dtype=np.int8).copy()


def _last_exits(out_edges: dict[bytes, list[int]], last: bytes, chooser: random.Random) -> dict[bytes, int]:
    """A uniformly random tree of edges leading to the last point, by Wilson's algorithm.

    From each point not yet in the tree, a random walk is taken until it
    reaches the tree, remembering only the edge it last left each point by -
    which erases any loops the walk made - and the path that remembering
    traces is added to the tree.

    Parameters
    ----------
    out_edges : dict
        The graph, from :func:`_out_edges`.
    last : bytes
        The point every walk ends at: the sequence's last (k-1)-let.
    chooser : random.Random
        The generator every choice is made with.

    Returns
    -------
    dict
        For each point but ``last``, the index in its ``out_edges`` of the edge
        it is left by for the last time.
    """
    in_tree = {last}
    last_exits: dict[bytes, int] = {}

    for start in out_edges:
        # walk until the tree is reached, keeping each point's latest exit
        point = start
        while point not in in_tree:
            edges = out_edges[point]
            last_exits[point] = chooser.randrange(len(edges))
            point = point[1:] + bytes((edges[last_exits[point]],))

        # the loop-erased path the kept exits trace joins the tree
        point = start
        while point not in in_tree:
            in_tree.add(point)
            point = point[1:] + bytes((out_edges[point][last_exits[point]],))

    return last_exits
