"""The background Markov chain: a sequence rebuilt residue by residue from a background's grammar.

The model learns, from a background of natural sequences, how likely each
residue is to follow each run of ``order`` residues - an order-``order``
Markov chain - and builds a null of a sequence one residue at a time, each
drawn given the ``order`` residues before it. Like the background k-mer
model, it gives the null the background's local grammar in place of the
sequence's own; unlike it, there are no joins between blocks, so every
residue follows on from the ones before it.

A null always has exactly the sequence's composition. Drawn from the
background's chain alone, a null would follow the background's composition
until the residues it favours ran out. So the null is drawn from the chain
given that it ends up with exactly the sequence's composition. Over a long
sequence, that turns the chain into another one (Csiszar, Cover & Choi 1987;
see :mod:`sparrow.null.tilt`), in which residue ``a`` follows the run ``c``
with a chance proportional to

``P(a | c) t(a) h(c + a)``

where ``P(a | c)`` is the background's chance of ``a`` after ``c``; ``t(a)``
is a weight for each residue, the same wherever it is, found for each
sequence as the one that gives the new chain the sequence's composition;
and ``h(c + a)``, the Perron eigenvector of the tilted chain, is how well the
run that ``a`` makes leads on to the rest. Together they make a chain that
has the sequence's composition all along it, so it does not, for instance,
use up a residue early in the runs the background is fond of and run out of
it later, as the background's chain weighted only by the composition does.
Each null's composition is
then made exact by multiplying by the number of each residue left and
dividing by its fraction ``q(a)`` of the sequence, which is close to 1 until
the residues left stray from the sequence's proportions, and pulls them back.

With no grammar in the background - every residue as likely after any run as
overall - ``h`` is constant, ``t(a)`` is ``q(a) / f(a)`` for the residue's
fraction ``f(a)`` of the background, and this is a plain shuffle.

The first ``order`` residues have no run before them, so they are drawn as
a run of their own, from the runs the new chain spends its time in (its
stationary distribution), as if the null were a stretch from the middle of a
long sequence. That way no part of a null is made differently from any
other, apart from its last residues, which the composition forces.

The correction that makes the composition exact assumes the residues still
to place vary about as much as a random sample of the sequence's would. For a
residue the chain puts in runs, they vary more, and inside a run the
correction pulls against carrying on, so a little of that residue moves
towards the end. It only shows when the runs are a good part of the sequence:
from the yeast proteome at order 3, the first 254 residues of Sup35 get runs
of Q and N about 16 long, and the first quarter of each null holds about 6%
fewer of them than the rest; at order 2 it is about 1%, and for the 1,081
residues of Med15 less than 0.5% at either order.

Should no residue left ever follow the run before it in the background -
which happens near the end of a null, or with a small background - the
residue is drawn from a shorter run, dropping the earliest residue until one
does, by how much more likely it is after that run than overall,
``P(a | c) / f(a)``, times the number left; the model warns how many were.
With no run at all, a residue is drawn from what is left, so a null can always
be finished.

The nulls of a sequence are built together, one residue of every null at a
time, which is what makes many of them quick to make. Finding the chain for a
sequence takes longer: with the yeast proteome as the background, about
0.05 s from its 3-mers (order 2) and 0.3 s from its 4-mers (order 3).
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from numpy.typing import NDArray

from sparrow.null.base import NullModel
from sparrow.null.kmers import MAX_KMER_SIZE, count_background_kmers, kmer_residues
from sparrow.null.tilt import fit_tilt
from sparrow.properties import encoding
from sparrow.sparrow_exceptions import NullModelWarning, SparrowException

__all__ = ["MAX_ORDER", "MarkovBackground"]

#: The highest order the chain can have: each residue drawn given the 3 before
#: it, from the background's 4-mers (see :data:`sparrow.null.kmers.MAX_KMER_SIZE`).
MAX_ORDER = MAX_KMER_SIZE - 1

# How closely ARPACK finds the Perron eigenvectors, relative to the
# eigenvalue. The tilt they give is only needed to TILT_TOLERANCE, far coarser.
_EIGEN_TOLERANCE = 1e-10

# ARPACK needs at least this many states to find one eigenvector; fewer are
# solved densely
_SMALLEST_SPARSE_PROBLEM = 3


class MarkovBackground(NullModel):
    """Nulls drawn residue by residue from a background's Markov chain, with the sequence's composition.

    Parameters
    ----------
    order : int, optional
        How many residues before each one it is drawn given, from 0 - a plain
        shuffle - to :data:`MAX_ORDER`. Default 2, from the background's
        3-mers.

    Attributes
    ----------
    order : int

    Examples
    --------
    >>> from sparrow.null import MarkovBackground
    >>> model = MarkovBackground(order=2).fit("human_idrs.fasta")  # doctest: +SKIP
    >>> nulls = model.generate(sequence, count=10, seed=1)  # doctest: +SKIP
    """

    needs_background = True

    def __init__(self, order: int = 2) -> None:
        super().__init__()
        # bool is an int, but order=True is a mistake
        if isinstance(order, bool) or not isinstance(order, (int, np.integer)) or not 0 <= order <= MAX_ORDER:
            raise SparrowException(f"order must be an integer from 0 to {MAX_ORDER}, not {order!r}")
        self.order = int(order)
        self._transitions = np.zeros((1, encoding.NUM_RESIDUES))
        self._enrichment: list[NDArray[np.float64]] = []
        self._frequencies = np.zeros(encoding.NUM_RESIDUES)

    def __repr__(self) -> str:
        return f"MarkovBackground(order={self.order})"

    def _fit(self, sequences: list[str]) -> None:
        """Learn how likely each residue is after each run of the chain's order, and of every shorter length.

        Parameters
        ----------
        sequences : list of str
            The background, upper case.
        """
        counts = count_background_kmers(sequences, self.order + 1, type(self).__name__)
        self._frequencies = counts[1] / counts[1].sum()
        self._transitions = _transitions_after(counts[self.order + 1], self.order)

        # for the residues drawn from a shorter run than the order: how much
        # more likely each is after the run than overall. With no run, that is 1.
        self._enrichment = [np.ones((1, encoding.NUM_RESIDUES))]
        for run_length in range(1, self.order):
            transitions = _transitions_after(counts[run_length + 1], run_length)
            enrichment = np.divide(
                transitions, self._frequencies, out=np.zeros_like(transitions), where=self._frequencies > 0
            )
            self._enrichment.append(enrichment)

    def _generate(self, codes: NDArray[np.int8], count: int, rng: np.random.Generator) -> list[NDArray[np.int8]]:
        """Draw ``count`` nulls of one sequence, together, a residue at a time.

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
            If the sequence has a residue the background never has, or is
            shorter than the chain's order.
        """
        length = len(codes)
        if length < self.order:
            raise SparrowException(
                f"A sequence of {length} residues is shorter than the {self.order} residues each one is drawn "
                f"given by {self}; use a lower order."
            )
        composition = np.bincount(codes, minlength=encoding.NUM_RESIDUES).astype(np.int64)
        missing = [encoding.RESIDUES[code] for code in np.flatnonzero((composition > 0) & (self._frequencies == 0))]
        if missing:
            raise SparrowException(
                f"The background has no {', '.join(missing)}, so {self} cannot place it; "
                "fit it to a background with every residue the sequence has."
            )

        left = np.tile(composition, (count, 1))
        nulls = np.empty((count, length), dtype=np.int8)
        every_null = np.arange(count)
        shortened = np.zeros(count, dtype=np.int64)

        # each null's last `order` residues, as a k-mer index (see sparrow.null.kmers)
        run_before = np.zeros(count, dtype=np.int64)

        chains = list(self._enrichment)
        if self.order > 0:
            chain, runs, stationary = _conditioned_chain(self._transitions, self.order, composition, self._frequencies)
            chains.append(chain)

            # the first `order` residues, as a run drawn from those the chain spends its time in
            chances = _first_run_chances(runs, self.order, stationary, composition)
            first_runs = runs[rng.choice(len(runs), size=count, p=chances)]
            nulls[:, : self.order] = kmer_residues(first_runs, self.order)
            for position in range(self.order):
                left[every_null, nulls[:, position]] -= 1
            run_before = first_runs

        uniforms = rng.random((length, count))
        for position in range(self.order, length):
            weights, shortened_here = _weights(chains, run_before, left, self.order)
            shortened += shortened_here

            residues = _draw_each_row(weights, uniforms[position])
            nulls[:, position] = residues
            left[every_null, residues] -= 1
            run_before = (run_before * encoding.NUM_RESIDUES + residues) % encoding.NUM_RESIDUES**self.order

        _warn_about_shortened_runs(shortened, length, self.order)
        return list(nulls)


def _transitions_after(counts: NDArray[np.int64], run_length: int) -> NDArray[np.float64]:
    """The background's chance of each residue after each run.

    Parameters
    ----------
    counts : numpy.ndarray
        The count of every k-mer one residue longer than the run, indexed by
        k-mer index.
    run_length : int
        The run length.

    Returns
    -------
    numpy.ndarray
        ``(20**run_length, 20)``: for each run, by k-mer index, and each
        residue, ``P(residue | run)``; 0 for a run the background never has a
        residue after.
    """
    # a k-mer's index is its run's index times 20 plus its last residue, so
    # reshaping puts each run on a row, and each residue after it in a column
    after_run = counts.reshape(encoding.NUM_RESIDUES**run_length, encoding.NUM_RESIDUES).astype(np.float64)
    run_totals = after_run.sum(axis=1, keepdims=True)
    return np.divide(after_run, run_totals, out=np.zeros_like(after_run), where=run_totals > 0)


def _conditioned_chain(
    transitions: NDArray[np.float64],
    run_length: int,
    composition: NDArray[np.int64],
    frequencies: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """The chain conditioned on a composition: its weights for each residue after each run, and where it spends its time.

    Only the runs made of the sequence's residues are used, since no other
    can occur in its nulls.

    Parameters
    ----------
    transitions : numpy.ndarray
        ``(20**run_length, 20)`` the background's ``P(residue | run)``.
    run_length : int
        The run length, at least 1.
    composition : numpy.ndarray
        ``(20,)`` the count of each residue in the sequence, every one of them
        in the background.
    frequencies : numpy.ndarray
        ``(20,)`` each residue's frequency in the background.

    Returns
    -------
    chain : numpy.ndarray
        ``(20**run_length, 20)``: ``P(a | c) t(a) h(c + a) / q(a)`` for each run
        ``c`` and residue ``a``, by k-mer index; 0 for any run or residue the
        sequence does not have.
    runs : numpy.ndarray
        The k-mer index of every run of the sequence's residues.
    stationary : numpy.ndarray
        The fraction of its time the conditioned chain spends at each of those
        runs: the product of the tilted chain's left and right eigenvectors.
    """
    present = np.flatnonzero(composition)
    target = composition[present] / composition.sum()
    states = len(present) ** run_length

    # each run of the sequence's residues, numbered in base len(present), and
    # its k-mer index among all runs
    runs = _runs_of(present, run_length)
    after = transitions[np.ix_(runs, present)]

    # each possible step: from run `rows`, the residue `columns`, to the run
    # that makes - the old run without its first residue, with the new one on the end
    rows, columns = np.nonzero(after)
    next_runs = (rows % (states // len(present))) * len(present) + columns
    chances = after[rows, columns]

    vectors: dict[str, NDArray[np.float64]] = {}

    def tilted_steps(
        log_tilt: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        # the tilted chain as a matrix between runs, with the log of its Perron
        # root, each step's tilted chance times h at the run it goes to, and the
        # right and left eigenvectors; the tilt is scaled so the largest is 1
        largest = float(np.max(log_tilt))
        steps = chances * np.exp(log_tilt[columns] - largest)
        matrix = scipy.sparse.csr_matrix((steps, (rows, next_runs)), shape=(states, states))
        root, right, left = _perron(matrix, vectors)
        return np.log(root) + largest, steps * right[next_runs], right, left

    def log_normalizer(log_tilt: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        # A(theta) is the log of the tilted chain's Perron root, and its gradient
        # the composition of the chain it conditions to
        log_root, ahead, _, left = tilted_steps(log_tilt)
        flow = np.bincount(columns, weights=left[rows] * ahead, minlength=len(present))
        return log_root, flow / flow.sum()

    # with no grammar in the background, the tilt is q / f
    log_tilt = fit_tilt(log_normalizer, target, np.log(target / frequencies[present]))
    _, ahead, right, left = tilted_steps(log_tilt)

    chain = np.zeros((encoding.NUM_RESIDUES**run_length, encoding.NUM_RESIDUES))
    chain[runs[rows], present[columns]] = ahead / target[columns]
    stationary = left * right
    return chain, runs, stationary / stationary.sum()


def _first_run_chances(
    runs: NDArray[np.int64], run_length: int, stationary: NDArray[np.float64], composition: NDArray[np.int64]
) -> NDArray[np.float64]:
    """The chance of each run being a null's first, given that it has the sequence's composition.

    That is the fraction of its time the conditioned chain spends at the run,
    times the chance of drawing the run's residues one after another from the
    sequence's over the chance of drawing them from its composition, as for
    every residue after (see the module's description).

    Parameters
    ----------
    runs : numpy.ndarray
        The k-mer index of each run.
    run_length : int
        The run length.
    stationary : numpy.ndarray
        The fraction of its time the conditioned chain spends at each run.
    composition : numpy.ndarray
        ``(20,)`` the count of each residue in the sequence.

    Returns
    -------
    numpy.ndarray
        The chance of each run, summing to 1.
    """
    residues = kmer_residues(runs, run_length)
    fractions = composition / composition.sum()

    # the number of ways of drawing each run's residues from the sequence's,
    # taking one fewer of a residue each time it comes again
    ways = np.ones(len(runs))
    for position in range(run_length):
        earlier = np.count_nonzero(residues[:, :position] == residues[:, [position]], axis=1)
        ways *= np.clip(composition[residues[:, position]] - earlier, 0, None) / fractions[residues[:, position]]

    chances = stationary * ways
    return chances / chances.sum()


def _runs_of(residues: NDArray[np.int64], run_length: int) -> NDArray[np.int64]:
    """The k-mer index of every run of some residues.

    Parameters
    ----------
    residues : numpy.ndarray
        The residue codes runs are made of.
    run_length : int
        The run length.

    Returns
    -------
    numpy.ndarray
        ``(len(residues)**run_length,)`` the k-mer index of each run, with the
        runs numbered in base ``len(residues)``, first residue most significant.
    """
    numbered = np.arange(len(residues) ** run_length)
    runs = np.zeros(len(numbered), dtype=np.int64)
    for position in range(run_length):
        digit = (numbered // len(residues) ** (run_length - 1 - position)) % len(residues)
        runs = runs * encoding.NUM_RESIDUES + residues[digit]
    return runs


def _perron(
    matrix: scipy.sparse.csr_matrix, vectors: dict[str, NDArray[np.float64]]
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """The Perron root of a non-negative matrix, and its right and left eigenvectors.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Square and non-negative.
    vectors : dict
        The eigenvectors found last time for a similar matrix, under
        ``"right"`` and ``"left"``, to start from; updated with these ones.

    Returns
    -------
    root : float
        The largest real eigenvalue.
    right, left : numpy.ndarray
        Its eigenvectors, non-negative.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ARPACK cannot find them.
    """
    if matrix.shape[0] < _SMALLEST_SPARSE_PROBLEM:
        dense = matrix.toarray()
        right_values, right_vectors = np.linalg.eig(dense)
        left_values, left_vectors = np.linalg.eig(dense.T)
        root = float(right_values.real.max())
        right = right_vectors[:, np.argmax(right_values.real)]
        left = left_vectors[:, np.argmax(left_values.real)]
    else:
        try:
            values, right_vectors = scipy.sparse.linalg.eigs(
                matrix, k=1, which="LR", tol=_EIGEN_TOLERANCE, v0=vectors.get("right")
            )
            _, left_vectors = scipy.sparse.linalg.eigs(
                matrix.T.tocsr(), k=1, which="LR", tol=_EIGEN_TOLERANCE, v0=vectors.get("left")
            )
        except scipy.sparse.linalg.ArpackNoConvergence as error:
            raise SparrowException(
                "Could not find the Markov chain conditioned on the sequence's composition: the eigenvector "
                "search did not converge."
            ) from error
        root = float(values[0].real)
        right, left = right_vectors[:, 0], left_vectors[:, 0]

    # with no way through the runs that can go on, there is no chain to condition
    if root <= 0:
        raise SparrowException(
            "The background never has the sequence's residues follow one another in a way that can go on, "
            "so there is no Markov chain through them to draw a null from; use a lower order or a larger background."
        )

    # an eigenvector comes back with an arbitrary sign or phase; the Perron one
    # is non-negative
    vectors["right"] = np.abs(right.real)
    vectors["left"] = np.abs(left.real)
    return root, vectors["right"], vectors["left"]


def _weights(
    chains: list[NDArray[np.float64]], run_before: NDArray[np.int64], left: NDArray[np.int64], run_length: int
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """How likely each residue is to come next in each null.

    Each null starts with the longest run before it, ``run_length``
    residues, and drops the earliest residue of the run until some residue
    it has left can follow the run.

    Parameters
    ----------
    chains : list of numpy.ndarray
        For each run length up to the order, the weight of each residue after
        each run: the conditioned chain's, from :func:`_conditioned_chain`,
        for the order itself, and ``P(a | c) / f(a)`` for each shorter one.
    run_before : numpy.ndarray
        ``(nulls,)`` each null's last residues, as a k-mer index.
    left : numpy.ndarray
        ``(nulls, 20)`` the count of each residue each null has still to place.
    run_length : int
        The longest run any null has before this residue.

    Returns
    -------
    weights : numpy.ndarray
        ``(nulls, 20)`` each residue's weight; every row has a positive one.
    shortened : numpy.ndarray
        ``(nulls,)`` True for each null whose run had to be shortened.
    """
    weights = np.zeros(left.shape, dtype=np.float64)
    undecided = np.arange(len(left))
    shortened = np.zeros(len(left), dtype=bool)

    for length in range(run_length, -1, -1):
        # the last `length` residues are the lowest digits of the run's index
        run = run_before[undecided] % encoding.NUM_RESIDUES**length
        candidate = chains[length][run] * left[undecided]
        decided = candidate.sum(axis=1) > 0
        weights[undecided[decided]] = candidate[decided]

        undecided = undecided[~decided]
        if undecided.size == 0:
            break
        shortened[undecided] = True

    return weights, shortened


def _draw_each_row(weights: NDArray[np.float64], uniforms: NDArray[np.float64]) -> NDArray[np.int64]:
    """A column for each row, picked with a probability proportional to its weight.

    Parameters
    ----------
    weights : numpy.ndarray
        ``(rows, columns)`` non-negative weights, every row with a positive one.
    uniforms : numpy.ndarray
        ``(rows,)`` random numbers in [0, 1), which decide the picks.

    Returns
    -------
    numpy.ndarray
        ``(rows,)`` the picked columns, each one with a positive weight.
    """
    cumulative = np.cumsum(weights, axis=1)
    targets = uniforms * cumulative[:, -1]
    picked = np.count_nonzero(cumulative <= targets[:, None], axis=1)

    # rounding can carry a target to the very end; the last column with any
    # weight is then the one picked
    last_with_weight = weights.shape[1] - 1 - np.argmax(weights[:, ::-1] > 0, axis=1)
    return np.minimum(picked, last_with_weight)


def _warn_about_shortened_runs(shortened: NDArray[np.int64], length: int, order: int) -> None:
    """Warn how many residues were drawn given a shorter run than they had, if any were.

    Parameters
    ----------
    shortened : numpy.ndarray
        ``(nulls,)`` for each null, how many of its residues were.
    length : int
        The length of each null.
    order : int
        The chain's order.
    """
    total = int(shortened.sum())
    if total == 0:
        return
    nulls_with_any = int(np.count_nonzero(shortened))
    warnings.warn(
        f"In {nulls_with_any} of {len(shortened)} nulls, {total} of the {length * len(shortened)} residues were "
        f"drawn given fewer of the residues before them than the chain's order of {order}, because the background "
        "never has any of the residues left after those. The nulls still have the sequence's composition.",
        NullModelWarning,
        stacklevel=4,
    )
