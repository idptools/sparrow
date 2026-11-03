"""The parent class of every null model, and what they share.

A null model makes sequences that match a natural sequence in what you choose
to keep - its composition, its pairs of neighbouring residues, the local
grammar of a background set of natural sequences - and in nothing else, so
whatever evolution or function wrote into the order of its residues is gone.
Comparing a sequence with its nulls asks whether some feature of it is more
than what was kept would give by itself.

Every null model is used the same way:

* make it with its options, such as ``BlockShuffle(k=2)``;
* if it learns from a background - a large set of natural sequences, such as
  a proteome or its IDRs - give it the background with :meth:`NullModel.fit`;
* ask it for nulls of any sequence with :meth:`NullModel.generate`.

.. code-block:: python

   from sparrow.null import BlockShuffle, KmerBackground

   nulls = BlockShuffle(k=2).generate(sequence, count=10, seed=1)

   model = KmerBackground(k=3).fit("human_idrs.fasta")
   nulls = model.generate(sequence, count=10, seed=1)

Given a seed, the nulls of a sequence depend on the seed and the sequence
alone. So a loop over many sequences with one seed gives each sequence nulls
of its own, and gives a sequence the same nulls however many others it is
generated alongside.

To write a new null model, derive from :class:`NullModel`, set
``needs_background``, and implement ``_generate`` (and ``_fit``, if it learns
from a background).
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Mapping
from typing import ClassVar, TypeVar

import numpy as np
import protfasta
from numpy.typing import NDArray

from sparrow.properties import encoding
from sparrow.protein import Protein
from sparrow.sparrow_exceptions import SparrowException

__all__ = ["NullModel", "BackgroundInput"]

#: What :meth:`NullModel.fit` takes as a background: the path to a FASTA file,
#: a dict of sequences (or ``Protein`` objects), or any iterable of them.
BackgroundInput = str | os.PathLike[str] | Mapping[Hashable, str | Protein] | Iterable[str | Protein]

# the residues as bytes, indexed by residue code, to turn codes back into a sequence
_RESIDUE_BYTES = np.frombuffer("".join(encoding.RESIDUES).encode("ascii"), dtype=np.uint8)

# the type of a null model, so fit() can return the model it was called on
_Model = TypeVar("_Model", bound="NullModel")


class NullModel(ABC):
    """A way of making null sequences that match a natural sequence.

    Every null model takes its options when it is made, learns from a
    background with :meth:`fit` if it needs one, and makes nulls of a sequence
    with :meth:`generate`. A null always has the length and the composition of
    the sequence it was made from.
    """

    #: Whether the model learns from a background, and so must be given one
    #: with :meth:`fit` before it can make nulls. The shuffles do not.
    needs_background: ClassVar[bool]

    def __init__(self) -> None:
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the model has learnt from a background.

        Returns
        -------
        bool
            True once :meth:`fit` has been called; always False for a model
            that needs no background.
        """
        return self._fitted

    def fit(self: _Model, background: BackgroundInput) -> _Model:
        """Learn from a background set of natural sequences.

        Fitting again replaces what the model learnt before.

        Parameters
        ----------
        background : str, os.PathLike, dict, or iterable
            The path to a FASTA file, a dict of sequences or ``Protein``
            objects, or an iterable of either. Sequences are upper-cased. A
            residue other than the 20 standard amino acids is not counted, and
            neither is anything that spans it; the model warns how many there
            were.

        Returns
        -------
        NullModel
            The model itself, so it can be made and fit in one line.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If the model needs no background, the path is not a file, there are
            no sequences, or an item is neither a string nor a ``Protein``.
        """
        if not self.needs_background:
            raise SparrowException(
                f"{type(self).__name__} makes its nulls from the sequence alone, so it takes no background; "
                "call generate() directly."
            )
        self._fit(read_background(background))
        self._fitted = True
        return self

    def generate(self, sequence: str | Protein, count: int = 1, seed: int | None = None) -> list[str]:
        """Make nulls of a sequence.

        Parameters
        ----------
        sequence : str or Protein
            The natural sequence. It is upper-cased, and must be made of the 20
            standard amino acids.
        count : int, optional
            How many nulls to make. Default 1.
        seed : int or None, optional
            Seed for the random number generator, a non-negative integer. The
            nulls depend on the seed and the sequence together, so one seed
            gives different sequences different nulls. With None, the default,
            they are different every time.

        Returns
        -------
        list of str
            ``count`` nulls, each the length of ``sequence`` and with its
            composition.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If the model needs a background and has not been fit, the sequence
            is empty or has a residue other than the 20 standard amino acids,
            ``count`` is not a positive integer, or ``seed`` is not None or a
            non-negative integer; and as each model describes.
        """
        if self.needs_background and not self._fitted:
            raise SparrowException(
                f"{type(self).__name__} makes its nulls from a background; give it one with fit() first."
            )
        codes = sequence_codes(sequence)
        _check_count(count)
        rng = _generator(seed, codes)
        return [decode(null) for null in self._generate(codes, count, rng)]

    def _fit(self, sequences: list[str]) -> None:
        """Learn from the background's sequences.

        Implemented by every model that needs a background.

        Parameters
        ----------
        sequences : list of str
            The background, upper-cased, exactly as given.
        """
        raise NotImplementedError(f"{type(self).__name__} does not learn from a background")

    @abstractmethod
    def _generate(self, codes: NDArray[np.int8], count: int, rng: np.random.Generator) -> list[NDArray[np.int8]]:
        """Make the nulls of one sequence.

        Parameters
        ----------
        codes : numpy.ndarray
            The sequence as residue codes (see
            :func:`sparrow.properties.encoding.residue_codes`), never empty.
        count : int
            How many nulls to make, at least 1.
        rng : numpy.random.Generator
            The generator every random choice is made with.

        Returns
        -------
        list of numpy.ndarray
            ``count`` nulls as residue codes, each the length of ``codes``.
        """


def read_background(background: BackgroundInput) -> list[str]:
    """The sequences of a background, upper-cased.

    A FASTA file is read with every record kept, whatever its header and
    whatever residues it has, since the k-mer counts decide what to do with a
    non-standard residue and say how many there were.

    Parameters
    ----------
    background : str, os.PathLike, dict, or iterable
        As for :meth:`NullModel.fit`.

    Returns
    -------
    list of str
        One sequence for each record or item, in order.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the path is not a file, there are no sequences, or an item is
        neither a string nor a ``Protein``.
    """
    if isinstance(background, (str, os.PathLike)):
        path = os.fspath(background)
        if not os.path.isfile(path):
            raise SparrowException(
                f"{path!r} is not a file. A background given as a single string is read as the path to a FASTA file."
            )

        records = protfasta.read_fasta(
            path,
            expect_unique_header=False,
            duplicate_record_action="ignore",
            invalid_sequence_action="ignore",
            return_list=True,
        )
        values: list[str | Protein] = [record[1] for record in records]
    elif isinstance(background, Mapping):
        values = list(background.values())
    else:
        values = list(background)

    if not values:
        raise SparrowException("The background has no sequences.")

    sequences = []
    for index, value in enumerate(values):
        if isinstance(value, Protein):
            sequences.append(value.sequence.upper())
        elif isinstance(value, str):
            sequences.append(value.upper())
        else:
            raise SparrowException(
                f"Background item {index} is a {type(value).__name__}; a background is made of amino acid "
                "sequences, as strings or Protein objects."
            )
    return sequences


def sequence_codes(sequence: str | Protein) -> NDArray[np.int8]:
    """A sequence to make nulls of, as residue codes.

    Parameters
    ----------
    sequence : str or Protein
        The sequence. It is upper-cased.

    Returns
    -------
    numpy.ndarray
        ``int8`` residue codes, one for each residue.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the sequence is not a string or ``Protein``, is empty, or has a
        residue other than the 20 standard amino acids.
    """
    if isinstance(sequence, Protein):
        sequence = sequence.sequence
    if not isinstance(sequence, str):
        raise SparrowException(
            f"A null is made from an amino acid sequence, as a string or Protein, not a {type(sequence).__name__}."
        )
    if len(sequence) == 0:
        raise SparrowException("Cannot make a null of an empty sequence.")
    return encoding.residue_codes(sequence.upper())


def decode(codes: NDArray[np.integer]) -> str:
    """Residue codes as a sequence.

    Parameters
    ----------
    codes : numpy.ndarray
        Residue codes, each 0-19.

    Returns
    -------
    str
        The sequence.
    """
    return _RESIDUE_BYTES[codes].tobytes().decode("ascii")


def _check_count(count: int) -> None:
    """Check that the number of nulls asked for is a positive integer.

    Parameters
    ----------
    count : int

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If it is not.
    """
    # bool is an int, but True nulls is a mistake
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise SparrowException(f"count must be a positive integer, not {count!r}")


def _generator(seed: int | None, codes: NDArray[np.int8]) -> np.random.Generator:
    """The random number generator for the nulls of one sequence.

    The generator is seeded from the seed and a hash of the sequence together,
    so one seed gives each sequence nulls of its own, and the same nulls
    however many other sequences are generated with it.

    Parameters
    ----------
    seed : int or None
        A non-negative integer, or None for a generator seeded afresh from the
        operating system.
    codes : numpy.ndarray
        The sequence, as residue codes.

    Returns
    -------
    numpy.random.Generator

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``seed`` is not None or a non-negative integer.
    """
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise SparrowException(f"seed must be None or a non-negative integer, not {seed!r}")

    sequence_hash = int.from_bytes(hashlib.sha256(codes.tobytes()).digest(), "big")
    return np.random.default_rng([int(seed), sequence_hash])
