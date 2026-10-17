"""Turning amino acid sequences into arrays.

Every calculation in :mod:`sparrow.properties` starts here. A sequence is
validated once, by one validator with one error message, and is then handled as
integers rather than characters - which is what lets the calculations downstream
be vectorized.

There are two integer views of a sequence:

* **ASCII codes** (:func:`ascii_codes`), one ``uint8`` per residue. These index
  directly into 256-entry lookup tables such as :func:`membership_lookup`, so
  asking "is this residue in this set?" for every position is a single array
  index rather than a Python loop.
* **Residue codes** (:func:`residue_codes`), one integer per residue giving its
  index in :data:`RESIDUES`, the 20 standard amino acids in alphabetical order.
  These are what composition counts are built from.

The kappa-style patterning statistics use a third, derived view - the winged
ternary encoding of :func:`ternarize_winged` - which also lives here because it
is a pure re-encoding of the sequence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from sparrow.data import amino_acids
from sparrow.sparrow_exceptions import SparrowException

__all__ = [
    "RESIDUES",
    "NUM_RESIDUES",
    "RESIDUE_INDEX",
    "ASCII_TABLE_SIZE",
    "NOT_A_RESIDUE",
    "validate_protein_sequence",
    "is_valid_protein_sequence",
    "normalize_residue_selector",
    "ascii_codes",
    "membership_lookup",
    "residue_membership",
    "residue_codes",
    "as_batch",
    "EncodedBatch",
    "encode_batch",
    "scale_values",
    "scale_vector",
    "ternarize_winged",
    "ternarize_winged_batch",
]

#: The 20 standard amino acids, in the order that defines their residue codes.
RESIDUES: tuple[str, ...] = tuple(amino_acids.VALID_AMINO_ACIDS)

#: Number of standard amino acids.
NUM_RESIDUES = len(RESIDUES)

#: Residue code of each standard amino acid.
RESIDUE_INDEX: dict[str, int] = {residue: index for index, residue in enumerate(RESIDUES)}

#: Number of entries in an ASCII-indexed lookup table.
ASCII_TABLE_SIZE = 256

#: Residue code given to any character that is not a standard amino acid.
NOT_A_RESIDUE = -1

# ASCII code -> residue code, NOT_A_RESIDUE for everything else
_RESIDUE_CODE_TABLE = np.full(ASCII_TABLE_SIZE, NOT_A_RESIDUE, dtype=np.int8)
for _index, _residue in enumerate(RESIDUES):
    _RESIDUE_CODE_TABLE[ord(_residue)] = _index


def validate_protein_sequence(
    sequence: object,
    allow_empty: bool = True,
    uppercase: bool = True,
    exception_cls: type[Exception] = ValueError,
    sequence_name: str = "sequence",
) -> str:
    """Validate and optionally upper-case an amino acid sequence.

    Parameters
    ----------
    sequence : str
        Sequence to validate.
    allow_empty : bool, optional
        If False, an empty sequence raises. Default True.
    uppercase : bool, optional
        If True, upper-case the sequence before validating it. Default True.
    exception_cls : type, optional
        Exception class raised on failure. Default ``ValueError``.
    sequence_name : str, optional
        Name used for the sequence in error messages. Default ``"sequence"``.

    Returns
    -------
    str
        The validated sequence, upper-cased if ``uppercase`` is True.

    Raises
    ------
    Exception
        An ``exception_cls`` if the input is not a string, is empty when that is
        not allowed, or contains anything other than the 20 standard amino
        acids.
    """
    if not isinstance(sequence, str):
        raise exception_cls(f"{sequence_name} must be a string")

    if uppercase:
        sequence = sequence.upper()

    if (not allow_empty) and len(sequence) == 0:
        raise exception_cls(f"{sequence_name} must be non-empty")

    invalid = sorted(set(sequence) - set(RESIDUES))
    if len(invalid) > 0:
        raise exception_cls(f"Invalid residue(s) found in {sequence_name}: {invalid}")

    return sequence


def is_valid_protein_sequence(sequence: str) -> bool:
    """Whether a sequence contains only the 20 standard amino acids.

    Case matters: lower-case residues are not valid here.

    Parameters
    ----------
    sequence : str
        Protein sequence.

    Returns
    -------
    bool
        True if every character is a standard upper-case amino acid (an empty
        sequence counts as valid), else False.
    """
    try:
        validate_protein_sequence(sequence, allow_empty=True, uppercase=False)
    except ValueError:
        return False

    return True


def normalize_residue_selector(
    residue_selector: str | Iterable[str],
    selector_name: str = "residue_selector",
    exception_cls: type[Exception] = ValueError,
    uppercase: bool = True,
    require_nonempty: bool = True,
    unique: bool = False,
    sort_unique: bool = False,
    expected_length: int | None = None,
    return_type: str = "list",
) -> list[str] | str:
    """Normalize and validate a set of residues passed as a selector.

    Parameters
    ----------
    residue_selector : str or iterable of str
        One-letter residue codes, either joined (``"ILV"``) or as an iterable
        (``["I", "L", "V"]``).
    selector_name : str, optional
        Name used for the selector in error messages.
    exception_cls : type, optional
        Exception class raised on failure. Default ``ValueError``.
    uppercase : bool, optional
        If True (default), upper-case the residues.
    require_nonempty : bool, optional
        If True (default), an empty selector raises.
    unique : bool, optional
        If True, remove duplicate residues. Default False.
    sort_unique : bool, optional
        If True and ``unique`` is True, sort the residues after removing
        duplicates; otherwise first-seen order is kept. Default False.
    expected_length : int or None, optional
        If given, the selector must contain exactly this many residues.
    return_type : {'list', 'str'}, optional
        Whether to return a list of residues or a joined string. Default
        ``'list'``.

    Returns
    -------
    list of str or str
        The normalized selector.

    Raises
    ------
    Exception
        An ``exception_cls`` if the selector is not a string or iterable, is empty
        when that is not allowed, has the wrong length, contains a non-standard
        residue, or ``return_type`` is unknown.
    """
    if isinstance(residue_selector, str):
        selector = list(residue_selector)
    else:
        try:
            selector = [str(residue) for residue in residue_selector]
        except TypeError:
            raise exception_cls(f"{selector_name} must be a string or iterable of residues")

    if uppercase:
        selector = [residue.upper() for residue in selector]

    if unique:
        if sort_unique:
            selector = sorted(set(selector))
        else:
            selector = list(dict.fromkeys(selector))

    if require_nonempty and len(selector) == 0:
        raise exception_cls(f"{selector_name} must contain one or more residues")

    if expected_length is not None and len(selector) != expected_length:
        raise exception_cls(f"{selector_name} must contain exactly {expected_length} residues")

    for residue in selector:
        if residue not in RESIDUES:
            raise exception_cls(f"Invalid residue in {selector_name}: {residue}")

    if return_type == "list":
        return selector
    if return_type == "str":
        return "".join(selector)

    raise exception_cls("return_type must be 'list' or 'str'")


def ascii_codes(sequence: str) -> NDArray[np.uint8]:
    """A sequence as an array of its ASCII codes.

    Parameters
    ----------
    sequence : str
        Amino acid sequence. Must be ASCII.

    Returns
    -------
    numpy.ndarray
        ``uint8`` array of length ``len(sequence)``. It is a read-only view onto
        the encoded bytes, so copy it before writing to it.

    Raises
    ------
    UnicodeEncodeError
        If the sequence contains a non-ASCII character. Use
        :func:`residue_codes` where non-ASCII input should get sparrow's own
        error instead.
    """
    return np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)


def membership_lookup(residues: Iterable[str]) -> NDArray[np.bool_]:
    """A 256-entry table that is True at the ASCII code of each residue.

    Indexing it with :func:`ascii_codes` gives a per-position membership mask
    in one step: ``membership_lookup("DE")[ascii_codes(sequence)]``.

    Parameters
    ----------
    residues : iterable of str
        The residues to mark.

    Returns
    -------
    numpy.ndarray
        Boolean array of length 256.
    """
    table = np.zeros(ASCII_TABLE_SIZE, dtype=bool)
    for residue in residues:
        table[ord(residue)] = True
    return table


def residue_membership(residues: Iterable[str]) -> NDArray[np.bool_]:
    """A 20-entry table that is True at the residue code of each residue.

    The residue-code counterpart of :func:`membership_lookup`: indexing it with
    :func:`residue_codes` gives a per-position membership mask in one step.

    Parameters
    ----------
    residues : iterable of str
        The residues to mark.

    Returns
    -------
    numpy.ndarray
        Boolean array of length :data:`NUM_RESIDUES`.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If a residue is not one of the 20 standard amino acids.
    """
    table = np.zeros(NUM_RESIDUES, dtype=bool)
    for residue in residues:
        if residue not in RESIDUE_INDEX:
            raise SparrowException(f"{residue!r} is not one of the 20 standard amino acids")
        table[RESIDUE_INDEX[residue]] = True
    return table


def residue_codes(sequence: str) -> NDArray[np.int8]:
    """A sequence as residue codes: each residue's index in :data:`RESIDUES`.

    Parameters
    ----------
    sequence : str
        Amino acid sequence, upper case.

    Returns
    -------
    numpy.ndarray
        ``int8`` array of length ``len(sequence)``, every value in 0-19.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the sequence contains anything other than the 20 standard amino
        acids. The message lists the offending characters, including non-ASCII
        ones, rather than failing on the encoding.
    """
    # errors="replace" turns a non-ASCII character into "?", which has no residue
    # code, so it reaches the error below rather than raising UnicodeEncodeError
    codes = _RESIDUE_CODE_TABLE[np.frombuffer(sequence.encode("ascii", errors="replace"), dtype=np.uint8)]

    if np.any(codes == NOT_A_RESIDUE):
        bad = sorted({sequence[position] for position in np.flatnonzero(codes == NOT_A_RESIDUE)})
        raise SparrowException(
            f"Sequence contains non-standard residue(s) {bad}. sparrow's sequence properties are "
            "defined for the 20 standard amino acids; strip or convert other residues first "
            "(e.g. Protein(seq, validate=True))."
        )

    return codes


def as_batch(sequences: str | Sequence[str]) -> list[str]:
    """A batch of sequences as an upper-case list; a single string is a batch of one.

    Parameters
    ----------
    sequences : str or sequence of str
        One sequence, or many.

    Returns
    -------
    list of str
        The sequences, upper-cased, in their original order.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If any element is not a string.
    """
    if isinstance(sequences, str):
        return [sequences.upper()]

    batch = []
    for index, sequence in enumerate(sequences):
        if not isinstance(sequence, str):
            raise SparrowException(
                f"Expected amino acid sequences as strings, but item {index} is a {type(sequence).__name__}"
            )
        batch.append(sequence.upper())
    return batch


@dataclass(frozen=True)
class EncodedBatch:
    """A batch of sequences as residue codes, laid end to end.

    Concatenating means a batch of thousands of sequences is validated and
    encoded in one pass, and that calculations which run along the sequence can
    work on all of them at once. Sequence ``i`` occupies
    ``codes[offsets[i]:offsets[i + 1]]``.

    Attributes
    ----------
    codes : numpy.ndarray
        ``int8`` residue codes of every sequence in turn.
    lengths : numpy.ndarray
        ``int64`` length of each sequence.
    """

    codes: NDArray[np.int8]
    lengths: NDArray[np.int64]

    def __len__(self) -> int:
        """Number of sequences in the batch."""
        return int(self.lengths.size)

    @cached_property
    def offsets(self) -> NDArray[np.int64]:
        """Where each sequence starts in :attr:`codes`, plus the total length at the end."""
        offsets = np.zeros(self.lengths.size + 1, dtype=np.int64)
        np.cumsum(self.lengths, out=offsets[1:])
        return offsets

    @cached_property
    def owner(self) -> NDArray[np.int64]:
        """Which sequence each position of :attr:`codes` belongs to."""
        owner: NDArray[np.int64] = np.repeat(np.arange(self.lengths.size, dtype=np.int64), self.lengths)
        return owner

    def by_length(self) -> list[tuple[NDArray[np.int64], NDArray[np.int8]]]:
        """The batch split into groups of sequences that share a length.

        Calculations that run along a sequence - windows, pairwise distances -
        are vectorized over a group as a ``(k, length)`` array. Grouping by exact
        length rather than padding to a common one means every sequence is
        reduced exactly as it would be on its own, so its value does not depend
        on what else is in the batch.

        Returns
        -------
        list of tuple
            One ``(indices, codes)`` pair per distinct length, shortest first:
            the positions of those sequences in the batch, and their residue
            codes as an ``int8`` array of shape ``(len(indices), length)``.
        """
        if self.lengths.size == 1:
            # one sequence is one group, and needs no sorting
            return [(np.zeros(1, dtype=np.int64), self.codes[None, :])]

        order = np.argsort(self.lengths, kind="stable")
        boundaries = np.flatnonzero(np.diff(self.lengths[order])) + 1
        groups = []
        for indices in np.split(order, boundaries):
            if indices.size == 0:
                continue
            length = int(self.lengths[indices[0]])
            positions = self.offsets[indices][:, None] + np.arange(length)
            groups.append((indices.astype(np.int64), self.codes[positions]))
        return groups


def encode_batch(sequences: str | Sequence[str] | EncodedBatch) -> EncodedBatch:
    """Validate and encode a batch of sequences.

    Parameters
    ----------
    sequences : str, sequence of str, or EncodedBatch
        One sequence, or many. Lower-case residues are upper-cased first. An
        :class:`EncodedBatch` is returned unchanged.

    Returns
    -------
    EncodedBatch
        The batch's residue codes and lengths.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If any sequence contains a non-standard residue. For a batch of more
        than one, the message says which sequence.
    """
    if isinstance(sequences, EncodedBatch):
        return sequences

    batch = as_batch(sequences)
    lengths = np.fromiter((len(sequence) for sequence in batch), dtype=np.int64, count=len(batch))
    try:
        codes = residue_codes("".join(batch))
    except SparrowException:
        if len(batch) == 1:
            raise
        # find the first offending sequence, so the error can say which one it was
        for index, sequence in enumerate(batch):
            try:
                residue_codes(sequence)
            except SparrowException as error:
                raise SparrowException(f"Sequence {index} of {len(batch)} in the batch: {error}") from error
        raise
    return EncodedBatch(codes=codes, lengths=lengths)


def scale_values(sequence: str, scale: Mapping[str, float | None]) -> NDArray[np.float64]:
    """The value of a per-residue scale at every position of a sequence.

    Parameters
    ----------
    sequence : str
        Amino acid sequence, upper case.
    scale : mapping of str to float
        Value of each residue, e.g. :data:`sparrow.data.amino_acids.AA_hydro_KD`.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of length ``len(sequence)``.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the sequence contains a non-standard residue, or a residue the scale
        has no value for.
    """
    values = scale_vector(scale, needed=set(sequence))
    return values[residue_codes(sequence)]


def scale_vector(scale: Mapping[str, float | None], needed: Iterable[str] | None = None) -> NDArray[np.float64]:
    """A per-residue scale as a length-20 array indexed by residue code.

    Residues the scale has no value for are NaN, so they cannot silently
    contribute a number; ``needed`` turns that into an error for the residues a
    calculation is actually going to use.

    Parameters
    ----------
    scale : mapping of str to float
        Value of each residue. A value of None counts as missing.
    needed : iterable of str, optional
        Residues that must have a value. Default None (no requirement).

    Returns
    -------
    numpy.ndarray
        ``float64`` array of length :data:`NUM_RESIDUES`.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If a residue in ``needed`` has no value in ``scale``.
    """
    values = np.full(NUM_RESIDUES, np.nan, dtype=np.float64)
    for index, residue in enumerate(RESIDUES):
        value = scale.get(residue)
        if value is not None:
            values[index] = float(value)

    if needed is not None:
        missing = sorted(
            residue for residue in set(needed) if residue in RESIDUES and np.isnan(values[RESIDUES.index(residue)])
        )
        if missing:
            raise SparrowException(f"The scale has no value for residue(s) {missing}")

    return values


def ternarize_winged(
    sequence: str,
    group1: Sequence[str],
    group2: Sequence[str],
    window_size: int,
) -> NDArray[np.int8]:
    """Encode a sequence as a ternary array padded with non-contributing wings.

    Positions holding a ``group1`` residue become ``+1``, ``group2`` residues
    become ``-1``, and everything else ``0``. The array is padded with
    ``window_size`` zeros at each end, which is what lets clusters at the
    termini contribute to kappa as much as those in the middle.

    If ``group2`` is empty the encoding is binary: every residue not in
    ``group1`` is treated as ``-1``.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    group1 : sequence of str
        Residues forming the first group. Must be non-empty.
    group2 : sequence of str
        Residues forming the second group. May be empty, see above.
    window_size : int
        Window size, which also sets the wing width.

    Returns
    -------
    numpy.ndarray
        ``int8`` array of length ``2 * window_size + len(sequence)``.
    """
    stack = ternarize_winged_batch(residue_codes(sequence)[None, :], group1, group2, window_size)
    ternary: NDArray[np.int8] = stack[0]
    return ternary


def ternarize_winged_batch(
    codes: NDArray[np.int8],
    group1: Sequence[str],
    group2: Sequence[str],
    window_size: int,
) -> NDArray[np.int8]:
    """Ternarize a stack of equal-length sequences at once.

    The batch form of :func:`ternarize_winged`. It takes residue codes rather
    than strings so that a caller scoring many features over the same scrambles
    can encode once and reuse the codes.

    Parameters
    ----------
    codes : numpy.ndarray
        Residue codes (:func:`residue_codes`) of shape ``(n_sequences, length)``.
    group1, group2 : sequence of str
        Residue groups; ``group2`` may be empty for the binary encoding.
    window_size : int
        Window size, which also sets the wing width.

    Returns
    -------
    numpy.ndarray
        ``int8`` array of shape ``(n_sequences, 2 * window_size + length)``.
    """
    codes = np.atleast_2d(codes)
    n_sequences, length = codes.shape

    ternary = np.zeros((n_sequences, 2 * window_size + length), dtype=np.int8)
    core = ternary[:, window_size : window_size + length]

    in_group1 = residue_membership(group1)[codes]

    if len(group2) == 0:
        # binary: everything that is not group1 is group2
        core[:] = np.where(in_group1, 1, -1)
    else:
        in_group2 = residue_membership(group2)[codes]
        core[:] = np.where(in_group1, 1, np.where(in_group2, -1, 0))

    return ternary
