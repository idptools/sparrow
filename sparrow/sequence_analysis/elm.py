"""Eukaryotic Linear Motifs (ELMs) in a sequence, and how a point mutation changes them.

ELMs are short linear motifs - degrons, docking sites, modification sites,
targeting signals - each defined by a regular expression in the ELM resource.
This module finds the matches of every ELM class in a sequence, and compares the
matches of a sequence with those of a point mutant to say which motifs the
mutation destroys, creates or leaves alone.

The class definitions are the ELM resource's ``elm_classes.tsv`` (download
version 1.4, 2 April 2024, 349 classes), bundled as
``sparrow/data/elm_classes.tsv``. If you use these annotations, please cite the
ELM resource: Kumar M, et al. ELM - the Eukaryotic Linear Motif resource - 2024
update. Nucleic Acids Res. 2024;52(D1):D442-D455.

A match is a candidate motif, not evidence of a functional site. Many ELM
patterns are short enough to match by chance, and each class's ``probability``
says how likely that is.

From a :class:`~sparrow.protein.Protein`, ``Protein.elms`` gives the matches
(found once and cached), and the comparison functions accept either a
``Protein`` or a sequence.
"""

import functools
import numbers
import re
from dataclasses import dataclass
from typing import Set, Tuple, Union

import sparrow
from sparrow.data import amino_acids
from sparrow.sparrow_exceptions import SparrowException

# The residues a substitution may introduce: the 20 standard amino acids
_STANDARD_RESIDUES = frozenset(amino_acids.VALID_AMINO_ACIDS)

# Three-letter codes as HGVS writes them ('Ser'), mapped to one-letter codes
_HGVS_THREE_LETTER = {code.capitalize(): letter for code, letter in amino_acids.AA_THREE_TO_ONE.items()}

# A protein substitution in HGVS notation: p.S7A or p.Ser7Ala, optionally in the
# parentheses HGVS uses for a predicted consequence, p.(Ser7Ala). A residue is a
# three-letter code or a single letter.
_HGVS_SUBSTITUTION = re.compile(r"^p\.(\()?([A-Z][a-z]{2}|[A-Za-z])(\d+)([A-Z][a-z]{2}|[A-Za-z])(?(1)\))$")


@dataclass(frozen=True)
class ELM:
    """One match of an ELM class in a sequence.

    Attributes
    ----------
    regex : str
        The ELM class's regular expression.
    identifier : str
        The ELM identifier, for example ``'DEG_Nend_UBRbox_1'``.
    functional_site_name : str
        The functional site the class belongs to, for example ``'N-degron'``.
    description : str
        The ELM resource's description of the class.
    probability : float
        The class's pattern probability from the ELM resource - how likely the
        pattern is to match by chance - or NaN where the table gives none.
    start : int
        Position of the first matched residue (0-indexed).
    end : int
        Position after the last matched residue, so ``sequence[start:end]`` of
        the searched sequence is the match.
    sequence : str
        The matched residues.

    Notes
    -----
    Two matches count as the same motif when they have the same regular
    expression, functional site and start position. That is what lets
    :func:`compute_lost_elms` and its relatives compare the matches of a sequence
    and of a point mutant with set operations, and it means a motif whose start
    moves counts as both lost and gained.
    """

    regex: str
    identifier: str
    functional_site_name: str
    description: str
    probability: float
    start: int
    end: int
    sequence: str

    def _motif_key(self):
        """What makes two matches the same motif; see the class Notes."""
        return (self.regex, self.functional_site_name, self.start)

    def __eq__(self, other):
        # let Python fall back to the reflected operation / identity comparison
        # rather than raising AttributeError on unrelated types
        if not isinstance(other, ELM):
            return NotImplemented
        return self._motif_key() == other._motif_key()

    def __hash__(self):
        # Equality and hash use the same key, which sets need. The matched
        # residues and the end are left out on purpose: a match that a point
        # substitution changes, but that still starts in the same place, is the
        # same motif. A start moved by an indel would count as a different one.
        return hash(self._motif_key())


def _residue_from_hgvs(code: str, hgvs_notation: str) -> str:
    """One-letter code of a residue written in HGVS notation, one- or three-letter."""
    residue = _HGVS_THREE_LETTER.get(code) if len(code) == 3 else code.upper()
    if residue not in _STANDARD_RESIDUES:
        raise SparrowException(
            f"{hgvs_notation!r}: {code!r} is not one of the 20 standard amino acids. Only substitutions between "
            "standard amino acids are supported (not stop codons, '=' or unknown residues)."
        )
    return residue


def _parse_substitution(hgvs_notation: str) -> Tuple[int, str, str]:
    """Read an HGVS protein substitution into (0-indexed position, wild type, mutant)."""
    match = _HGVS_SUBSTITUTION.match(hgvs_notation)
    if match is None:
        raise SparrowException(
            f"{hgvs_notation!r} is not an HGVS protein substitution. Write it as 'p.S7A' or 'p.Ser7Ala' (1-indexed "
            "position), optionally in parentheses as 'p.(Ser7Ala)'."
        )
    _, wild_type_code, position_text, mutant_code = match.groups()
    position = int(position_text)
    if position < 1:
        raise SparrowException(f"{hgvs_notation!r}: HGVS positions start at 1, not {position}")
    wild_type = _residue_from_hgvs(wild_type_code, hgvs_notation)
    mutant = _residue_from_hgvs(mutant_code, hgvs_notation)
    return position - 1, wild_type, mutant


def parse_hgvs(hgvs_notation: str) -> Tuple:
    """Read a protein point substitution written in HGVS notation.

    Parameters
    ----------
    hgvs_notation : str
        A substitution in HGVS protein notation with a 1-indexed position, in
        one-letter (``'p.S7A'``) or three-letter (``'p.Ser7Ala'``) codes,
        optionally in the parentheses HGVS uses for a predicted consequence
        (``'p.(Ser7Ala)'``). Both residues must be standard amino acids.

    Returns
    -------
    tuple of (int, str)
        The position, converted to 0-indexed, and the mutant residue as a
        one-letter code. The wild-type residue is checked against the sequence
        by :func:`compute_lost_elms` and its relatives, which know it.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the notation is not a substitution in that form, its position is
        below 1, or either residue is not a standard amino acid (a stop codon,
        ``=`` or an unknown residue).

    Examples
    --------
    >>> parse_hgvs("p.S7A")
    (6, 'A')
    >>> parse_hgvs("p.Ser7Val")
    (6, 'V')
    """
    position, _, mutant = _parse_substitution(hgvs_notation)
    return position, mutant


def read_elm_table(file: str) -> Tuple[list, list]:
    """Parse an ELM class TSV into a header and a list of rows.

    Parameters
    ----------
    file : str
        Path to an ``elm_classes.tsv``. The latest one can be downloaded from
        http://elm.eu.org/downloads.html

    Returns
    -------
    tuple
        ``(columns, rows)`` where ``columns`` is the list of field names and
        ``rows`` is a list of lists of strings.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the expected header row is missing.
    """
    elm_data = []
    columns = None
    with open(f"{file}", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if line.startswith('"Accession"'):
                columns = line.strip().split("\t")
                columns = [col.replace('"', "") for col in columns]
            else:
                elm_data.append(line.replace('"', "").strip().split("\t"))

    if columns is None:
        raise SparrowException(
            f'Could not find the expected header row (starting with "Accession") '
            f"in the ELM class file {file}. The latest elm_classes.tsv can be "
            "downloaded from http://elm.eu.org/downloads.html"
        )

    return columns, elm_data


def generate_elm_df(file: str):
    """Read an ELM class TSV into a pandas DataFrame, one row per ELM class.

    Parameters
    ----------
    file : str
        Path to an ``elm_classes.tsv`` - for example the bundled one,
        ``sparrow.data_path("elm_classes.tsv")``. The latest one can be downloaded
        from http://elm.eu.org/downloads.html

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the elm annotations.

    See Also
    --------
    read_elm_table : The same data as plain lists, without building a DataFrame.
    """
    import pandas as pd

    columns, elm_data = read_elm_table(file)
    return pd.DataFrame(elm_data, columns=pd.Index(columns))


@functools.lru_cache(maxsize=1)
def _elm_class_records() -> tuple:
    """Return the bundled ELM class table as a cached tuple of records.

    ``find_all_elms`` is called once per :class:`sparrow.Protein`, and the
    lost/gained/retained helpers build a second Protein for the mutant, so the
    class table would otherwise be re-read and re-parsed on every call. Parsing
    it once and caching the immutable records keeps repeated ELM queries cheap.

    Returns
    -------
    tuple
        Tuple of ``(regex, identifier, functional_site_name, description,
        probability)`` tuples, one per ELM class.
    """
    columns, rows = read_elm_table(sparrow.data_path("elm_classes.tsv"))
    index = {name: position for position, name in enumerate(columns)}

    records = []
    for row in rows:
        if len(row) < len(columns):
            continue

        # the TSV stores everything as strings; coerce so ELM.probability really
        # is the float its annotation promises
        try:
            probability = float(row[index["Probability"]])
        except (TypeError, ValueError):
            probability = float("nan")

        records.append(
            (
                row[index["Regex"]],
                row[index["ELMIdentifier"]],
                row[index["FunctionalSiteName"]],
                row[index["Description"]],
                probability,
            )
        )

    return tuple(records)


def _match_spans(regex: str, sequence: str) -> list:
    """(start, end) of every match of a pattern, overlapping matches included.

    :func:`re.finditer` alone reports non-overlapping matches, so a match that
    starts inside an earlier one is never seen. Searching for the pattern inside a
    lookahead instead tries every start position: the lookahead consumes nothing,
    and its group (group 1, since it opens first) holds the match starting there.
    """
    return [(match.start(1), match.end(1)) for match in re.finditer(f"(?=({regex}))", sequence)]


def find_all_elms(sequence: str) -> Set[ELM]:
    """Every match of every bundled ELM class in a sequence.

    Each class's regular expression is tried at every start position, so matches
    of the same class that overlap are all reported - one per start position,
    the one the regular expression finds there.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.

    Returns
    -------
    set of ELM
        One :class:`ELM` per match. This is a set, so the matches are unordered.

    Examples
    --------
    >>> motifs = find_all_elms("MKKRPKSPSGSPTSRRPLL")
    >>> sorted(motif.identifier for motif in motifs if motif.start == 0)
    ['DEG_Nend_UBRbox_1', 'TRG_NLS_MonoExtC_3']
    """
    elms = []
    for (
        regex,
        elm_class,
        site,
        elm_description,
        elm_probability,
    ) in _elm_class_records():
        for start, end in _match_spans(regex, sequence):
            elm = ELM(
                regex,
                elm_class,
                site,
                elm_description,
                elm_probability,
                start,
                end,
                sequence[start:end],
            )
            elms.append(elm)
    return set(elms)


def _wild_type_and_mutant(target_protein, query):
    """The wild-type Protein and its point mutant, once the substitution is checked.

    Parameters
    ----------
    target_protein : sparrow.protein.Protein or str
        The wild-type protein, or its sequence.
    query : tuple of (int, str) or str
        ``(position, mutant_residue)`` with a 0-indexed position, or an HGVS
        substitution (see :func:`parse_hgvs`).

    Returns
    -------
    tuple of (sparrow.protein.Protein, sparrow.protein.Protein)
        The wild type and the mutant.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the substitution cannot be read, its position is outside the
        sequence, the mutant residue is not a standard amino acid, or an HGVS
        substitution's wild-type residue is not the one in the sequence.
    """
    if isinstance(target_protein, str):
        target_protein = sparrow.Protein(target_protein)
    sequence = target_protein.sequence

    if isinstance(query, str):
        position, wild_type, mutant = _parse_substitution(query)
    else:
        try:
            position, mutant = query
        except (TypeError, ValueError):
            raise SparrowException(
                f"A substitution is (position, mutant_residue) or an HGVS string such as 'p.S7A', not {query!r}"
            ) from None
        wild_type = None
        if not isinstance(position, numbers.Integral) or isinstance(position, bool):
            raise SparrowException(f"The position of a substitution must be an integer, not {position!r}")
        if not isinstance(mutant, str) or mutant.upper() not in _STANDARD_RESIDUES:
            raise SparrowException(f"The mutant residue must be one of the 20 standard amino acids, not {mutant!r}")
        mutant = mutant.upper()

    if not 0 <= position < len(sequence):
        raise SparrowException(
            f"Substitution {query!r} is at 0-indexed position {position}, outside a sequence of length {len(sequence)}"
        )
    if wild_type is not None and sequence[position] != wild_type:
        raise SparrowException(
            f"{query!r} expects {wild_type} at position {position + 1}, but the sequence has {sequence[position]} there"
        )

    mutant_sequence = sequence[:position] + mutant + sequence[position + 1 :]
    return target_protein, sparrow.Protein(mutant_sequence)


def compute_lost_elms(target_protein, query: Union[Tuple[int, str], str]) -> Set:
    """ELM matches a point substitution destroys.

    Parameters
    ----------
    target_protein : sparrow.protein.Protein or str
        The wild-type protein, or its sequence.
    query : tuple of (int, str) or str
        The substitution: either ``(position, mutant_residue)`` with a 0-indexed
        position, or HGVS notation such as ``'p.S7A'`` or ``'p.Ser7Ala'``, whose
        position is 1-indexed and whose wild-type residue must match the
        sequence (see :func:`parse_hgvs`).

    Returns
    -------
    set of ELM
        The wild-type matches (``Protein.elms``) the mutant does not have.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the substitution cannot be read, lies outside the sequence, names a
        non-standard residue, or names a wild-type residue the sequence does not
        have there.

    Notes
    -----
    Matches are compared as described under :class:`ELM`, so a motif counts as
    lost when no match of its class starts at the same position in the mutant.

    Examples
    --------
    >>> lost = compute_lost_elms("MKKRPKSPSGSPTSRRPLL", "p.S7A")
    >>> sorted(motif.identifier for motif in lost)
    ['DOC_WW_Pin1_4', 'MOD_DYRK1A_RPxSP_1', 'MOD_GSK3_1', 'MOD_ProDKin_1']
    """
    wild_type, mutant = _wild_type_and_mutant(target_protein, query)
    return wild_type.elms - mutant.elms


def compute_gained_elms(target_protein, query: Union[Tuple[int, str], str]) -> Set:
    """ELM matches a point substitution creates.

    Parameters
    ----------
    target_protein : sparrow.protein.Protein or str
        The wild-type protein, or its sequence.
    query : tuple of (int, str) or str
        The substitution, as for :func:`compute_lost_elms`.

    Returns
    -------
    set of ELM
        The mutant's matches the wild type does not have.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        As for :func:`compute_lost_elms`.
    """
    wild_type, mutant = _wild_type_and_mutant(target_protein, query)
    return mutant.elms - wild_type.elms


def compute_retained_elms(target_protein, query: Union[Tuple[int, str], str]) -> Set:
    """ELM matches a point substitution leaves in place.

    Parameters
    ----------
    target_protein : sparrow.protein.Protein or str
        The wild-type protein, or its sequence.
    query : tuple of (int, str) or str
        The substitution, as for :func:`compute_lost_elms`.

    Returns
    -------
    set of ELM
        The matches the wild type and the mutant share. A match whose residues
        change but which still matches at the same start counts as retained.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        As for :func:`compute_lost_elms`.
    """
    wild_type, mutant = _wild_type_and_mutant(target_protein, query)
    return wild_type.elms & mutant.elms
