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
import re
from dataclasses import dataclass
from typing import Set, Tuple, Union

import sparrow
from sparrow.sparrow_exceptions import SparrowException


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

    def __eq__(self, other):
        # let Python fall back to the reflected operation / identity comparison
        # rather than raising AttributeError on unrelated types
        if not isinstance(other, ELM):
            return NotImplemented

        if self.start > other.end or self.end < other.start:
            return False

        # Only compare regex patterns for equality - all regexes for ELMs are unique - we could also check functional site names?
        return self.regex == other.regex

    def __hash__(self):
        # I THINK this works since we're basically saying we dont CARE about sequences if they're the same or not
        # this will let us do set differences and intersections
        # Does restrict motif to starting at the same position though which we know these could be diff spots from indels
        # This is fine for point mutation comparison, but this could probably be generalized.
        # don't want just look for "in the sequence" because their might be
        # multiple occurences of the same motif and motif positioning may matter.
        return hash((self.regex, self.functional_site_name, self.start))


def parse_hgvs(hgvs_notation: str) -> Tuple:
    """Read a protein point substitution written in HGVS notation.

    Parameters
    ----------
    hgvs_notation : str
        A substitution in one-letter HGVS protein notation, such as ``'p.S7A'``,
        with a 1-indexed position. Only one-letter codes are read correctly: the
        mutant residue is taken to be the last character, so three-letter
        notation such as ``'p.Ser7Val'`` is misread (as ``L``). The wild-type
        residue is not checked against any sequence.

    Returns
    -------
    tuple of (int, str)
        The position, converted to 0-indexed, and the mutant residue in upper
        case.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the notation does not start with ``'p.'``, or its position is below 1.
    ValueError
        If the notation holds no position.

    Examples
    --------
    >>> parse_hgvs("p.S7A")
    (6, 'A')
    """
    if not hgvs_notation.startswith("p."):
        raise SparrowException("Invalid HGVS notation. Must start with 'p.'")

    parts = hgvs_notation.split("p.")
    if len(parts) < 2:
        raise SparrowException("Invalid HGVS notation. Must be in the form p.xxx")

    # Extract the position and amino acids
    position = int("".join(filter(str.isdigit, parts[1])))  # shift indexing to 0
    if position <= 0:
        raise SparrowException(
            f"Invalid position in HGVS notation, must be a 1 indexed integer greater than 0. Received {position}"
        )
    mutation = parts[1][-1]

    # position shifted to 0 index
    return position - 1, mutation.upper()


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
        ``sparrow.get_data("elm_classes.tsv")``. The latest one can be downloaded
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
    columns, rows = read_elm_table(sparrow.get_data("elm_classes.tsv"))
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


def find_all_elms(sequence: str) -> Set[ELM]:
    """Every match of every bundled ELM class in a sequence.

    Each class's regular expression is searched with :func:`re.finditer`, which
    returns *non-overlapping* matches: where two matches of the same class
    overlap, only the first is reported.

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
        match_indices = [(m.start(0), m.end(0)) for m in re.finditer(regex, sequence)]
        for start, end in match_indices:
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


def compute_lost_elms(target_protein, query: Union[Tuple[int, str], str]) -> Set:
    """ELM matches a point substitution destroys.

    Parameters
    ----------
    target_protein : sparrow.protein.Protein or str
        The wild-type protein, or its sequence.
    query : tuple of (int, str) or str
        The substitution: either ``(position, mutant_residue)`` with a 0-indexed
        position, or one-letter HGVS notation such as ``'p.S7A'``, whose position
        is 1-indexed (see :func:`parse_hgvs`).

    Returns
    -------
    set of ELM
        The wild-type matches (``Protein.elms``) the mutant does not have.

    Notes
    -----
    Matches are compared as described under :class:`ELM`. Because
    :func:`find_all_elms` reports only non-overlapping matches of each class, a
    wild-type match can hide an overlapping one; if the substitution removes the
    first, the hidden one is reported as gained by :func:`compute_gained_elms`
    even though the wild type has it too.

    Examples
    --------
    >>> lost = compute_lost_elms("MKKRPKSPSGSPTSRRPLL", "p.S7A")
    >>> sorted(motif.identifier for motif in lost)
    ['DOC_WW_Pin1_4', 'MOD_DYRK1A_RPxSP_1', 'MOD_GSK3_1', 'MOD_ProDKin_1']
    """

    if isinstance(target_protein, str):
        target_protein = sparrow.Protein(target_protein)

    if isinstance(query, str):
        position, mutation = parse_hgvs(query)
    else:
        position, mutation = query

    mutant_protein = sparrow.Protein(
        target_protein.sequence[:position] + mutation + target_protein.sequence[position + 1 :]
    )

    wt_elms = target_protein.elms
    mutant_elms = mutant_protein.elms
    lost_elms = wt_elms - mutant_elms

    return lost_elms


def compute_gained_elms(target_protein, query: Union[Tuple[int, str], str]) -> Set:
    """ELM matches a point substitution creates.

    Parameters
    ----------
    target_protein : sparrow.protein.Protein or str
        The wild-type protein, or its sequence.
    query : tuple of (int, str) or str
        The substitution: either ``(position, mutant_residue)`` with a 0-indexed
        position, or one-letter HGVS notation such as ``'p.S7A'``, whose position
        is 1-indexed (see :func:`parse_hgvs`).

    Returns
    -------
    set of ELM
        The mutant's matches the wild type does not have. See the Notes of
        :func:`compute_lost_elms` for when a match the wild type already has can
        appear here.
    """

    if isinstance(target_protein, str):
        target_protein = sparrow.Protein(target_protein)

    if isinstance(query, str):
        position, mutation = parse_hgvs(query)
    else:
        position, mutation = query

    mutant_protein = sparrow.Protein(
        target_protein.sequence[:position] + mutation + target_protein.sequence[position + 1 :]
    )

    wt_elms = target_protein.elms
    mutant_elms = mutant_protein.elms
    gained_elms = mutant_elms - wt_elms

    return gained_elms


def compute_retained_elms(target_protein, query: Union[Tuple[int, str], str]) -> Set:
    """ELM matches a point substitution leaves in place.

    Parameters
    ----------
    target_protein : sparrow.protein.Protein or str
        The wild-type protein, or its sequence.
    query : tuple of (int, str) or str
        The substitution: either ``(position, mutant_residue)`` with a 0-indexed
        position, or one-letter HGVS notation such as ``'p.S7A'``, whose position
        is 1-indexed (see :func:`parse_hgvs`).

    Returns
    -------
    set of ELM
        The matches the wild type and the mutant share. A match whose residues
        change but which still matches at the same start counts as retained.
    """
    if isinstance(target_protein, str):
        target_protein = sparrow.Protein(target_protein)

    if isinstance(query, str):
        position, mutation = parse_hgvs(query)
    else:
        position, mutation = query

    mutant_protein = sparrow.Protein(
        target_protein.sequence[:position] + mutation + target_protein.sequence[position + 1 :]
    )

    wt_elms = target_protein.elms
    mutant_elms = mutant_protein.elms
    retained_elms = wt_elms & mutant_elms

    return retained_elms
