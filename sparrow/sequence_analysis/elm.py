import functools
import re
from dataclasses import dataclass
from typing import Set, Tuple, Union

import sparrow
from sparrow.sparrow_exceptions import SparrowException


@dataclass(frozen=True)
class ELM:
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
    """This function takes an HGVS notation and returns a tuple of the form (position, mutation)
    where position is the position of the mutation and mutation is the amino acid change.

    Parameters
    ----------
    hgvs_notation : str
        HGVS notation of the form p.XXXX

    Returns
    -------
    Tuple[int,str]
        Tuple containing the position of the mutation and the amino acid change.
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
    """Generates a pandas DataFrame object containing all the information
    annotated as an elm.

    Parameters
    ----------
    file : str
        This generates a dataframe from the elm_classes.tsv in the data directory.
        The latest elm class list can be found at http://elm.eu.org/downloads.html

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
    """This function takes an input sequence and returns the set of ELM
    annotations found in it. Each entry records the regex used to find the ELM,
    its identifier and functional annotation, its description and probability,
    the start and stop position, and the matched subsequence.

    Parameters
    ----------
    sequence : str
        Amino Acid Sequence

    Returns
    -------
    Set[sparrow.sequence_analysis.elm.ELM]
        A set of :class:`ELM` objects for all possible elms in a given sequence.
        Note this is a set (not a list), so entries are unordered.
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
    """This function takes a protein sequence and a target query and returns a
    the set of ELMs that were lost due to the mutation. The query can either be
    a list or tuple of the form (position, mutant) where position is the position
    of the mutation. or it can be a string in the HGVS format.
    Parameters
    ----------
    target_protein : Union[sparrow.Protein, str]
        sparrow.Protein or amino acid sequence
    queries : Union[str, List[int,str], Tuple[int,str]]
        List or tuple of the form (position, mutant) where position is the position of the mutation.
    Returns
    -------
    Set
        A set of ELMs containing the functional site name, the start and stop position,
        the sequence of the elm.
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
    """This function takes a protein sequence and a target query and returns a
    the set of ELMs that were gained due to the mutation. The query can either be
    a list or tuple of the form (position, mutant) where position is the position
    of the mutation. or it can be a string in the HGVS format.

    Parameters
    ----------
    target_protein : Union[sparrow.Protein, str]
        sparrow.Protein or amino acid sequence
    queries : Union[List[int,str], Tuple[int,str]]
        List or tuple of the form (position, mutant) where position is the position of the mutation.
    Returns
    -------
    Set
        A set of ELMs containing the functional site name, the start and stop position,
        the sequence of the elm.
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
    """This function takes a protein sequence and a target query and returns a
    the set of ELMs that were retained (no change) after mutation. The query can
    either be a list or tuple of the form (position, mutant) where position is
    the position of the mutation. or it can be a string in the HGVS format.

    Parameters
    ----------
    target_protein : Union[sparrow.Protein, str]
        sparrow.Protein or amino acid sequence
    queries : Union[List[int,str], Tuple[int,str]]
        List or tuple of the form (position, mutant) where position is the position of the mutation.
    Returns
    -------
    Set
        A set of ELMs containing the functional site name, the start and stop position,
        the sequence of the elm.
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
