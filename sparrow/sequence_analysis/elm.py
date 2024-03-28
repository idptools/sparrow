import pandas as pd 
from sparrow.sparrow_exceptions import SparrowException
from collections import namedtuple
import re
from typing import List, NamedTuple, Union
import sparrow
from dataclasses import dataclass



@dataclass(frozen=True)
class ELM:
    regex: str
    functional_site_name: str
    start: int
    end: int
    sequence: str

    def __eq__(self, other):
        if self.start > other.end or self.end < other.start:
            return False

        # Only compare regex patterns for equality
        return self.regex == other.regex  

        # regex_pattern = re.compile(self.regex)
        # return bool(regex_pattern.match(self.sequence)) and bool(regex_pattern.match(other.sequence))

    def __hash__(self):
        # I THINK this works since we're basically saying we dont CARE about sequences if they're the same or not
        # this will let us do set differences and intersections?
        return hash((self.regex, self.functional_site_name, self.start))
        

def parse_hgvs(hgvs_notation):
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
    
    parts = hgvs_notation.split('p.')
    if len(parts) < 2:
        raise SparrowException("Invalid HGVS notation. Must be in the form p.xxx")

    # Extract the position and amino acids
    position = int(''.join(filter(str.isdigit, parts[1])))
    mutation = parts[1][-1]

    return tuple(position, mutation)

def generate_elm_df(file : str) -> pd.DataFrame:
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
        
    """
    elm_data = []
    with open(f"{file}", "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            elif line.startswith('"Accession"'):
                columns = line.strip().split("\t")
                columns = [col.replace('"','') for col in columns]
            else: 
                elm_data.append(line.replace('"','').strip().split("\t"))
    df = pd.DataFrame(elm_data,columns=columns)
    return df

def find_all_elms(sequence : str) -> List[NamedTuple]:
    """This function takes an input sequence and returns a namedtuple
    containing the regex used to find the elm from sequence, it's functional annotation,
    the start and stop position, as well as the sequence of the e

    Parameters
    ----------
    sequence : str
        Amino Acid Sequence

    Returns
    -------
    List[NamedTuple]
        A list of NamedTuples containing all possible elms in a given sequence.
    """
    elm_file = sparrow.get_data("elm_classes.tsv")
    df = generate_elm_df(elm_file)
    mapper = {regex : site for regex, site in zip(df["Regex"],df["FunctionalSiteName"])}

    elms = []
    for regex in df["Regex"]:
        match_indices = [(m.start(0), m.end(0)) for m in re.finditer(regex, sequence)]
        for (start,end) in match_indices:
            elm = ELM(regex, mapper[regex], start, end, sequence[start:end]) 
            elms.append(elm)
    return set(elms)

def compute_lost_elms(target_protein, query):
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
    List[NamedTuple]
        A list of NamedTuples containing the functional site name, the start and stop position,
        the sequence of the elm, and the mutation that occurs in the elm.
    """
    
    if isinstance(target_protein, str):
        target_protein = sparrow.Protein(target_protein)
    
    if isinstance(query, str):
        position, mutation = parse_hgvs(query)
    else:
        position, mutation = query

    mutant_protein = sparrow.Protein(target_protein.sequence[:position] + mutation + target_protein[position+1:])
    
    wt_elms = target_protein.elms
    mutant_elms = mutant_protein.elms
    lost_elms = wt_elms - mutant_elms

    return lost_elms

def compute_gained_elms(target_protein, query):
    """This function takes a protein sequence and a list of mutations and returns
    a list of NamedTuples containing the functional site name, the start and stop position,
    the sequence of the elm, and the mutation that occurs in the elm.

    Parameters
    ----------
    target_protein : Union[sparrow.Protein, str]
        sparrow.Protein or amino acid sequence
    queries : Union[List[int,str], Tuple[int,str]]
        List or tuple of the form (position, mutant) where position is the position of the mutation.
    Returns
    -------
    List[NamedTuple]
        A list of NamedTuples containing the functional site name, the start and stop position,
        the sequence of the elm, and the mutation that occurs in the elm.
    """
    
    if isinstance(target_protein, str):
        target_protein = sparrow.Protein(target_protein)
    
    if isinstance(query, str):
        position, mutation = parse_hgvs(query)
    else:
        position, mutation = query

    mutant_protein = sparrow.Protein(target_protein.sequence[:position] + mutation + target_protein[position+1:])
    
    
    wt_elms = target_protein.elms
    mutant_elms = mutant_protein.elms
    gained_elms = mutant_elms - wt_elms
    
    return gained_elms

def compute_retained_elms(target_protein, query):
    """This function takes a protein sequence and a list of mutations and returns
    a list of NamedTuples containing the functional site name, the start and stop position,
    the sequence of the elm, and the mutation that occurs in the elm.

    Parameters
    ----------
    target_protein : Union[sparrow.Protein, str]
        sparrow.Protein or amino acid sequence
    queries : Union[List[int,str], Tuple[int,str]]
        List or tuple of the form (position, mutant) where position is the position of the mutation.
    Returns
    -------
    List[NamedTuple]
        A list of NamedTuples containing the functional site name, the start and stop position,
        the sequence of the elm, and the mutation that occurs in the elm.
    """
    if isinstance(target_protein, str):
        target_protein = sparrow.Protein(target_protein)

    if isinstance(query, str):
        position, mutation = parse_hgvs(query)
    else:
        position, mutation = query

    mutant_protein = sparrow.Protein(target_protein.sequence[:position] + mutation + target_protein[position+1:])
    
    wt_elms = target_protein.elms
    mutant_elms = mutant_protein.elms
    retained_elms = wt_elms & mutant_elms
    
    return retained_elms