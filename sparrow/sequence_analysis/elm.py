import pandas as pd 
from collections import namedtuple
import re
from typing import List, NamedTuple
import sparrow

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
                elm_data.append(line.strip().split("\t"))
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

    elm_motif = namedtuple("ELM",["regex","functional_site_name","start","end","sequence"])
    elms = []
    for regex in df["Regex"]:
        match_indices = [(m.start(0), m.end(0)) for m in re.finditer(regex, sequence)]
        for (start,end) in match_indices:
            elm = elm_motif(regex, mapper[regex], start, end, sequence[start:end]) 
            elms.append(elm)
    return elms


