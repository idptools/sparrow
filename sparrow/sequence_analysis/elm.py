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

    elm_motif = namedtuple("ELM",["regex","functional_site_name","start","end","sequence"])
    elms = []
    for regex in df["Regex"]:
        match_indices = [(m.start(0), m.end(0)) for m in re.finditer(regex, sequence)]
        for (start,end) in match_indices:
            elm = elm_motif(regex, mapper[regex], start, end, sequence[start:end]) 
            elms.append(elm)
    return elms

#elm_motif = namedtuple("ELM",["regex","functional_site_name","start","end","sequence"])

def remove_duplicate_elms(elms1: NamedTuple) -> List[NamedTuple]:
    """This function takes in a list of NamedTuples and pulls only the values 
    that are unique. It returns a NamedTuple

    Parameters
    ----------
    elms1: List[NamedTuple]
        List of Slims and their properties.

    Returns
    -------
    List[NamedTuple]
        A list of NamedTuples all unique elements
    """
    return list(set(elms1))

def get_property_elms(elms1 : NamedTuple, property_list : list) -> List[NamedTuple]:
    """This function takes in a list of NamedTuples and pulls only the values 
    in the feilds specificied by property_list. This returns back a NamedTuple

    Parameters
    ----------
    elms1: List[NamedTuples]
        List of Slims and their properties.
    property_list: List[String]
        A list of the properties in the NamedTuple to be considered when comparing elements. Possible values are 
        "regex", "functional_site_name", "start", "end", "sequence"

    Returns
    -------
    List[NamedTuple]
        A list of NamedTuples containing all the properties specified in property_list
    """
    FilteredTuple = namedtuple('FilteredTuple', property_list)
    
    filtered_list = []
    for t in elms1:
        # Extract values for selected fields using getattr()
        values = tuple(getattr(t, field) for field in property_list)
        # Create a new namedtuple with the selected values
        filtered_tuple = FilteredTuple(*values)
        filtered_list.append(filtered_tuple)
    return filtered_list

def difference_elms(elms1 : NamedTuple, elms2 : NamedTuple) -> List[NamedTuple]:
    """This function takes two input proteins elms as a list of NamedTuples and returns a  list of NamedTuple
    containing the differences in elms from the first protein to the second protein (non symmetric difference).

    Parameters
    ----------
    elms1 : List[NamedTuple]
        A list of NamedTuples containing elms to subtract from
    elms2 : List[NamedTuple]
        A list of NamedTuples containing elms to subtract

    Returns
    -------
    List[NamedTuple]
        A list of NamedTuples containing the non symetric difference fo the two sets of elms
    """
    #convert the input lists into sets
    s1 = set(elms1)
    s2 = set(elms2)



    #take the non symetric difference of the sets
    sdif = s1.difference(s2)

    #convert the set back to a list and return the value
    return list(sdif)

def sym_difference_elms(elms1 : NamedTuple, elms2 : NamedTuple) -> List[NamedTuple]:
    """This function takes two input proteins elms as a list of NamedTuples and returns a  list of NamedTuple
    containing the differences in elms from the first protein to the second protein (symmetric difference).
    The order does not matter that you pass the elms. Either order will return the same values.

    Parameters
    ----------
    elms1 : List[NamedTuple]
        A list of NamedTuples containing elms to subtract
    elms2 : List[NamedTuple]
        A list of NamedTuples containing elms to subtract

    Returns
    -------
    List[NamedTuple]
        A list of NamedTuples containing the symetric difference fo the two sets of elms
    """
    #convert the input lists into sets 
    s1 = set(elms1)
    s2 = set(elms2)

    #take the symetric difference of the sets
    sdif = s1.symmetric_difference(s2)

    #convert the set back to a list and return the value
    return list(sdif)

def common_elms(elms1 : NamedTuple, elms2 : NamedTuple) -> List[NamedTuple]:
    """This function takes two input proteins elms as a list of NamedTuples and returns a  list of NamedTuple
    containing all elements that are common in the two inputs elms1 and elms2

    Parameters
    ----------
    elms1 : List[NamedTuple]
        A list of NamedTuples containing elms 
    elms2 : List[NamedTuple]
        A list of NamedTuples containing elms

    Returns
    -------
    List[NamedTuple]
        A list of NamedTuples containing the common elements in elms1 and elms2
    """
    #convert the input lists into sets 
    s1 = set(elms1)
    s2 = set(elms2)

    #take the intersection of the two sets
    sintersection = s1 & s2

    #convert the set back to a list and return the value
    return list(sintersection)

def combine_elms(elms1 : NamedTuple, elms2 : NamedTuple) -> List[NamedTuple]:
    """This function takes two input proteins elms as a list of NamedTuples and returns a  list of NamedTuple
    containing all elements from the two input elms

    Parameters
    ----------
    elms1 : List[NamedTuple]
        A list of NamedTuples containing elms to combine
    elms2 : List[NamedTuple]
        A list of NamedTuples containing elms to combine

    Returns
    -------
    List[NamedTuple]
        A list of NamedTuples containing the union of the two sets of elms
    """
    #convert the input lists into sets 
    s1 = set(elms1)
    s2 = set(elms2)

    #take the union of the two sets
    sunion = s1 | s2

    #convert the set back to a list and return the value
    return list(sunion)

def is_subset_elms(elmssub : NamedTuple, elmssuper : NamedTuple) -> bool:
    """This function takes two input proteins elms as a list of NamedTuples and returns a boolean determining
    if elmssub is a subset of elmssuper.

    Parameters
    ----------
    elmssub : List[NamedTuple]
        A list of NamedTuples containing elms to be tested for subsethood of elssuper
    elmssuper : List[NamedTuple]
        A list of NamedTuples containing elms to be tested for supersethood of elmssub

    Returns
    -------
    bool
        Indicates if elmssub is a subset of elmssuper
    """
    #convert the input lists into sets 
    s1 = set(elmssub)
    s2 = set(elmssuper)


    #determin if elmssub is a subset of elmssuper
    return s1.issubset(s2)