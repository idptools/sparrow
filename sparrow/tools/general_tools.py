from sparrow.data import amino_acids


def is_valid_protein_sequence(sequence):
    """
    Function that tests if a passed sequence contains non-standard ammino acids

    Parameters
    ----------------
    sequence : str
        Protein sequence

    Returns
    ---------------
    bool
        If sequences contains non-standard amino acids returns False, else returns 
        True
    """
    
    for i in sequence:
        if i not in amino_acids.VALID_AMINO_ACIDS:
            return False

    return True



def compare_sequence(s1, s2, verbose=False, ignore_gaps=False, return_positions=False):
    """
    Function that compares two sequences of the same length and returns 
    either the set of positions where the sequences are different (indxed at 0) or 
    the number of differences between them, depending on the status of the flag 
    return_position. This function Will also print the differences if verbose is 
    set to True.
    
    If ignore_gaps is set to True, will ignore gaps in the comparison (i.e.
    will ignore '-' characters in either sequence). This is useful when running
    analyses for aligned sequences.

    WARNING: Sequence must have the same length - if two passed sequences are not
    identical in terms of length then this function throws a ValueError

    Parameters
    ----------------
    s1 : str
        First sequence to compare

    s2 : str
        Second sequence to compare

    verbose : bool
        If True, will print the differences between the two sequences.
        Default is False

    return_positions : bool
        If True, will return a list of positions where the two sequences 
        differ. If false return the count only.

    Returns
    ---------------
    int
        Number of differences between the two sequences

    Raises
    ---------------
    ValueError
        If sequences are not the same length.

    """

    # first things first check if sequences are the same length and
    # freak out if not!
    if len(s1) != len(s2):
        raise ValueError("Sequences must have the same length")

    # define comparison function based on ignore_gaps
    if ignore_gaps:
        def _compare(p1,p2):
            if p1 == "-" or p2 == "-":
                return False
            elif p1 == p2:
                return False
            else:
                return True
    else:
        def _compare(p1,p2):
            if p1 == p2:
                return False
            else:
                return True
            

    # cycle through each position in the sequence
    positions = []
    for i in range(len(s1)):
        if _compare(s1[i],s2[i]):
            positions.append(i)
            if verbose:
                print(f"{i+1}: {s1[i]} vs. {s2[i]}")


    if return_positions:
        return positions
    else:
        return len(positions)
