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
        if i not in amino_acids:
            return False

    return True
        
    
