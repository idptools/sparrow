from sparrow.data import amino_acids 

## The physical properties module contains stateless functions that compute sequence-dependent
## physical properties. See the "calculate_molecular_weight" function as a template for how
## these functions should work.
##
##

def calculate_molecular_weight(sequence):
    """
    Function that returns the molecular weight of a protein sequence assuming standard 
    amino acid molecular weights.

    Parameters
    -------------
    sequence : str
        String containing the amino acid sequence (upper case one-letter residue codes)

    Returns
    -----------
    float
        Returns the residue or polypeptide molecular weight. 

    """

    # compute niave MW
    MW = 0
    for i in sequence:
        MW = MW + amino_acids.AA_MOLECULAR_WEIGHT[i]

    if len(sequence) == 1:
        return MW

    else:
        return MW - 18*(len(sequence)-1)
        
