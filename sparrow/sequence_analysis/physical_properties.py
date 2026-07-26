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

    The weight is the sum of the free-residue molecular weights minus one water
    molecule (18 Da) per peptide bond, i.e. one fewer than the number of
    residues.

    Parameters
    -------------
    sequence : str
        String containing the amino acid sequence (upper case one-letter residue codes)

    Returns
    -----------
    float
        Returns the residue or polypeptide molecular weight in Daltons (Da). An
        empty sequence returns 0.0.

    """

    # an empty sequence has no residues and no peptide bonds; handled explicitly
    # because the peptide-bond correction below would otherwise ADD one water
    if len(sequence) == 0:
        return 0.0

    # compute niave MW
    MW = 0
    for i in sequence:
        MW = MW + amino_acids.AA_MOLECULAR_WEIGHT[i]

    # subtract one water per peptide bond (there are len(sequence)-1 of them,
    # so a single residue is returned as-is)
    return MW - 18*(len(sequence)-1)
        
