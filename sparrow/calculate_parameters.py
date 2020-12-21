from sparrow.data import amino_acids




def calculate_aa_fractions(s):
    """
    Standalone function that computes amino-acid fractions for
    a given sequence.

    Parameters:
    --------------
    s : str
        Amino acid sequence

    Returns
    ---------------
    dict
        Returns dictionary with per-residue amino acid fraction
    
    """
    
    aa_dict = {}
    for i in amino_acids.VALID_AMINO_ACIDS:
        aa_dict[i] = 0

    for i in s:
        aa_dict[i] = aa_dict[i] + 1

    
    len_s = len(s)
    for i in amino_acids.VALID_AMINO_ACIDS:
        aa_dict[i] = aa_dict[i]/len_s

    return aa_dict
    

        
def calculate_disorder(s):
    """
    Standalone function that uses Metapredict to calculate the per-residue 


    """
    # local import
    from metapredict import meta

    return meta.predict_disorder(s)


def calculate_hydrophobicity(s, mode='KD', normalize=False):
    """
    Standalone function that computes hydrophobicity 

    Parameters:
    --------------
    s : str
        Amino acid sequence

    mode : str 
        Hydrophobicity mode to be used. Currently only KD supported
        but can be expanded. Allowed values: 'KD'

    normalize : Bool
        If set to True hydrophobicity scales are normalized to be between 0
        and 1. Default = False.

    Returns
    ---------------
    Float
        Returns a floating point value with the mean hydrophobicity 
        as defined based on the passed scale

    """
    return np.mean(calculate_linear_hydrophobicity(s, mode, normalize))
    


def calculate_linear_hydrophobicity(s, mode='KD', normalize=False):

    if mode == 'KD':
        try:
            if normalize:
                return [amino_acids.AA_hydro_KD_normalized[r] for r in s]
            else:
                return [amino_acids.AA_hydro_KD[r] for r in s]
        except KeyError:
            raise CalculationException('Invalid residue found in %s' %(s))
    else:
        raise CalculationException('Invalid mode passed: %s' %(mode))
