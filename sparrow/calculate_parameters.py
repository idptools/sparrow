from sparrow.data import amino_acids
import numpy as np
import math
from . import sparrow_exceptions

# .................................................................
#
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
 


def calculate_seg_complexity(s, alphabet=amino_acids.VALID_AMINO_ACIDS):
    """
    Function to calculate the Wootton-Federhen complexity of a sequence (also called
    seg complexity, as this the theory used in the classic SEG algorithm.

    Parameters
    -----------
    s : str
        Amino acid sequence

    alphabet : list
        List of amino acids found in alphabet. Note this does not sanity check in the 
        case of non-standard amino acids. Default is the standard 20 amino acids

    Returns
    ----------
    float
        Returns a float that corresponds to the compositional complexity associated with 
        the passed sequence.

    """

    alphabet_size = len(alphabet)
    seq_len = len(s)

    complexity = 0 
    for a in alphabet:
        p = s.count(a)/seq_len

        if p > 0:
            complexity = p * math.log(p, alphabet_size) + complexity  

    return -complexity
    
   

# .................................................................
#
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
    

# .................................................................
#
def calculate_linear_hydrophobicity(s, mode='KD', normalize=False):
    """
    Compute linear hydrophobicity from sequence using one of several possible 
    hydrophobicity scales. 

    By default this is Kyte-Doolitle, but, we'll add in additional scales
    as/when needed.

    Parameters:
    --------------
    s : str
         Amino acid sequence

    mode : str
        Selector for hydrophobicity table. Options available are

        'KD'    | Kyte-Doolittle
        "ab"    | Aboderin
        'al'    | Abraham and Leo
        'ag'    | Argos
        'bm'    | Black and Mould
        'bb'    | Bull and Breese
        'cs'    | Casari and Sippl
        'ci'   | Cid
        'cw3.4' | Cowan3.4
        'cw7.5' | Cowan7.5
        'es'    | Eisenberg
        'eg'    | Engelman
        'fs'    | Fasman
        'fc'    | Fauchere
        'gd'    | Goldsack
        'guy'   | Guy
        'jo'    | jones
        'ju' | Juretic
        'ki'    | Kidera
        'mi' |  Miyazawa
        'pa' | parker
        'po' | Ponnuswamy
        'ro' | Rose
        'rm' | Roseman
        'sw' | Sweet
        'ta' | tanford
        'wi' | wilson
        'zi' | zimmerman


    normalize : bool
        Boolean that means hydrophobicity scales operate on a normalized
        dynamic range of 0 to 1. This only works for Kyle-Doolittle scale.

    Returns:
    ------------
    list 
        List of values that correspond to per-residue hydrophobicity based on
        a given hydrophobicity scale.
    
    """
    if mode == 'KD':
        try:
            if normalize:
                return [amino_acids.HYDROPHOBICITY_SCALES["normalized_kd"][r] for r in s]
            else:
                return [amino_acids.HYDROPHOBICITY_SCALES["scaled_kd"][r] for r in s]
        except KeyError:
            raise sparrow_exceptions.CalculationException('Invalid residue found in %s' %(s))
    elif mode == "cw3.4":
        cw3_4 = amino_acids.HYDROPHOBICITY_SCALES["cw"][3.4]
        return [cw3_4[r] for r in s]
    elif mode == "cw7.5":
        cw7_5 = amino_acids.HYDROPHOBICITY_SCALES["cw"][7.5]
        return [cw7_5[r] for r in s]
    else:
        try:
            return [amino_acids.HYDROPHOBICITY_SCALES[mode][r] for r in s]
        except KeyError:
            raise sparrow_exceptions.CalculationException('Invalid mode passed: %s' %(mode))
