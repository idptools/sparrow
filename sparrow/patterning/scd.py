import numpy as np
from sparrow.data.amino_acids import VALID_AMINO_ACIDS
from sparrow.sparrow_exceptions import SparrowException


def compute_scd_x(sequence, group1=['E','D'], group2=['R','K']):
    """
    Backend function for computing the sequence charge decoration
    (SCD) parameter, generalized to two residue groups. The default
    behavior reproduces the standard SCD parameter (see Sawle & Ghosh),
    but users can pass different groups of residues to group1 and 
    group2.

    Parameters
    ------------
    sequence : str 
        Amino acid sequence passed as string

    group1 : list
        List of residues to include in the first group. By default
        this is set to negative residues (['E','D']).

    group2 : list
        List of residues to include in the first group. By default
        this is set to positive residues (['R','K']).

    Returns
    -----------
    float
        Returns a floating point value that reports on the sequence
        charge decoration. This in principle should be a positive
        number.
    

    Reference
    -------------
    Sawle, L., & Ghosh, K. (2015). A theoretical method to compute 
    sequence dependent configurational properties in charged polymers 
    and proteins. The Journal of Chemical Physics, 143(8), 085101.
        
    """

    total = 0
    seqlen = len(sequence)
        
    for m in range(1, seqlen):
            
        m_val = m + 1
            
        for n in range(0, m-1):
                
            n_val = n+1

            cur_m_res = sequence[m]
            cur_n_res = sequence[n]
            
            if cur_m_res in group1:
                cur_m_charge = -1
                    
            elif cur_m_res in group2:
                cur_m_charge = 1
                    
            else:
                cur_m_charge = 0

            if cur_n_res in group1:
                cur_n_charge = -1
                    
            elif cur_n_res in group2:
                cur_n_charge = 1
                    
            else:
                cur_n_charge = 0

            charge_val = cur_m_charge * cur_n_charge
            
            final_val = charge_val * (np.sqrt((m_val)-(n_val)))

            total = total + final_val

    return total/seqlen



def __compute_scd_alternative(seq):
    """
    This is an alternative implementation of calculating the SCD. As best I can
    tell its performance is almost indistuguishable from the function above,
    but I've left it in because it more faithfully matches the original
    mathematical definition.

    Parameters
    ------------
    sequence : str 
        Amino acid sequence passed as string

    Returns
    -----------
    float
        Returns a floating point value that reports on the sequence
        charge decoration. This in principle should be a positive
        number.
    

    Reference
    -------------
    Sawle, L., & Ghosh, K. (2015). A theoretical method to compute 
    sequence dependent configurational properties in charged polymers 
    and proteins. The Journal of Chemical Physics, 143(8), 085101.

    """
    
    # charge converision dictionary 
    charge_dict = {'K':-1,'R':-1, 'E':1, 'D':1}
    
    # number of residues
    N = len(seq)
    
    # charge vector 
    q = [charge_dict[i] if i in charge_dict else 0 for i in seq]
    
    t = 0
    # double sumation to compute t 
    for m in range(1,N):
        for n in range(0,m-1):
            t = t + (q[m]*q[n])*np.power((m-n), 0.5)
    # divde t by N             
    SCD=t/N
    
    return SCD

def compute_shd(seq, hydro_dict=False):
    """
    Function takes in a sequence and returns Sequence 
    Hydropathy Decoration (SHD), IE. patterning of hydrophobic 
    residues in the sequence. This is computed as define in ref 1
    
    As as an optional parameter this function can take in a predefined 
    hydropathy conversion dictionary for the amino acids, where the keys 
    are Amino acids and values are floats.
    
    If a conversion dict is not provided the following conversion is used:

    'A': 0.730
    'R': 0.000, 
    'N': 0.432, 
    'D': 0.378, 
    'C': 0.595, 
    'Q': 0.514, 
    'E': 0.459,
    'G': 0.649, 
    'H': 0.514, 
    'I': 0.973, 
    'L': 0.973, 
    'K': 0.514, 
    'M': 0.838, 
    'F': 1.000,
    'P': 1.000, 
    'S': 0.595, 
    'T': 0.676, 
    'W': 0.946, 
    'Y': 0.865, 
    'V': 0.892

    These are the Kyte Doolitle normalized hydrophobicity. 

    Parameters
    ------------
    sequence : str 
        Amino acid sequence passed as string

    hydro : dict
        Dictionary that maps amino acid to hydrophobicity score
        (optional).

    Returns
    -----------
    float
        Returns a floating point value that reports on the sequence
        hydropathy decoration. This in principle should be a positive
        number.
        
    References
    --------------
    [1] Zheng, W., Dignon, G. L., Brown, M., Kim, Y. C. & Mittal, J. Hydropathy Patterning
    Complements Charge Patterning to Describe Conformational Preferences of Disordered 
    Proteins. J. Phys. Chem. Lett. (2020). doi:10.1021/acs.jpclett.0c00288

    
    """
    
    # define Hydropathy converision dictionary if not provided 
    if not hydro_dict: 
        hydro_dict = {'A': 0.730, 'R': 0.000, 'N': 0.432, 'D': 0.378, 'C': 0.595, 'Q': 0.514, 'E': 0.459,
                      'G': 0.649, 'H': 0.514, 'I': 0.973, 'L': 0.973, 'K': 0.514, 'M': 0.838, 'F': 1.000,
                      'P': 1.000, 'S': 0.595, 'T': 0.676, 'W': 0.946, 'Y': 0.865, 'V': 0.892}

    all_res = list(set(seq))
    for i in all_res:
        if i not in hydro_dict:
            raise SparrowException(f'When calculating SHD the hydrophobicity dictionary lacked the residue {i}')
        
    # number of residues
    N = len(seq)
    
    # Hydropathy vector - get per-residue hydrophobicity
    h = [hydro_dict[i] for i in seq]
    
    t = 0
    # double sumation to compute h 
    for m in range(1, N):
        for n in range(0, m-1):
            t = t + (h[m]+h[n])*(1 /np.absolute(m-n))
                
    # divde h by N             
    SHD = t/N
    
    return SHD 
