# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
import numpy as np
cimport numpy as np
from cython.view cimport array
from libc.math cimport sqrt,abs, fabs
from sparrow.data.amino_acids import VALID_AMINO_ACIDS
from sparrow.sparrow_exceptions import SparrowException

# Define a typed memoryview for efficient access to numpy arrays
ctypedef np.float64_t DOUBLE_t
ctypedef np.int64_t INT64_t

cdef dict DEFAULT_HYDRO_DICT = {'A': 0.730, 'R': 0.000, 'N': 0.432, 'D': 0.378, 'C': 0.595, 'Q': 0.514, 'E': 0.459,
                      'G': 0.649, 'H': 0.514, 'I': 0.973, 'L': 0.973, 'K': 0.514, 'M': 0.838, 'F': 1.000,
                      'P': 1.000, 'S': 0.595, 'T': 0.676, 'W': 0.946, 'Y': 0.865, 'V': 0.892}


cpdef double compute_scd_x(str sequence, group1=['E','D'], group2=['R','K']):
    cdef int m, n, seqlen
    cdef double total, m_val, n_val, charge_val, final_val
    cdef int cur_m_charge, cur_n_charge
    cdef char cur_m_res, cur_n_res

    # Pre-calculate group membership
    cdef int[:] group_membership = np.zeros(256, dtype=np.int32)
    for residue in group1:
        group_membership[ord(residue)] = -1
    for residue in group2:
        group_membership[ord(residue)] = 1

    total = 0
    seqlen = len(sequence)

    # Convert sequence to array of integers
    cdef int[:] sequence_array = np.array([ord(char) for char in sequence], dtype=np.int32)
        
    for m in range(1, seqlen):
        m_val = m + 1
            
        for n in range(0, m-1):
            n_val = n + 1

            # Access residues using array indexing
            cur_m_res = sequence_array[m]
            cur_n_res = sequence_array[n]
            
            # Retrieve group charge
            cur_m_charge = group_membership[cur_m_res]
            cur_n_charge = group_membership[cur_n_res]

            charge_val = cur_m_charge * cur_n_charge
            final_val = charge_val * sqrt(m_val - n_val)
            total += final_val

    return total / seqlen

cdef validate_sequence(str seq, dict hydro_dict):
    cdef set all_res = set(seq)
    for res in all_res:
        if res not in hydro_dict:
            raise ValueError(f'When calculating SHD the hydrophobicity dictionary lacked the residue {res}')

cpdef double compute_shd(str seq, dict hydro_dict=None):
    """
    Function takes in a sequence and returns Sequence 
    Hydropathy Decoration (SHD), IE. patterning of hydrophobic 
    residues in the sequence. This is computed as defined in ref 1
    
    As an optional parameter this function can take in a predefined 
    hydropathy conversion dictionary for the amino acids, where the keys 
    are Amino acids and values are floats.
    
    If a conversion dict is not provided the following conversion is used:

    'A': 0.730,
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
    'V': 0.892,

    These are the Kyte Doolitle normalized hydrophobicity. 

    Parameters
    ------------
    seq : str 
        Amino acid sequence passed as string

    hydro_dict : dict
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
    if hydro_dict is None:
        hydro_dict = DEFAULT_HYDRO_DICT

    validate_sequence(seq, hydro_dict)

    cdef Py_ssize_t N = len(seq)
    cdef double[:] h = np.array([hydro_dict[res] for res in seq], dtype=np.double)
    cdef double t = 0.0
    cdef Py_ssize_t m, n
    
    for m in range(1, N):
        for n in range(m):
            t += (h[m] + h[n]) / abs(m - n)

    return t / N

    