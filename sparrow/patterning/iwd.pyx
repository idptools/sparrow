# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
import numpy as np
cimport cython
cimport numpy as np
from libc.math cimport abs, fabs

# Define a typed memoryview for efficient access to numpy arrays
ctypedef np.float64_t DOUBLE_t
ctypedef np.int64_t INT64_t


@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_average_inverse_distance_from_sequence(str sequence, str target_residues):
    """
    Function that takes an amino acid sequence and a set of target residues and
    computes the inverse-weighted distance (IWD) between residues in the 
    target_residue group.

    Parameters
    ----------------
    sequence : str
        Valid amino acid sequence

    target_residues : list
        List that defines the residues of interest i.e. residues here
        are the "1" in the binary mask used to perform IWD clustering

    Returns
    ----------------
    float
        Returns the IWD value, where is a number between 0 and some 
        positive value.

    """
    cdef:
        Py_ssize_t i, n = len(sequence)
        INT64_t[:] binary_mask = np.zeros(n, dtype=np.int64)
        double result

    for i in range(n):
        if sequence[i] in target_residues:
            binary_mask[i] = 1

    result = __compute_IWD_from_binary_mask(binary_mask)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double __compute_IWD_from_binary_mask(INT64_t[:] binary_mask, DOUBLE_t[:] weights=None):
    """
    Internal Cython function to compute the inverse weighted distance (IWD) from a binary mask.
    This function is not intended to be called directly, but should be used by functions
    within this module. It is the single point of calculation for the monovariate IWD.

    Parameters
    ----------
    binary_mask : memoryview of np.ndarray[int64_t]
        A memory view of a numpy array where each element is either 0 or 1. 
        This mask is used to calculate the inverse weighted distance.

    weights : memoryview of np.ndarray[double], optional
        A memory view of a numpy array where each element is a real number 
        used to weight the IWD calculation at a per-residue level. If not provided, 
        equal weights are assumed.

    Returns
    -------
    double
        The computed IWD value, which is a positive number.

    """
    cdef:
        Py_ssize_t i, j, n = binary_mask.shape[0]
        Py_ssize_t num_hits = 0
        double total_distance, inverse_distance_sum = 0.0
        INT64_t hit_index
        DOUBLE_t weight, weight_sum = 0.0

    # if no weights are passed, use a null weight mask
    if weights is None:
        weights = np.ones(n, dtype=np.float64)

    # Count hits and allocate space for hit indices
    for i in range(n):
        if binary_mask[i] == 1:
            num_hits += 1

    if num_hits == 0:
        return 0.0

    # Calculate inverse weighted distances
    for i in range(n):
        if binary_mask[i] == 1:
            hit_index = i
            total_distance = 0.0
            weight = weights[hit_index]
            weight_sum += weight

            for j in range(n):
                if binary_mask[j] == 1 and i != j:
                    total_distance += 1.0 / fabs(hit_index - j)

            inverse_distance_sum += total_distance * weight

    return inverse_distance_sum / weight_sum

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_average_inverse_distance_charge(DOUBLE_t[:] linear_NCPR, str sequence, str charge):
    """
    Function that returns the charge weighted average inverse distance of either the positive or
    negative residues in the sequence. For more information on the average_inverse_distance see 
    the calculate_average_inverse_distance function. The only difference is here the per residue sum 
    of the inverse distances is weighted by the absolute value of the charge at that posision. IE 
    residues with higher charge hold more more weight when evaluating how clustered that specific 
    charge residue is. This metric can be used to quantify the local clustering of either positive 
    or negitive charge residues in an amino acid sequence.

    Parameters
    -------------
    linear_NCPR : memoryview of np.ndarray[double]
        Linear net charge per residue (NCPR) as np.ndarray calculated accross the sequence. For 
        more information on how to calculate the NCPR see SPARROW track tools.

    sequence : str
        Amino acid sequence spit into list, where element 0 is the first residue in sequence and 
        element N-1 is the Nth residue in the sequence
    
    charge : string  options=['-','+']
        Pass '-' to quantify the clustering of negitive residues.
        Pass '+' to quantify the clustering of positive residues.

    Returns
    --------
    float 
        Returns average charge weighted IWD value for the sequence based on the passed,
        NCPR mask, sequence, and which prefered charge to calculate the charge-weighted average IWD    
    """    
    cdef:
        Py_ssize_t i, n = len(sequence)
        list charged_indices = []
        int num_charged = 0
        double result
        bint is_negatively_charged = 0, is_positively_charged = 0

    if charge == '-':
        is_negatively_charged = 1
        is_positively_charged = 0
    else:
        is_negatively_charged = 0
        is_positively_charged = 1

    for i in range(n):
        if (is_negatively_charged and sequence[i] in ['D', 'E']) or \
           (is_positively_charged and sequence[i] in ['R', 'K']):
            charged_indices.append(i)
            num_charged += 1

    if num_charged == 0:
        return 0.0

    result = __compute_charge_IWD(linear_NCPR, charged_indices, num_charged, is_negatively_charged)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double __compute_charge_IWD(DOUBLE_t[:] linear_NCPR, list charged_indices, int num_charged, bint is_negatively_charged):
    """
    Internal Cython function to compute the inverse weighted distance (IWD) for charged residues.
    This function calculates the IWD based on the positions and charges of specified residues.

    Parameters
    ----------
    linear_NCPR : memoryview of np.ndarray[double]
        A memory view of a numpy array representing the net charge per residue (NCPR) values for the sequence.

    charged_indices : list of int
        A list of indices corresponding to charged residues within the sequence.

    num_charged : int
        The number of charged residues (length of `charged_indices`).

    is_negatively_charged : bint
        A boolean indicating whether the calculation is for negatively charged residues. If True, 
        the function considers negative charges; otherwise, it considers positive charges.

    Returns
    -------
    double
        The computed inverse weighted distance (IWD) value for the charged residues, which is a positive number.

    Notes
    -----
    This function should not be called directly. It is intended for use within this module by other functions
    that handle higher-level calculations and user interactions.

    The function iterates over all pairs of charged residues, calculating the inverse of the distance between
    each pair and summing these values. The sum is weighted by the absolute NCPR value of each residue, 
    depending on the charge type (negative or positive) specified by `is_negatively_charged`.
    """
    cdef:
        Py_ssize_t i, j
        double inverse_distance_sum = 0.0
        double total_distance
        double charge_value = -1.0 if is_negatively_charged else 1.0
        
    for i in range(num_charged):
        total_distance = 0.0
        for j in range(num_charged):
            if i != j:
                total_distance += 1.0 / abs(charged_indices[j] - charged_indices[i])

        if np.sign(linear_NCPR[charged_indices[i]]) == charge_value:
            inverse_distance_sum += fabs(linear_NCPR[charged_indices[i]]) * total_distance
        else:
            inverse_distance_sum += 0.0

    return inverse_distance_sum / num_charged

Copy code
@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_average_bivariate_inverse_distance_charge(DOUBLE_t[:] linear_NCPR, str sequence):
    """
    Calculates the charge-weighted average inverse distance of either positive or negative residues 
    in the given sequence. This metric quantifies the local clustering of charged residues (positive 
    or negative) within an amino acid sequence. Higher charge residues have more weight in the 
    calculation, highlighting their influence on clustering.

    Parameters
    ----------
    linear_NCPR : memoryview of np.ndarray[double]
        A memory view of a numpy array representing the net charge per residue (NCPR) values for the sequence.
        For more information on how to calculate the NCPR, refer to the SPARROW documentation.

    sequence : str
        The amino acid sequence as a string where each character represents a residue.

    charge : str
        A string indicating which type of charge to consider for the calculation. 
        Options are:
        - '-' : Quantifies the clustering of negative residues.
        - '+' : Quantifies the clustering of positive residues.

    Returns
    -------
    float
        The average charge-weighted inverse weighted distance (IWD) value for the sequence based 
        on the specified NCPR values, sequence, and chosen charge type.

    Notes
    -----
    - This function uses the absolute value of the charge at each position to weight the sum of the inverse distances.
    - The calculated metric can be used to understand the local clustering of either positive or negative residues 
      in the sequence.
    """
    cdef:
        Py_ssize_t i, j, n = len(sequence)
        INT64_t[:] negative_indices = np.zeros(n, dtype=np.int64)
        INT64_t[:] positive_indices = np.zeros(n, dtype=np.int64)
        int num_negative = 0, num_positive = 0
        double result

    for i in range(n):
        if sequence[i] in ['D', 'E']:
            negative_indices[num_negative] = i
            num_negative += 1
        elif sequence[i] in ['R', 'K']:
            positive_indices[num_positive] = i
            num_positive += 1

    if num_negative == 0 or num_positive == 0:
        return 0.0

    result = __compute_bivariate_charge_IWD(
        linear_NCPR, negative_indices, num_negative,
        positive_indices, num_positive, sequence
    )
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double __compute_bivariate_charge_IWD(DOUBLE_t[:] linear_NCPR, INT64_t[:] negative_indices, int num_negative, INT64_t[:] positive_indices, int num_positive, str sequence):
    cdef:
        Py_ssize_t i, j
        double inverse_distance_sum = 0.0
        double total_distance
        DOUBLE_t[:] l_linear_NCPR = np.zeros_like(linear_NCPR)

    for i in range(linear_NCPR.shape[0]):
        if linear_NCPR[i] < 0 and sequence[i] in 'DE':
            l_linear_NCPR[i] = linear_NCPR[i]
        elif linear_NCPR[i] > 0 and sequence[i] in 'RK':
            l_linear_NCPR[i] = linear_NCPR[i]

    for i in range(num_negative):
        total_distance = 0.0
        for j in range(num_positive):
            total_distance += fabs(l_linear_NCPR[positive_indices[j]] - l_linear_NCPR[negative_indices[i]]) / abs(positive_indices[j] - negative_indices[i])
        inverse_distance_sum += total_distance

    return inverse_distance_sum / num_negative