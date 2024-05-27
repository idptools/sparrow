# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
import numpy as np
cimport cython
cimport numpy as np
from libc.math cimport abs, fabs
from cython.view cimport array as cvarray

# Define a typed memoryview for efficient access to numpy arrays
ctypedef np.float64_t DOUBLE_t
ctypedef np.int64_t INT64_t


@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_average_inverse_distance_from_sequence(str sequence, str target_residues):
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
    cdef:
        Py_ssize_t i, j, n = binary_mask.shape[0]
        Py_ssize_t num_hits = 0
        double total_distance, inverse_distance_sum = 0.0
        INT64_t[:] hit_indices
        DOUBLE_t[:] tmp_weights

    # if no weights are passed, use a null weight mask
    if weights is None:
        tmp_weights = np.ones(n, dtype=np.float64)
    else:
        tmp_weights = weights

    # Count hits and allocate space for hit indices
    for i in range(n):
        if binary_mask[i] == 1:
            num_hits += 1

    if num_hits == 0:
        return 0.0

    hit_indices = np.empty(num_hits, dtype=np.int64)

    # Populate hit indices
    num_hits = 0
    for i in range(n):
        if binary_mask[i] == 1:
            hit_indices[num_hits] = i
            num_hits += 1

    # Calculate inverse weighted distances
    for i in range(num_hits):
        total_distance = 0.0
        for j in range(num_hits):
            if i != j:
                total_distance += 1.0 / fabs(hit_indices[j] - hit_indices[i])
        inverse_distance_sum += total_distance * tmp_weights[hit_indices[i]]

    return inverse_distance_sum / num_hits


@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_average_inverse_distance_charge(DOUBLE_t[:] linear_NCPR, str sequence, str charge):
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

@cython.boundscheck(False)
@cython.wraparound(False)
def calculate_average_bivariate_inverse_distance_charge(DOUBLE_t[:] linear_NCPR, str sequence):
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