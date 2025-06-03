import numpy as np
import numba as nb

@nb.njit(cache=True, fastmath=True)
def _compute_scd_optimized(charges_all, seq_lengths, valid_seqs, group_membership):
    """
    Highly optimized SCD computation that matches the exact Cython logic.
    """
    num_seqs = charges_all.shape[0]
    scd_totals = np.zeros(num_seqs, dtype=np.float64)
    
    for i in range(num_seqs):
        if not valid_seqs[i]:
            continue
            
        seq_len = seq_lengths[i]
        if seq_len < 2:
            continue
            
        total = 0.0
        
        # Match exact Cython logic: m in range(1, seqlen), n in range(0, m-1)
        for m in range(1, seq_len):
            charge_m = charges_all[i, m]
            
            # Early skip if charge_m is zero
            if charge_m == 0:
                continue
                
            m_val = m + 1  # Match Cython: m_val = m + 1
            
            # n in range(0, m-1) means n goes from 0 to m-2 (inclusive)
            for n in range(0, m):  # This is range(0, m-1+1) = range(0, m)
                if n >= m - 1:  # But we want n < m-1, so n <= m-2
                    break
                    
                charge_n = charges_all[i, n]
                
                # Early skip if charge_n is zero
                if charge_n == 0:
                    continue
                
                n_val = n + 1  # Match Cython: n_val = n + 1
                
                # Match Cython calculation exactly
                charge_val = charge_m * charge_n
                final_val = charge_val * np.sqrt(m_val - n_val)
                total += final_val
        
        scd_totals[i] = total / seq_len
        
    return scd_totals

@nb.njit(cache=True, fastmath=True)
def _build_group_membership(group1_indices, group2_indices):
    """Build group membership array with Numba for better performance."""
    group_membership = np.zeros(20, dtype=np.int8)
    
    for idx in group1_indices:
        group_membership[idx] = -1
    
    for idx in group2_indices:
        group_membership[idx] = 1
        
    return group_membership

@nb.njit(cache=True, fastmath=True)
def _convert_to_charges(seq_matrix_int, valid_mask, group_membership):
    """Convert sequences to charges with Numba optimization."""
    num_seqs, max_len = seq_matrix_int.shape
    charges_all = np.zeros((num_seqs, max_len), dtype=np.int8)
    
    for i in range(num_seqs):
        for j in range(max_len):
            if valid_mask[i, j]:
                aa_idx = seq_matrix_int[i, j]
                if 0 <= aa_idx < 20:
                    charges_all[i, j] = group_membership[aa_idx]
    
    return charges_all

def calculate_batch_SCD(seq_matrix, valid_mask, seq_lengths, group1=['E', 'D'], group2=['R', 'K']):
    """
    Highly optimized SCD calculation using advanced Numba techniques.
    
    Parameters
    ----------
    seq_matrix : np.ndarray
        2D numpy array of sequences with amino acid indices
    valid_mask : np.ndarray
        Boolean mask indicating valid positions (not NaN)
    seq_lengths : np.ndarray
        Array of sequence lengths
    group1 : list
        List of residues in group 1 (acidic)
    group2 : list
        List of residues in group 2 (basic)
    
    Returns
    -------
    np.ndarray
        Array of SCD values for each sequence
    """
    # Convert groups to indices
    AA_TO_INT = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 
                 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 
                 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
    
    group1_indices = np.array([AA_TO_INT[res] for res in group1], dtype=np.int32)
    group2_indices = np.array([AA_TO_INT[res] for res in group2], dtype=np.int32)

    num_seqs = seq_matrix.shape[0]
    result = np.full(num_seqs, np.nan)
    
    # Early return for sequences too short
    valid_seqs = seq_lengths >= 2
    if not valid_seqs.any():
        return result
    
    # Build group membership with Numba
    group_membership = _build_group_membership(group1_indices, group2_indices)
    
    # Convert to integer matrix efficiently
    seq_matrix_int = np.zeros_like(seq_matrix, dtype=np.int32)
    seq_matrix_int[valid_mask] = seq_matrix[valid_mask].astype(np.int32)
    
    # Convert to charges with Numba for better cache performance
    charges_all = _convert_to_charges(seq_matrix_int, valid_mask, group_membership)
    
    # Compute SCD values using optimized Numba function
    scd_values = _compute_scd_optimized(charges_all, seq_lengths, valid_seqs, group_membership)
    
    # Set results
    result[valid_seqs] = scd_values[valid_seqs]
    
    return result

