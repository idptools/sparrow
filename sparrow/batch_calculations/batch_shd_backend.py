import numpy as np
import numba as nb

@nb.njit(cache=True, fastmath=True)
def _compute_shd_optimized(hydro_all, seq_lengths, valid_seqs):
    """
    Highly optimized SHD computation that matches the exact Cython logic.
    Following Cython: for m in range(1, N), for n in range(m-1), t += (h[m] + h[n]) / abs(m - n)
    """
    num_seqs = hydro_all.shape[0]
    shd_totals = np.zeros(num_seqs, dtype=np.float64)
    
    for i in range(num_seqs):
        if not valid_seqs[i]:
            continue
            
        seq_len = seq_lengths[i]
        if seq_len < 2:
            continue
            
        total = 0.0
        
        # Match exact Cython logic: m in range(1, N), n in range(m-1)
        for m in range(1, seq_len):
            hydro_m = hydro_all[i, m]
            
            # n in range(m-1) means n goes from 0 to m-2 (inclusive)
            for n in range(m - 1):  # This is the exact Cython logic
                hydro_n = hydro_all[i, n]
                
                # Match Cython calculation: (h[m] + h[n]) / abs(m - n)
                distance = abs(m - n)
                if distance > 0:
                    contribution = (hydro_m + hydro_n) / distance
                    total += contribution
        
        # Match Cython: return t / N
        shd_totals[i] = total / seq_len
        
    return shd_totals

@nb.njit(cache=True, fastmath=True)
def _build_hydrophobicity_mapping():
    """Build hydrophobicity mapping array with Numba for better performance."""
    # Hydrophobicity values for the 20 standard amino acids matching Cython DEFAULT_HYDRO_DICT
    # Order: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
    hydro_values = np.array([0.730, 0.595, 0.378, 0.459, 1.000, 0.649, 0.514, 0.973, 
                            0.514, 0.973, 0.838, 0.432, 1.000, 0.514, 0.000, 0.595, 
                            0.676, 0.892, 0.946, 0.865], dtype=np.float64)
    return hydro_values

@nb.njit(cache=True, fastmath=True)
def _convert_to_hydrophobicity(seq_matrix_int, valid_mask, hydro_mapping):
    """Convert sequences to hydrophobicity values with Numba optimization."""
    num_seqs, max_len = seq_matrix_int.shape
    hydro_all = np.zeros((num_seqs, max_len), dtype=np.float64)
    
    for i in range(num_seqs):
        for j in range(max_len):
            if valid_mask[i, j]:
                aa_idx = seq_matrix_int[i, j]
                if 0 <= aa_idx < 20:
                    hydro_all[i, j] = hydro_mapping[aa_idx]
    
    return hydro_all

def calculate_batch_SHD(seq_matrix, valid_mask, seq_lengths):
    """
    Highly optimized SHD calculation using advanced Numba techniques.
    
    Parameters
    ----------
    seq_matrix : np.ndarray
        2D numpy array of sequences with amino acid indices
    valid_mask : np.ndarray
        Boolean mask indicating valid positions (not NaN)
    seq_lengths : np.ndarray
        Array of sequence lengths
    
    Returns
    -------
    np.ndarray
        Array of SHD values for each sequence
    """
    num_seqs = seq_matrix.shape[0]
    result = np.full(num_seqs, np.nan)
    
    # Early return for sequences too short
    valid_seqs = seq_lengths >= 2
    if not valid_seqs.any():
        return result
    
    # Build hydrophobicity mapping with Numba
    hydro_mapping = _build_hydrophobicity_mapping()
    
    # Convert to integer matrix efficiently
    seq_matrix_int = np.zeros_like(seq_matrix, dtype=np.int32)
    seq_matrix_int[valid_mask] = seq_matrix[valid_mask].astype(np.int32)
    
    # Convert to hydrophobicity values with Numba for better cache performance
    hydro_all = _convert_to_hydrophobicity(seq_matrix_int, valid_mask, hydro_mapping)
    
    # Compute SHD values using optimized Numba function
    shd_values = _compute_shd_optimized(hydro_all, seq_lengths, valid_seqs)
    
    # Set results
    result[valid_seqs] = shd_values[valid_seqs]
    
    return result
