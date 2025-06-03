'''
For doing batch calculations of kappa values.
Individually the cythonized version wins, but for batch
the numpy version is faster.
Because this code ended up being... long, it gets its own
special file as it deserves.
'''

'''
Numpy vectorized implementation of kappa calculations for multiple sequences
'''

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def vectorized_ternarize(seq_matrix, group1, group2, valid_mask):
    """
    Only accepts numeric (from seqs_to_matrix) input.
    
    Parameters
    ----------
    sequences : list or np.ndarray
        List of amino acid sequences (all sequences must be same length)
        
    group1 : list
        List of residues considered as group 1 (assigned value 1)
        
    group2 : list
        List of residues considered as group 2 (assigned value -1)
        
    Returns
    -------
    np.ndarray
        Array of shape (n_sequences, sequence_length) with ternary values
    """
    if not isinstance(seq_matrix, np.ndarray):
        raise ValueError("Input must be a numpy array from seqs_to_matrix")

    n_sequences, seq_length = seq_matrix.shape
    ternary_array = np.zeros((n_sequences, seq_length), dtype=np.float32 if not valid_mask.all() else np.int8)

    # Optimized approach using np.isin for better performance with multiple residues
    if group1:
        group1_mask = np.isin(seq_matrix, group1) & valid_mask
        ternary_array[group1_mask] = 1
    
    if group2:
        group2_mask = np.isin(seq_matrix, group2) & valid_mask
        ternary_array[group2_mask] = -1

    # For NaN-padded sequences, keep NaN in invalid positions
    if not valid_mask.all():
        ternary_array[~valid_mask] = np.nan

    return ternary_array


def vectorized_global_asymmetry(ternary_arrays_winged):
    """
    Compute global compositional asymmetry for multiple sequences.
    This matches the Cython implementation which calculates asymmetry on the winged sequence.
    Handles NaN-padded sequences by only counting valid positions.
    
    Parameters
    ----------
    ternary_arrays_winged : np.ndarray
        Array of shape (n_sequences, winged_length) with ternary values (includes wings)
        
    Returns
    -------
    np.ndarray
        Array of global asymmetry values for each sequence
    """
    n_sequences, winged_length = ternary_arrays_winged.shape

    # For NaN-padded sequences, we need to be more careful
    if ternary_arrays_winged.dtype.kind == 'f':
        # Handle NaN values - only count valid positions
        valid_mask = ~np.isnan(ternary_arrays_winged)
        
        # Use boolean indexing for faster counting
        count_A = np.sum((ternary_arrays_winged == 1) & valid_mask, axis=1)
        count_B = np.sum((ternary_arrays_winged == -1) & valid_mask, axis=1)
        valid_lengths = np.sum(valid_mask, axis=1)
        
        # Calculate fractions using valid length - vectorized division with zero handling
        FA = np.divide(count_A, valid_lengths, out=np.zeros_like(count_A, dtype=np.float64), where=valid_lengths>0)
        FB = np.divide(count_B, valid_lengths, out=np.zeros_like(count_B, dtype=np.float64), where=valid_lengths>0)
    else:
        # Integer arrays - no NaN, use full winged length like Cython
        count_A = np.sum(ternary_arrays_winged == 1, axis=1)
        count_B = np.sum(ternary_arrays_winged == -1, axis=1)
        
        # Calculate fractions using the full winged length (like Cython does)
        FA = count_A / winged_length
        FB = count_B / winged_length

    # Vectorized asymmetry calculation
    denominator = FA + FB
    asymmetry = np.divide((FA - FB)**2, denominator, out=np.zeros_like(denominator), where=denominator>0)

    return asymmetry



def vectorized_patterning_asymmetry_optimized(ternary_arrays_winged, window_size, overall_asymmetry):
    """
    Optimized version using sliding window view for better performance.
    Compute patterning asymmetry for multiple sequences using sliding windows.
    Handles NaN-padded positions by ignoring them.
    
    Parameters
    ----------
    ternary_arrays_winged : np.ndarray
        Array of shape (n_sequences, winged_length) with ternary values
        
    window_size : int
        Size of the sliding window
        
    overall_asymmetry : np.ndarray
        Array of global asymmetry values for each sequence
        
    Returns
    -------
    np.ndarray
        Array of patterning asymmetry values for each sequence
    """
    n_sequences, winged_length = ternary_arrays_winged.shape
    
    if winged_length < window_size:
        return np.zeros(n_sequences)
    
    # Pre-allocate arrays for better memory efficiency
    delta_sum = np.zeros(n_sequences, dtype=np.float64)
    valid_window_counts = np.zeros(n_sequences, dtype=np.int32)
    

    # Create sliding windows for all sequences at once
    windows = sliding_window_view(ternary_arrays_winged, window_size, axis=1)
    # Shape: (n_sequences, n_windows, window_size)
    n_windows = windows.shape[1]
    
    # Handle NaN values efficiently
    if ternary_arrays_winged.dtype.kind == 'f':
        # For each window, check if all values are valid (not NaN)
        valid_mask = ~np.isnan(windows)  # Shape: (n_sequences, n_windows, window_size)
        valid_windows = np.all(valid_mask, axis=2)  # Shape: (n_sequences, n_windows)
    else:
        valid_windows = np.ones((n_sequences, n_windows), dtype=bool)
        valid_mask = np.ones_like(windows, dtype=bool)
    
    # Count residues for all windows at once
    count_A = np.sum((windows == 1) & valid_mask, axis=2)  # Shape: (n_sequences, n_windows)
    count_B = np.sum((windows == -1) & valid_mask, axis=2)  # Shape: (n_sequences, n_windows)
    
    # Calculate window lengths (all should be window_size for valid windows)
    if ternary_arrays_winged.dtype.kind == 'f':
        window_lengths = np.sum(valid_mask, axis=2)
    else:
        window_lengths = np.full((n_sequences, n_windows), window_size)
    
    # Calculate fractions only for valid windows
    FA_local = np.zeros_like(count_A, dtype=np.float64)
    FB_local = np.zeros_like(count_B, dtype=np.float64)
    
    valid_calc = valid_windows & (window_lengths > 0)
    FA_local[valid_calc] = count_A[valid_calc] / window_lengths[valid_calc]
    FB_local[valid_calc] = count_B[valid_calc] / window_lengths[valid_calc]
    
    # Calculate window asymmetry for all windows
    denominator = FA_local + FB_local
    window_asymmetry = np.zeros_like(FA_local, dtype=np.float64)
    valid_asym = valid_calc & (denominator > 0)
    window_asymmetry[valid_asym] = ((FA_local[valid_asym] - FB_local[valid_asym])**2) / denominator[valid_asym]
    
    # Calculate delta contributions
    overall_asymmetry_expanded = overall_asymmetry[:, np.newaxis]  # Shape: (n_sequences, 1)
    delta_contributions = (window_asymmetry - overall_asymmetry_expanded)**2
    
    # Sum valid contributions for each sequence
    delta_sum = np.sum(delta_contributions * valid_windows, axis=1)
    valid_window_counts = np.sum(valid_windows, axis=1)
        

    # Calculate average delta
    result = np.zeros(n_sequences, dtype=np.float64)
    has_valid_windows = valid_window_counts > 0
    result[has_valid_windows] = delta_sum[has_valid_windows] / valid_window_counts[has_valid_windows]
    
    return result

def vectorized_deltamax_patterning_asymmetry(ternary_arrays, window_size, overall_asymmetry, 
                                             valid_mask, seq_lengths):
    """
    Compute the deltamax patterning asymmetry for multiple sequences.
    Handles NaN-padded positions by ignoring them and creating properly winged maximally segregated sequences.
    
    Parameters
    ----------
    ternary_arrays : np.ndarray
        Array of shape (n_sequences, sequence_length) with ternary values
        
    window_size : int
        Size of the sliding window
        
    overall_asymmetry : np.ndarray
        Array of global asymmetry values for each sequence

    valid_mask : np.ndarray
        Precomputed valid mask for sequences, if available
    
    seq_lengths : np.ndarray
        Precomputed sequence lengths, if available
        
    Returns
    -------
    np.ndarray
        Array of deltamax values for each sequence
    """
    n_sequences, seq_length = ternary_arrays.shape

    # Mask for valid positions
    if ternary_arrays.dtype.kind == 'f':
        # Get actual sequence lengths for each sequence
        actual_lengths = np.sum(valid_mask, axis=1)
    else:
        valid_mask = np.ones_like(ternary_arrays, dtype=bool)
        actual_lengths = np.full(n_sequences, seq_length)

    count_A = np.sum((ternary_arrays == 1) & valid_mask, axis=1)
    count_B = np.sum((ternary_arrays == -1) & valid_mask, axis=1)
    count_C = np.sum((ternary_arrays == 0) & valid_mask, axis=1)

    # Create winged arrays for each sequence based on their actual length
    winged_length = seq_length + 2 * window_size
    
    # For float arrays, initialize with NaN, for int arrays with zeros
    if ternary_arrays.dtype.kind == 'f':
        max_segregated_winged = np.full((n_sequences, winged_length), np.nan, dtype=ternary_arrays.dtype)
    else:
        max_segregated_winged = np.zeros((n_sequences, winged_length), dtype=ternary_arrays.dtype)

    for i in range(n_sequences):
        actual_len = int(actual_lengths[i])
        
        if actual_len == 0:
            # Empty sequence, skip
            continue
            
        # Build maximally segregated sequence within the actual length
        max_segregated_seq = np.zeros(actual_len, dtype=ternary_arrays.dtype)
        pos = 0
        
        # Place group1 residues (value 1)
        if count_A[i] > 0:
            end_pos = min(pos + int(count_A[i]), actual_len)
            max_segregated_seq[pos:end_pos] = 1
            pos = end_pos
            
        # Place neutral residues (value 0)
        if count_C[i] > 0 and pos < actual_len:
            end_pos = min(pos + int(count_C[i]), actual_len)
            max_segregated_seq[pos:end_pos] = 0
            pos = end_pos
            
        # Place group2 residues (value -1)
        if count_B[i] > 0 and pos < actual_len:
            end_pos = min(pos + int(count_B[i]), actual_len)
            max_segregated_seq[pos:end_pos] = -1
            pos = end_pos

        # Create the winged sequence: left_wing + actual_sequence + right_wing
        # Calculate the actual winged sequence length for this sequence
        actual_winged_length = window_size + actual_len + window_size
        
        if ternary_arrays.dtype.kind == 'f':
            # For float arrays: zeros for wings, actual sequence in middle, NaN for remainder
            # Left wing (zeros)
            max_segregated_winged[i, :window_size] = 0
            # Actual sequence
            max_segregated_winged[i, window_size:window_size + actual_len] = max_segregated_seq
            # Right wing (zeros)
            max_segregated_winged[i, window_size + actual_len:window_size + actual_len + window_size] = 0
            # Everything beyond the right wing remains NaN (already initialized)
        else:
            # For integer arrays: zeros for wings, actual sequence in middle, zeros for remainder
            # Left wing is already zero
            # Place actual sequence
            max_segregated_winged[i, window_size:window_size + actual_len] = max_segregated_seq
            # Right wing and remainder are already zero

    # Calculate patterning asymmetry on the maximally segregated winged sequences
    deltamax = vectorized_patterning_asymmetry_optimized(max_segregated_winged, window_size, overall_asymmetry)
    return deltamax


def vectorized_kappa_x_optimized(sequences, group1, group2, window_size, valid_mask, 
                                 seq_lengths, flatten=False, ternarize=True):
    """
    Optimized version of kappa calculation with better memory management and vectorization.
    Calculate kappa values for multiple sequences simultaneously using vectorized operations.
    Accepts both string and numeric (from seqs_to_matrix) input.
    
    Parameters
    ----------
    sequences : list or np.ndarray
        List of amino acid sequences (all must be same length)
        
    group1 : list
        List of residues considered as group 1
        
    group2 : list
        List of residues considered as group 2
        
    window_size : int
        Size of the sliding window
        
    flatten : bool
        If True, kappa values above 1 will be set to 1

    ternarize : bool, optional
        If True, sequences will be ternarized before kappa calculation    

    valid_mask : np.ndarray
        Precomputed valid mask

    seq_lengths : np.ndarray  
        Precomputed sequence lengths

    Returns
    -------
    np.ndarray
        Array of kappa values for each sequence
    """
    
    # Validate window size
    if window_size < 2:
        raise ValueError('window_size must be 2 or larger')
    
    if not group1:
        raise ValueError('group1 must contain one or more possible residues')
    
    # Determine input type and shape
    if isinstance(sequences, np.ndarray):
        n_sequences = sequences.shape[0]
        seq_length = sequences.shape[1]
    else:
        n_sequences = len(sequences)
        seq_length = len(sequences[0])

    # Check if all sequences have the same length (for string input)
    if not isinstance(sequences, np.ndarray):
        if not all(len(seq) == seq_length for seq in sequences):
            raise ValueError('All sequences must have the same length')
    
    # Early exit for invalid window sizes
    if isinstance(sequences, np.ndarray) and np.issubdtype(sequences.dtype, np.number):
        valid_lengths = np.sum(~np.isnan(sequences), axis=1)
        min_valid_length = np.min(valid_lengths)
        if min_valid_length == 0:
            return np.full(n_sequences, -1.0)
    else:
        if window_size > seq_length:
            return np.full(n_sequences, -1.0)
    
    # Ternarize sequences
    if ternarize:
        ternary_arrays = vectorized_ternarize(sequences, group1, group2, valid_mask)
    else:
        ternary_arrays = sequences

    # Create winged arrays more efficiently
    wing_size = window_size
    winged_length = seq_length + 2 * wing_size

    # Use appropriate dtype for memory efficiency
    dtype = np.float32 if ternary_arrays.dtype.kind == 'f' else np.int8
    ternary_winged = np.zeros((n_sequences, winged_length), dtype=dtype)
    
    if ternary_arrays.dtype.kind == 'f':
        # For NaN-padded sequences, handle carefully
        ternary_winged[:, wing_size:wing_size+seq_length][valid_mask] = ternary_arrays[valid_mask]
        
        # Vectorized NaN assignment for positions beyond actual sequence
        for i in range(n_sequences):
            actual_winged_end = seq_lengths[i] + wing_size * 2
            if actual_winged_end < winged_length:
                ternary_winged[i, actual_winged_end:] = np.nan
    else:
        # For integer sequences, direct assignment
        ternary_winged[:, wing_size:wing_size+seq_length] = ternary_arrays

    # Vectorized validation checks
    has_group1 = np.sum(ternary_arrays == 1, axis=1) > 0
    has_group2 = np.sum(ternary_arrays == -1, axis=1) > 0
    
    if isinstance(sequences, np.ndarray) and np.issubdtype(sequences.dtype, np.number):
        sufficient_length = valid_lengths >= window_size
    else:
        sufficient_length = np.ones(n_sequences, dtype=bool)
    
    valid_sequences = has_group1 & has_group2 & sufficient_length

    # Calculate asymmetries using optimized functions
    overall_asymmetry = vectorized_global_asymmetry(ternary_winged)
    delta = vectorized_patterning_asymmetry_optimized(ternary_winged, window_size, overall_asymmetry)
    deltamax = vectorized_deltamax_patterning_asymmetry(ternary_arrays, window_size, overall_asymmetry, valid_mask, seq_lengths)
    
    # Vectorized kappa calculation
    kappa = np.full(n_sequences, -1.0, dtype=np.float64)
    valid_calc = valid_sequences & (deltamax > 0)
    kappa[valid_calc] = delta[valid_calc] / deltamax[valid_calc]
    
    # Flatten if requested
    if flatten:
        kappa = np.minimum(kappa, 1.0)
    
    return kappa

def calculate_batch_kappa(sequences, group1=[2,3], group2=[8,14], flatten=True, ternarize=True,
                valid_mask=None, seq_lengths=None):
    """
    User-friendly function to calculate kappa values for multiple sequences.
    Accepts both string and numeric (from seqs_to_matrix) input.
    
    Parameters
    ----------
    sequences : list
        List of amino acid sequences (can be variable length)
        
    group1 : list, optional
        List of residues for group 1, default: positively charged residues
        
    group2 : list, optional
        List of residues for group 2, default: negatively charged residues
        
    window_size : int, optional
        Size of sliding window, default: 5
        
    flatten : bool, optional
        If True, kappa values above 1 will be set to 1
    
    ternarize : bool, optional
        If True, sequences will be ternarized before kappa calculation

    valid_mask : np.ndarray, optional
        Precomputed valid mask for sequences, if available
    seq_lengths : np.ndarray, optional
        Precomputed sequence lengths, if available
            
    
    Returns
    -------
    np.ndarray
        Array of kappa values for each sequence
    """
    # Check if sequences are encoded (from seqs_to_matrix) and convert groups accordingly
    if isinstance(sequences, np.ndarray) and np.issubdtype(sequences.dtype, np.number):
        if isinstance(group1[0], str) or isinstance(group2[0], str):

            # Amino acid to integer mapping from batch_tools.py seqs_to_matrix function
            AA_TO_INT_CONVERT = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 
                    'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 
                    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'B':11, 'U':1, 'X':5, 'Z':13}
            
            # Convert string-based groups to integer groups
            group1_converted = [AA_TO_INT_CONVERT[aa] for aa in group1 if aa in AA_TO_INT_CONVERT]
            group2_converted = [AA_TO_INT_CONVERT[aa] for aa in group2 if aa in AA_TO_INT_CONVERT]
            
            if not group1_converted:
                raise ValueError(f"None of the residues in group1 {group1} are valid amino acids")
            if not group2_converted:
                raise ValueError(f"None of the residues in group2 {group2} are valid amino acids")
                
            group1_final = group1_converted
            group2_final = group2_converted
        else:
            # Numeric groups are already valid
            group1_final = group1
            group2_final = group2

        # get sequence lengths if not provided
        if seq_lengths is None:
            seq_lengths = np.sum(~np.isnan(sequences), axis=1)
        # get valid mask if not provided
        if valid_mask is None:
            valid_mask = ~np.isnan(sequences)
        
        # calculate kappa values using optimized vectorized_kappa_x
        k5 = vectorized_kappa_x_optimized(
            sequences, 
            group1=group1_final, 
            group2=group2_final, 
            window_size=5, 
            flatten=flatten, 
            ternarize=ternarize,
            valid_mask=valid_mask,
            seq_lengths=seq_lengths 
        )
        k6 = vectorized_kappa_x_optimized(
            sequences, 
            group1=group1_final, 
            group2=group2_final, 
            window_size=6, 
            flatten=flatten, 
            ternarize=ternarize,
            valid_mask=valid_mask,
            seq_lengths=seq_lengths
        )
        # kappa is the average of k5 and k6, so get that for each sequence.
        kappa = np.nanmean(np.vstack((k5, k6)), axis=0)
        return kappa
        
    else:
        raise ValueError("Input sequences must be a numpy array from seqs_to_matrix or a list of strings")

