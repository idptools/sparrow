"""
backend for implementing the patterning.pyx moduel except using numpy 
vectorized operations instead of cython
"""

import numpy as np
from .. import sparrow_exceptions

def patterning_percentile(seq, group, window_size, count, seed, return_distribution=False):
    """
    User-facing general-purpose patterning function. Computes the P-value of observing the sequence
    passed by random chance when patterning of residues in group (vs. those residues not in group) are
    assessed.

    Returns a value between 0 and 1, where 0 means the real sequence is much more evenly patterned than
    expected by random chance, while 1 means the real sequence is much more blocky than expected by
    random chance.

    Parameters
    -----------
    seq : str
        Amino acid sequence (or any other string)

    group : list
        List of characters used to define patterning alphabet. Residues found in this group are designated as
        "1" while all others designated as 0. As such, this is the alphabet used for binary patterning to be 
        computed over.

    window_size : int
        Window size to be used when computing local patterning. Cannot be larger than the sequence length or
        an exception will be raised

    count : int
        The number of random shuffles of real sequence used (recommend between 100 and 1000; more are required
        for longer sequences

    seed: int 
        Seed used to initialize random number generator

    return_distribution : bool
        Boolean which if set to true means a tuple is returned with patterning value, distribution, and 
        percentile. If false only the Percentile is returned.

    Returns
    ------------
    float 
        Returns a single float that lies between 0 and 1. The value reports on the likleihood of observing a 
        sequence more evenly mixed than The real sequence by chance. If this approaches 0 it says likelyhood of 
        observing a more well-mixed sequence becomes vanishingly small. As this approaches 1 it says likelihood 
        of observing a more well-mixed sequence is more and more likely.
    """
    
    string_length = len(seq)
    
    # Sanity checks
    if window_size > len(seq):
        raise sparrow_exceptions.PatterningException('window_size is larger than sequence length')
    
    if window_size < 2:
        raise sparrow_exceptions.PatterningException('window_size must be 2 or larger')
    
    try:
        seed = int(seed)
    except:
        raise sparrow_exceptions.PatterningException('Cannot convert seed into an integer')
    
    if len(group) == 0:
        raise sparrow_exceptions.PatterningException('Passed group is empty, group must contain one or more possible residues')
    
    # Set random seed - use numpy random for consistency
    np.random.seed(seed)
    
    # Convert sequence to binary array - match Cython exactly
    seq_binary = binarize_exact(seq, group)
    
    # Calculate global compositional asymmetry - match Cython exactly
    wt_sigma = global_compositional_asymmetry_exact(seq_binary)
    
    # Calculate real patterning asymmetry - match Cython exactly
    real_delta = patterning_asymmetry_exact(seq_binary, window_size, wt_sigma)
    
    # Generate random shuffles - use exact Cython approach but with vectorized calculation
    random_shuffle_pvals = batch_shuffle_and_calculate_exact(seq_binary, window_size, wt_sigma, count)
    
    # Sort random values using quicksort to match Cython
    np.asarray(random_shuffle_pvals).sort(kind='quicksort')
    
    # Find percentile - match Cython logic exactly
    return_val = 1.0
    for i in range(count):
        if real_delta < random_shuffle_pvals[i]:
            return_val = i / count
            break
    
    if return_distribution:
        return (return_val, real_delta, random_shuffle_pvals)
    else:
        return return_val

def binarize_exact(seq, group):
    """
    Fully vectorized binarization function.
    """
    # Convert sequence to numpy array and group to set for fast lookup
    seq_array = np.array(list(seq))
    group_set = set(group)
    
    # Vectorized membership test
    binary_array = np.array([char in group_set for char in seq_array], dtype=np.int32)
    
    return binary_array

def global_compositional_asymmetry_exact(binary_array):
    """
    Fully vectorized global compositional asymmetry calculation.
    """
    string_length = len(binary_array)
    count = np.sum(binary_array)  # Vectorized sum instead of loop
    
    FA = count / string_length
    FB = 1 - FA
    
    return (FA - FB) * (FA - FB)

def patterning_asymmetry_exact(binary_seq, window_size, overall_asymmetry):
    """
    Fully vectorized patterning asymmetry calculation using advanced NumPy operations.
    """
    string_length = len(binary_seq)
    end = (string_length - window_size) + 1
    
    # Use sliding window view for maximum vectorization
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(binary_seq, window_size)
    except ImportError:
        # Fallback for older NumPy versions
        windows = np.array([binary_seq[i:i+window_size] for i in range(end)])
    
    # Vectorized calculations for all windows at once
    totals = np.sum(windows, axis=1)  # Sum each window
    FA_values = totals / window_size  # Calculate FA for all windows
    
    # Vectorized window asymmetries
    window_asymmetries = (FA_values - (1 - FA_values)) ** 2
    
    # Vectorized deviations from overall asymmetry
    deviations = (window_asymmetries - overall_asymmetry) ** 2
    
    # Return mean of all deviations
    return np.mean(deviations)

def batch_shuffle_and_calculate_exact(seq_binary, window_size, wt_sigma, count):
    """
    Smart dispatcher that chooses the most efficient approach based on count size.
    """
    # For smaller counts, use mega-vectorized approach
    if count <= 500:
        try:
            return batch_shuffle_and_calculate_mega_vectorized(seq_binary, window_size, wt_sigma, count)
        except MemoryError:
            pass  # Fall back to chunked approach
    
    # For larger counts or memory constraints, use chunked approach
    return batch_shuffle_and_calculate_exact_chunked(seq_binary, window_size, wt_sigma, count)

def batch_patterning_asymmetry_vectorized(shuffles_matrix, window_size, overall_asymmetry):
    """
    Fully vectorized calculation of patterning asymmetry for multiple sequences.
    This eliminates ALL loops using advanced NumPy broadcasting.
    """
    num_sequences, seq_length = shuffles_matrix.shape
    end = seq_length - window_size + 1
    
    # Create all possible windows for all sequences using advanced indexing
    # This creates a 3D array: (num_sequences, num_windows, window_size)
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        # Apply sliding window to each row of the matrix
        all_windows = np.array([sliding_window_view(seq, window_size) for seq in shuffles_matrix])
    except ImportError:
        # Fallback using broadcasting for older NumPy versions
        indices = np.arange(window_size)[None, None, :] + np.arange(end)[None, :, None]
        sequence_indices = np.arange(num_sequences)[:, None, None]
        all_windows = shuffles_matrix[sequence_indices, indices]
    
    # Vectorized sum across window dimension: shape (num_sequences, num_windows)
    window_sums = np.sum(all_windows, axis=2)
    
    # Vectorized FA calculation for all windows of all sequences
    FA_values = window_sums / window_size
    
    # Vectorized window asymmetries for all windows of all sequences
    window_asymmetries = (FA_values - (1 - FA_values)) ** 2
    
    # Vectorized deviations from overall asymmetry
    deviations = (window_asymmetries - overall_asymmetry) ** 2
    
    # Return mean deviation for each sequence
    return np.mean(deviations, axis=1)

def shuffle_array_vectorized_batch(sequences_matrix, chunk_size=1000):
    """
    Vectorized shuffling for large batches with memory management.
    """
    num_sequences = sequences_matrix.shape[0]
    
    # Process in chunks to manage memory for large count values
    for start_idx in range(0, num_sequences, chunk_size):
        end_idx = min(start_idx + chunk_size, num_sequences)
        chunk = sequences_matrix[start_idx:end_idx]
        
        # Vectorized shuffling for the chunk
        for i in range(chunk.shape[0]):
            np.random.shuffle(chunk[i])
    
    return sequences_matrix

# Alternative fully vectorized approach for smaller datasets
def batch_shuffle_and_calculate_mega_vectorized(seq_binary, window_size, wt_sigma, count):
    """
    Ultra-vectorized approach that processes everything at once.
    Use this for smaller count values to avoid memory issues.
    """
    seq_length = len(seq_binary)
    
    # Create all shuffles at once
    all_shuffles = np.tile(seq_binary, (count, 1))
    
    # Vectorized shuffling using advanced indexing
    # Generate random permutation indices for all sequences at once
    random_indices = np.array([np.random.permutation(seq_length) for _ in range(count)])
    
    # Apply permutations vectorized
    sequence_indices = np.arange(count)[:, None]
    all_shuffles = all_shuffles[sequence_indices, random_indices]
    
    # Vectorized asymmetry calculation
    return batch_patterning_asymmetry_vectorized(all_shuffles, window_size, wt_sigma)

def batch_shuffle_and_calculate_exact_chunked(seq_binary, window_size, wt_sigma, count, chunk_size=100):
    """
    Memory-efficient vectorized approach using chunking.
    """
    results = []
    
    for start_idx in range(0, count, chunk_size):
        end_idx = min(start_idx + chunk_size, count)
        current_chunk_size = end_idx - start_idx
        
        # Process chunk with full vectorization
        chunk_shuffles = np.tile(seq_binary, (current_chunk_size, 1))
        
        # Vectorized shuffling for chunk
        for i in range(current_chunk_size):
            np.random.shuffle(chunk_shuffles[i])
        
        # Vectorized asymmetry calculation for chunk
        chunk_results = batch_patterning_asymmetry_vectorized(chunk_shuffles, window_size, wt_sigma)
        results.extend(chunk_results)
    
    return np.array(results, dtype=np.float32)

# Remove unused functions

