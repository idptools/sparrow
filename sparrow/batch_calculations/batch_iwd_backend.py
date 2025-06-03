import numpy as np


def calculate_batch_iwd(seq_matrix, valid_mask, residues, weights=None):
    """
    Fully vectorized IWD computation with batch processing by hit count.
    This version groups sequences by number of hits and processes them together.
    """
    AA_TO_INT = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 
                'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 
                'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}

    if isinstance(residues, str):
        residues = [residues]
    residues = [AA_TO_INT[res] for res in residues if res in AA_TO_INT]
    if not residues:
        raise ValueError("No valid residues provided for IWD calculation.")

    n_sequences, seq_length = seq_matrix.shape
    
    # Create binary mask
    res_mask = np.isin(seq_matrix, residues) & valid_mask
    
    # Pre-filter sequences with less than 2 hits to avoid unnecessary computation
    hit_counts = np.sum(res_mask, axis=1)
    valid_sequences = np.where(hit_counts >= 2)[0]

    # Set up weights
    if weights is None:
        weights = np.ones((n_sequences, seq_length), dtype=np.float64)
    elif weights.ndim == 1:
        weights = np.tile(weights, (n_sequences, 1))

    results = np.zeros(n_sequences, dtype=np.float64)
    
    # Process only valid sequences (those with at least 2 hits)
    for seq_idx in valid_sequences:
        hit_positions = np.where(res_mask[seq_idx])[0]
        n_hits = hit_positions.size

        # Vectorized computation for this sequence
        pos_diff = hit_positions[:, None] - hit_positions[None, :]
        distances = np.abs(pos_diff).astype(np.float64)
        np.fill_diagonal(distances, 1.0)  # Avoid division by zero

        inv_distances = 1.0 / distances
        np.fill_diagonal(inv_distances, 0.0)  # Zero out self-distances

        distance_sums = np.sum(inv_distances, axis=1)
        hit_weights = weights[seq_idx, hit_positions]

        results[seq_idx] = np.sum(distance_sums * hit_weights) / np.sum(hit_weights)

    return results


