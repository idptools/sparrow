'''
Functionality for batch tools. 
'''

import numpy as np

def group_sequences_by_length(seq_list, tolerance=1):
    """
    Group sequences by similar lengths to minimize padding overhead.
    
    Parameters
    ----------
    seq_list : list
        List of sequences
    tolerance : int
        Maximum length difference within a group
        
    Returns
    -------
    dict
        Dictionary where keys are representative lengths and values are lists of sequences
    """
    length_groups = {}
    
    for seq in seq_list:
        seq_len = len(seq)
        
        # Find existing group within tolerance
        assigned = False
        group_to_update = None
        
        for group_len in list(length_groups.keys()):
            if abs(seq_len - group_len) <= tolerance:
                length_groups[group_len].append(seq)
                assigned = True
                
                # Update key if this sequence is longer
                if seq_len > group_len:
                    sequences = length_groups.pop(group_len)
                    length_groups[seq_len] = sequences
                break
        
        # Create new group if none found
        if not assigned:
            length_groups[seq_len] = [seq]
    
    return length_groups

def seqs_to_matrix(seq_list, convert_aas=True, pad=np.nan, tolerance=10):
    '''
    function to take in a list of sequences and convert them to a matrix
    of values that can be used downstream for calculating things. 
    
    Parameters
    -----------
    seq_list : list
        List of sequences to convert to a matrix. 
        Sequences must be the same length or will be padded with nans.
    convert_aas : bool
        If True, will convert amino acids to numbers. 
        Default is True.
        If False, will raise an exception if there are no amino acid characters
        in the sequence
        Note: Conversion is
            * ``B -> N``
            * ``U -> C``
            * ``X -> G``
            * ``Z -> Q``
    pad : np.nan
        Value to use for padding. Default is np.nan.

    Returns
    --------
    seq_matrix : np.ndarray
        Matrix of sequences. 
        Rows are sequences and columns are positions in the sequence.
        If convert_aas is True, will be a matrix of numbers. 
    '''
    # make mapping
    AA_TO_INT = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 
            'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
    AA_TO_INT_CONVERT = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 
            'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'B':11, 'U':1, 'X':5, 'Z':13}
    
    # choose mapping
    chosen_mapping = AA_TO_INT_CONVERT if convert_aas else AA_TO_INT
    
    # make dict of seqs by length
    length_groups = group_sequences_by_length(seq_list, tolerance=tolerance)

    # make dict to hold the seq_matrix, valid_mask, and seq_lengths
    seq_matrix_dict={}

    for group_len, group_seqs in length_groups.items():
        # get dimensions
        num_seqs = len(group_seqs)
        max_length = group_len
        act_lens = [len(seq) for seq in group_seqs]
        
        # pre-allocate output matrix
        seq_matrix = np.full((num_seqs, max_length), pad, dtype=float)
        
        # fill matrix row by row
        for i, seq in enumerate(group_seqs):
            try:
                # vectorized conversion using list comprehension (still fastest for this operation)
                encoded_seq = [chosen_mapping[char] for char in seq]
                seq_matrix[i, :len(encoded_seq)] = encoded_seq
            except KeyError as e:
                raise ValueError(f"Invalid character in sequence {i}: {seq}")
        valid_mask = ~np.isnan(seq_matrix)
        seq_lengths = np.sum(~np.isnan(seq_matrix), axis=1)
        seq_matrix_dict[group_len] = (seq_matrix, valid_mask, seq_lengths)

    return seq_matrix_dict

"""
s1='GAQTAGSRDG'
s2='EKRKRRKRREDEKEKDEKEE'
s3='KDEDRETKKRGVRTKNAASTKKEDENNHEDDEEADEQDDS'
test=seqs_to_matrix([s1, s2, s3], convert_aas=True, pad=np.nan)
print(test)
print(np.nansum(test))
lengths=np.array([np.count_nonzero(~np.isnan(a)) for a in test]).reshape(3, 1)
# use lengths to get th values in test up to the nan
"""