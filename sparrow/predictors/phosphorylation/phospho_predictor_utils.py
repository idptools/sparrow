from sparrow.sparrow_exceptions import SparrowException

def return_hits(seq, phospho_probability, target_res, windowsize=4, threshold=0.6, return_sites_only=False):
    """
    Function that parses through a sequence and annotated phosphosite 
    probabilities to extract out specific positions or a per-residue 
    binary mask of phosphorylatio or non-phosphorylation.
    
    This function works by sliding a +/- windowsize window across the 
    sequence and if the central residue in that window has a probability 
    > threshold then all the target_res in that window are set to be 
    putative phosphosites.
        
    Parameters
    --------------
    seq : str
        Amino acid sequence

    phospho_probability : list
        A list with per-residue probabilities for a residue to have been 
        phosphorylated or not.

    windowsize : int
        Define the size of the window this algorithm uses to extend the
        influenc of a local phosphosite probability. Note the windowsize
        gets applied +/- a central position
        
    target_res : str
        A string with a single residue which each residue in the sequence 
        is compared against. 

    threshold : float
        A threshold value used to deliniate between phosphosites for masking.
        Default is 0.6.

    return_sites_only : bool
        A flag which, if set to True, means the function returns only the positions 
        found in a list. If set to False the function returns a binary mask
        list equal in length to the sequence, where '1's mean the residue
        is predicted to be a phosphosite and '0' mean they're not. Default
        is False.

    Returns
    -----------
    list 
        Returns EITHER a list (len == seq) if return_positions = False which
        contains a per-residue phosphomask (i.e. 1 = phospho 0 if not) OR
        returns a list of index positions that correspond to phosphosites.

        If return_positions is True, the function guarentees the order of 
        indices returned will be numerical

    """

    ## sanity checking first
    if len(target_res) != 1:
        raise SparrowException('Target res must be a single amino acid')
    
    if threshold > 1 or threshold < 0:
        raise SparrowException('Probability threshold used in phosphosite masking must be between 0 and 1')

    if windowsize < 1:
        raise SparrowException('Window size must be a positive integer')

    if len(seq) != len(phospho_probability):
        raise SparrowException('Sequence length and probability vector must be the same length')
    

    seqlen = len(seq)
    
    potential_hits = set([])

    if seqlen <  (2*windowsize)+1:
        raise SparrowException(f'Cannot predict phosphosites when the sequence length is less than 1+{2*windowsize}. NB: length = {seqlen}')

    # for each residue
    print(seq)
    for idx, res in enumerate(seq):

        # if this is a low-probablity residue skip and move on
        if phospho_probability[idx] < threshold:
            continue

        # if we're in the N-terminal residues just excise out a fragment of
        # varying size until we get into the sequence
        if idx < windowsize:
            slice_start = 0
            current_slice = seq[slice_start:idx+windowsize]


        # while in the 'middle' of the sequence
        elif idx >= windowsize and idx <= (seqlen - (windowsize+1)):
            slice_start = idx-windowsize
            current_slice = seq[slice_start:idx+windowsize]
            
        # at the C-terminus 
        else:
            slice_start = idx-windowsize
            current_slice = seq[slice_start:]

        # for each residue in the
        for local_idx, aa in enumerate(current_slice):
            if aa == target_res:
                global_pos = local_idx + slice_start
                
                if global_pos not in potential_hits:
                    potential_hits.add(global_pos)


    # if we just want to return the phosphoindices. Note
    # we sort these to guarentee the order of return.
    if return_sites_only:        
        return sorted(list(potential_hits))
    else:

        return_list = []
        for i in range(0,len(seq)):
            if i in potential_hits:
                return_list.append(1)
            else:
                return_list.append(0)
                
        return return_list
                    


                                    
