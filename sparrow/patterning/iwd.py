import numpy as np



# ....................................................................................................
#
def calculate_average_inverse_distance_from_sequence(sequence, target_residues):
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

    binary_mask = []
    for i in sequence:
        if i in target_residues:
            binary_mask.append(1)
        else:
            binary_mask.append(0)

    # compute IWD 
    return __compute_IWD_from_binary_mask(binary_mask)



# ....................................................................................................
#
def __compute_IWD_from_binary_mask(binary_mask):
    """
    Internal function that actually computes the inverse weighted distance
    from a binary mask. This function should not be called directly, but
    intsead should be called by functions inside this module. This is the
    single place where the monovariate IWD should be calculated .

    Parameters
    ----------------
    binary_mask : list
        A list where each element is either 0 or 1. This mask is then used
        to calculate the inverse weighted distance.

    Returns
    ----------------
    float
        Returns the IWD value, where is a number between 0 and some 
        positive value.

    """

    # get the indices where the binary mask is 1
    hit_indices = []
    all_hits = {}
    
    for idx, value in enumerate(binary_mask):
        if value == 1:
            hit_indices.append(idx)
            all_hits[idx] = 0

    # convert to an array so we can reference multiple positions with 
    # slice notation
    hit_indices = np.array(hit_indices)
            
    # now cycle over each position that was '1' and calculate the index
    # position between 
    for index, position in enumerate(hit_indices):
        resi_distances = 0
        
        # for every OTHER hit index  
        for other_position in hit_indices[np.arange(len(hit_indices)) != index]:
            resi_distances = resi_distances + 1/(np.abs(other_position - position))

        all_hits[position] = resi_distances

    if len(all_hits) > 0:
        return sum(all_hits.values())/len(all_hits)
    else:
        return 0
