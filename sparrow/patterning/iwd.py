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
def __compute_IWD_from_binary_mask(binary_mask, weights=None):
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

    weights : list 
        A list were each element is a real number that is used to weight the 
        IWD calculation at a per residue level. 

    Returns
    ----------------
    float
        Returns the IWD value, where is a number between 0 and some 
        positive value.

    """

    # if no weights are passed build null weight mask
    if weights is None:
        weights = [1]*len(binary_mask)

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

        all_hits[position] = resi_distances*weights[position]

    if len(all_hits) > 0:
        return sum(all_hits.values())/len(all_hits)
    else:
        return 0



# ------------------------------------------------------------------
#
def calculate_average_inverse_distance_charge(mask, sequence, charge=['-','+']):
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
    mask : list
        Linear net charge per residue (NCPR) as list calculated accross the sequence. For 
        more information on how to calculate the NCPR see SPARROW

    sequence : list
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
    
    # dictionary to select delimiter function based on passed charge prefference 
    delimiter_function = {'-':lambda a: a in ['D','E'], '+':lambda a: a in ['R','K']}[charge]

    # dictionary to test map prefered charge to sign of NCPR     
    charge_tester = {'-':-1.0,'+':1.0}
    
    # dictionary of empty values for each index point 
    # (extracts all hit residues based on charge definition)
    all_hits = {i:0 for i in find_all_indices(sequence, delimiter_function)} 
    hits = np.array([i for i in all_hits.keys()])
    
    # iterate through index (here p is an index in the sequence)
    for i, p in enumerate(hits):
        resi_distances= 0
        # iterate through pairs for that index 
        for p1 in hits[np.arange(len(hits))!=i]:
            resi_distances += 1 / np.abs(p1-p)

        # multiply the residue charge by the resi_distances
        # if NCPR of residue in the mask is opposite of the charge being evaluated set charge to 0 
        if np.sign(mask[p]) == charge_tester[charge]:
            all_hits[p] = abs(mask[p])*resi_distances
        else:
            all_hits[p] = 0*resi_distances
        
    if len(hits) > 0:
        return sum(all_hits.values())/len(hits)
    else:
        return 0

# ------------------------------------------------------------------
#
def calculate_average_bivariate_inverse_distance_charge(mask, sequence):
    """
    Function returns the charge-weight average inverse distance BETWEEN positive and negative
    residues in an amino acid sequence. This function is similar to the 
    calculate_average_bivariate_inverse_distance in its calculation except it is specific to charge
    residues and is weigthed by the difference in NCPR between charge pairs. 

    The per residue pair bivariate inverse distance is weighted by the difference in charge 
    between the two oppositly charged residues of intrest: 
    (charge2 - charge1) / (distance2 - distance1)

    This metric can be used to quantify the relitive clustering between charge residues or 
    the interdespersion of charge residues in a sequence relitive to each other. 

    Parameters
    -------------
    mask : list
        Linear net charge per residue (NCPR) as list calculated accross the sequence. For 
        more information on how to calculate the NCPR see SPARROW

    sequence : list
        Amino acid sequence spit into list, where element 0 is the first residue in sequence and 
        element N-1 is the Nth residue in the sequence
  
    Returns
    --------
    float 
        Returns average bivariate charge-weighted IWD value for the sequence based on the passed,
        NCPR mask and sequence. NOTE the intuition of bivariate clustering the parameter is nuanced 
        in that the higher the value the more interspersed the the positive and negative residues are. 
    """
    
    # dictionary to select delimiter function based on passed charge prefference 
    delimiter_function = {'-':lambda a: a in ['D','E'], '+':lambda a: a in ['R','K']}

    #  dictionary to test map prefered charge to sign of NCPR     
    charge_tester = {-1.0:['D','E'], 1.0:['R','K'], 0:[None]}
    
    # adjust charge mask such that charged residues with a charge opposite to that of the 
    # their defined charge are automatically set to zero. This is important to do so that during 
    # the charge difference calculation for weighting the difference of between charges is always
    # calculated a the difference between a - & + charge 
    l_mask = [c if sequence[i] in charge_tester[np.sign(c)] else 0 for i,c in enumerate(mask)]
    
    # dictionary of empty values for each index point for both catagories
    all_neg_hits = {i:0 for i in find_all_indices(sequence, delimiter_function['-'])} 
    neg_hits = np.array([i for i in all_neg_hits.keys()])
    
    all_pos_hits = {i:0 for i in find_all_indices(sequence, delimiter_function['+'])}
    pos_hits = np.array([i for i in all_pos_hits.keys()])
    
    
    # make sure both catigories are populated with hits 
    if len(neg_hits) == 0 or len(neg_hits) == 0:
        return 0
    
    else:
    
        # iterate through index
        for i, p in enumerate(neg_hits):
            resi_distances= 0
            # iterate through pairs for that index 
            for p1 in pos_hits:
                # sum over inverse distance
                # resi_distances += 1 / np.abs(p1-p)
                
                # sum over charge difference over linear distance
                resi_distances += np.abs(l_mask[p1]-l_mask[p]) / np.abs(p1-p)

            # save to all_hits
            all_neg_hits[p] = resi_distances
        
        
        # return bivariate average IWD of pairs 
        return sum(all_neg_hits.values())/(len(neg_hits)+len(pos_hits))


# ------------------------------------------------------------------
def find_all_indices(input_list, boolean_function):
    """
    Function that returns the list indices where values in the input_list
    are True as assessed by the boolean function.

    Parameters
    -------------
    input_list : list
        List of elements, where each element will be evaluated by the 
        boolean_function()

    boolean_function : function
        Function which takes each element from input_list and must return
        either True or False. 

    Returns
    --------
    list
        Returns a list of indices where the input_list elements 
        where True
    """

    indices = []

    # cycle over each element in input_list
    for idx, value in enumerate(input_list):

        # if an element is True (as assessed by the boolean_function)
        # then add that index to the indices list
        if boolean_function(value):
            indices.append(idx)

    return indices
