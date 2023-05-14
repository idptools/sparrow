import numpy as np
from sparrow import sparrow_exceptions

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

    # build a binary mask based on the sequence and the set of target
    # residues passed
    binary_mask = [1 if i in target_residues else 0 for i in sequence]

    # compute IWD 
    return __compute_IWD_from_binary_mask(binary_mask)



# ....................................................................................................
#
def __compute_IWD_from_binary_mask(binary_mask, weights=None):
    """
    Internal function that actually computes the inverse weighted distance
    from a binary mask. This function should not be called directly, but
    instead should be called by functions inside this module. This is the
    single place where the monovariate IWD should be calculated.

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
    hit_indices = [idx for idx, value in enumerate(binary_mask) if value == 1]

    # if we fouond no hit indices then return 0 and be done!
    if len(hit_indices) == 0:
        return 0

    # Initialize a dictionary to store the intial hit value 
    inverse_distance_sums = {idx: 0 for idx in hit_indices}

    # Now cycle over each position that was '1' and calculate the index
    # position between 
    for position in hit_indices:

        total_distance = 0
        
        # For every OTHER hit index  
        for other_position in hit_indices:
            if other_position != position:
                total_distance = total_distance + ( 1 / abs(other_position - position))

        # Update the total distance for the current position, weighted by the corresponding weight
        inverse_distance_sums[position] = total_distance * weights[position]

    
    return sum(inverse_distance_sums.values()) / len(inverse_distance_sums)


# ------------------------------------------------------------------
#
def calculate_average_inverse_distance_charge(linear_NCPR, sequence, charge):
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
    linear_NCPR : list
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

    if charge not in ['-','+']:
        raise sparrow_exceptions('charge parameter must be either "+" or "-"')
        
    # Define delimiter functions for amino acids carrying negative and positive charge
    is_negatively_charged = lambda a: a in ['D', 'E']
    is_positively_charged = lambda a: a in ['R', 'K']

    # choose the appropriate delimiter function based on the charge preference
    if charge == '-':
        is_charged = is_negatively_charged
        charge_value = -1.0
    else:
        is_charged = is_positively_charged
        charge_value = 1.0
        
    # get indices of charged residues as per the requested charge
    # sign
    charged_indices = find_all_indices(sequence, is_charged)

    # if we found no charged residues then return 0 and be done!
    if len(charged_indices) == 0:
        return 0

    # initialize an empty dictionary
    inverse_distance_sums = {}

    # for each index where a charged residue is found
    for p in charged_indices:
        
        total_distance = 0

        # for every other index
        for p1 in charged_indices:

            # if not the same one
            if p != p1:
                total_distance = total_distance + ( 1 / abs(p1 - p))

        # if for this residue the NCPR-weighted charge matches
        # the sign of charge clustering we're interested in
        if np.sign(linear_NCPR[p]) == charge_value:
            inverse_distance_sums[p] = abs(linear_NCPR[p]) * total_distance
        else:
            inverse_distance_sums[p] = 0

    # return average
    return sum(inverse_distance_sums.values()) / len(inverse_distance_sums)
    
    
# ------------------------------------------------------------------
#
def calculate_average_bivariate_inverse_distance_charge(linear_NCPR, sequence):
    """
    Function returns the charge-weight average inverse distance BETWEEN positive and negative
    residues in an amino acid sequence. This function is similar to the 
    calculate_average_bivariate_inverse_distance in its calculation except it is specific to charge
    residues and is weighted by the difference in NCPR between charge pairs. 

    The per residue pair bivariate inverse distance is weighted by the difference in charge 
    between the two oppositly charged residues of intrest: 
    (charge2 - charge1) / (distance2 - distance1)

    This metric can be used to quantify the relitive clustering between charge residues or 
    the interdespersion of charge residues in a sequence relitive to each other. 

    Parameters
    -------------
    linear_NCPR : list
        Linear net charge per residue (NCPR) as list calculated accross the sequence. For 
        more information on how to calculate the NCPR see SPARROW. 

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

    # lambda functions to find negative and positive residues
    is_negatively_charged = lambda a: a in ['D', 'E']
    is_positively_charged = lambda a: a in ['R', 'K']

    #  dictionary to test map prefered charge to sign of NCPR     
    charge_tester = {-1.0:['D','E'], 1.0:['R','K'], 0:[None]}
    
    # adjust linear_NCPR such that charged residues with a charge opposite to that of the 
    # their defined charge are automatically set to zero. This is important to do so that during 
    # the charge difference calculation for weighting the difference of between charges is always
    # calculated a the difference between a - & + charge 
    l_linear_NCPR = [c if sequence[i] in charge_tester[np.sign(c)] else 0 for i,c in enumerate(linear_NCPR)]
    
    # find the indices of D/E/R/K residues 
    negative_indices = find_all_indices(sequence, is_negatively_charged)
    positive_indices = find_all_indices(sequence, is_positively_charged)

    # make sure both catigories are populated with hits 
    if len(negative_indices) == 0 or len(positive_indices) == 0:
        return 0

    inverse_distance_sums = {}
    # Calculate the sum of charge differences over linear distances. Specifically, 
    # for each index that has a negative residue associated with it..    
    for p in negative_indices:
        
        # for this residue initialize the total at 0
        total = 0
        
        # for each positive residue in the sequence, calculate the absolute difference
        # in charge divided by the distance in linear sequence space, adding this
        # to a total score
        for p1 in positive_indices:
            total = total + abs(l_linear_NCPR[p1] - l_linear_NCPR[p]) / abs(p1 - p)

        # finally, that total is associated with all the inverse distance sums
        # at position p
        inverse_distance_sums[p] = total    
   
    # return average
    return sum(inverse_distance_sums.values()) / len(inverse_distance_sums)


    

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
