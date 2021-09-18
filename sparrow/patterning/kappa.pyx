import numpy as np
cimport numpy as np
cimport cython 

from cpython cimport array
import array

from libc.stdlib cimport rand, srand, RAND_MAX

from .. import sparrow_exceptions 



# ....................................................................................................
#
def kappa_x(str seq, list group1, list group2, int window_size):
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
        "1" while all others designated as 0. As such, this is the alphabet used for ternary patterning to be 
        computed over.

    window_size : int
        Window size to be used when computing local patterning. Cannot be larger than the sequence length or
        an exception will be raised


    Returns
    ------------
    float 
        Returns a single float that lies between 0 and 1. The value reports on the likleihood of observing a 
        sequence more evenly mixed than The real sequence by chance. If this approaches 0 it says likelyhood of 
        observing a more well-mixed sequence becomes vanishingly small. As this approaches 1 it says likelihood 
        of observing a more well-mixed sequence is more and more likely.

    """

    cdef int string_length, winged_string_length
    cdef float wt_sigma
    string_length = len(seq)

    wing_size = int(window_size)*2
    

    winged_string_length = wing_size*2 + string_length

    cdef array.array seq_ternary_winged = array.array('i', [0]*(winged_string_length))


    ## Block of sanity check
    # --------------------------------------------------------
    if window_size > len(seq):
        raise sparrow_exceptions.PatterningException('window_size is larger than sequence length')

    if window_size < 2:
        raise sparrow_exceptions.PatterningException('window_size must be 2 or larger')

    if len(group1) == 0:
        raise sparrow_exceptions.PatterningException('Passed group is empty, group must contain one or more possible residues')


    # --------------------------------------------------------
     
    seq_ternary_winged = ternarize(seq, group1, group2, seq_ternary_winged, wing_size, string_length)

    wt_sigma = global_compositional_asymmetry(seq_ternary_winged, winged_string_length)

    delta  = patterning_asymmetry(seq_ternary_winged, window_size, wt_sigma, winged_string_length)

    delta_max = deltamax_patterning_asymmetry(seq_ternary_winged, wing_size, window_size, wt_sigma, winged_string_length, string_length)

    return delta/delta_max



# ....................................................................................................
#
cdef array.array ternarize(str seq, 
                           list group1, 
                           list group2, 
                           array.array ternarized_winged, 
                           int wing_size, 
                           int string_length):
    """
    Function that takes in an amino acid sequence and converts it into a ternary integer 
    array. Specifically, each position is set to 0 depending on if the residue in question
    is present or absent from the group list.


    Parameters
    --------
    seq : str
        Amino acid string 

    group1 : list
        List of one or more residues for which membership of every position in the
        seq is compared against

    group2 : list
        List of one or more residues for which membership of every position in the
        seq is compared against

    ternarized : array.array
        Pre-allocated integer array of the same length as the string

    string_length : int 
        Length of the seq and ternarized arrays

    Returns
    ------------
    array.array
       Returns a ternary integer array where each position is either 1 or 0 
       as defined by the input list
        
    """
    
    cdef int i

    for i in range(string_length):
        if seq[i] in group1:
            ternarized_winged[i+wing_size] = 1
        elif seq[i] in group2:
            ternarized_winged[i+wing_size] = -1

    return ternarized_winged


# ....................................................................................................
#
@cython.boundscheck(False)
@cython.wraparound(False) 
cdef float global_compositional_asymmetry(array.array ternary_array, int string_length):
    """
    Function that computes the global compositional asymmetry for the entire string. This value 
    tends to 1 when the two components are highly assymetric and to 0 when they are highly
    similar in abundance.

    Parameter
    -----------
    ternary_array : array.arry of type int
        Array where each element is either a 0 or 1
 
    string_length : int
        Pre-computed length of the array

   Returns
    ------------
    float
        Returns a float that quantifies the overall compositional asymmetry of the sequence
    

    """

    # declare 
    cdef int i;
    cdef int count_A = 0, count_B=0;
    cdef float FA, FB

    # for each element in the integer ternary array ..
    for i in range(string_length):
        if i == 1:
            count_A = count_A + 1
        elif i == -1:
            count_B = count_B + 1
        
    # fraction of A and B
    FA = count_A/string_length
    FB = count_B/string_length

    # compute compositional asymmetry as (Fa - Fb)^2
    return (FA-FB)*(FA-FB)

# ....................................................................................................
#
@cython.boundscheck(False)
@cython.wraparound(False) 
cdef float patterning_asymmetry(array.array seq_ternary_winged, 
                                int window_size, 
                                float overall_asymmetry, 
                                int winged_string_length): 
    """
    Function that takes a ternary sequence (ternary_seq) and computes the compositional patterning compared to the
    overall sequence assymetry.

    Parameters
    ------------
    ternary_seq : array.array of type Int
        Integer array where each element is either 0 or 1

    window_size : int
        Window size over where local asymmetry is calculated

    overall_asymmetry : float
        Overall sequence compositional asymmetry, as calculated by gobal_composition_asymmetry()

    string_length : int
        Pre-computed length of the array

    Returns
    -----------
    float
        Returns a float value that reports on the local patterning for this binarized sequence. A lower
        value reflects a more evenly distributed value

    """

    cdef int end, i, j, total;
    cdef float local_sum, window_asymmetry, FA, FB

    #cdef array.array f = array.array('i', [0]*window_size)

    end = (winged_string_length - window_size) + 1

    local_sum = 0

    # for each possible window
    for i in range(end):

        # extract out the window sequence
        f = seq_ternary_winged[i:i+window_size]
        
        # compute the local fraction of hits within 
        # the window
        total_A = 0
        total_B = 0
        for j in range(window_size):
            if f[j] == 1:                
                total_A = total_A + 1
            elif f[j] == -1:
                total_B = total_B + 1
                
        FA = total_A/window_size
        FB = total_B/window_size
                    
        # compute local window asymmetry
        window_asymmetry = (FA - FB)*(FA - FB)
    
        # how much does local window asymmetry deviate from overall asymmetry 
        local_sum = (window_asymmetry - overall_asymmetry) * (window_asymmetry - overall_asymmetry) + local_sum


    return local_sum / end
        
        
# ....................................................................................................
#
@cython.boundscheck(False)
@cython.wraparound(False) 
cdef float deltamax_patterning_asymmetry(array.array seq_ternary_winged, 
                                         int wing_size, 
                                         int window_size, 
                                         float wt_sigma, 
                                         int winged_string_length, 
                                         int string_length):
    
    cdef int c_A = 0
    cdef int c_B = 0
    cdef int c_C = 0

    cdef int c_A_used = 0
    cdef int c_B_used = 0
    cdef int c_C_used = 0

    cdef array.array deltamax_seq = array.array('i', [0]*(winged_string_length))
    
    for i in range(string_length):
        if seq_ternary_winged[i+wing_size] == 1:
            c_A = c_A + 1
            
        elif seq_ternary_winged[i+wing_size] == -1:
            c_B = c_B + 1

        else:
            c_C = c_C + 1


    # build most well-mixed sequence with same composition
    for i in range(string_length):
        
        if c_A_used < c_A:
            deltamax_seq[i+wing_size] = 1
            c_A_used = c_A_used + 1

        elif c_C_used < c_C:
            deltamax_seq[i+wing_size] = 0
            c_C_used = c_C_used + 1

        else:
            deltamax_seq[i+wing_size] = -1
            c_B_used = c_B_used + 1

    delta  = patterning_asymmetry(deltamax_seq, window_size, wt_sigma, winged_string_length)

    return delta
        

            
                
