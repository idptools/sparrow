import numpy as np
cimport numpy as np
cimport cython 

from cpython cimport array
import array

from libc.stdlib cimport rand, srand, RAND_MAX

from .. import sparrow_exceptions

def calculate_sigma(str seq, list group1, list group2):
    """
    User-facing function to calculate the sigma parameter. Sigma captures
    the asymmetry between two groups of residues in a sequence, and is 
    defined as 

    (fraction group 1 - fraction group 2)^2 / (fraction group 1 + fraction group 2)

    For more information see Das & Pappu 2013, who define sigma in terms
    of charged residues, i.e. group1 = ['K','R'] and group2 = ['E','D'].
    
    NB: sigma is used to calulate kappa, but we have a more efficient
    implementation for actually doing the kappa calculation, so if you
    wish to calculate kappa we recommend using kappa_x() in this 
    module.

    Parameters
    --------------
    seq : str
        Amino acid sequence. Note that, comparison here is done between
        residues in seq and those in group1 and group1.

    group1 : list
        A list of one or more single amnino acids compared for calculating
        delta.

    group2 : list
        A list of one or more single amnino acids compared for calculating
        delta.

    Returns
    ----------
    float
        sigma is returned as a positive float

    """

    total_count = 0
    net_count = 0
    seqlen = len(seq)
    
    for i in seq:
        
        if i in group1:
            total_count = total_count + 1
            net_count   = net_count + 1
            
        if i in group2:
            total_count = total_count + 1
            net_count   = net_count - 1

    if total_count == 0:
        return 0

    net_count = net_count/seqlen
    total_count = total_count/seqlen
    
    return (net_count**2) / total_count



def calculate_delta(str seq, list group1, list group2, int window_size):
    """
    User-facing function to calculate the delta parameter. Delta captures
    the local deviation in sequence patterning on the lengthscale defined
    by window_size compared to the overal sequence assymetry. i.e., how 
    different, on average, are the window_size regions of the sequence in
    terms of assymetry to one another vs. the overall sequence average.

    For more information see Das & Pappu 2013, who define delta in terms
    of charged residues, i.e. group1 = ['K','R'] and group2 = ['E','D'].
    
    NB: delta is used to calulate kappa, but we have a more efficient
    implementation for actually doing the kappa calculation, so if you
    wish to calculate kappa we recommend using kappa_x() in this 
    module, which takes in the same input as this function signature.

    Parameters
    --------------
    seq : str
        Amino acid sequence. Note that, comparison here is done between
        residues in seq and those in group1 and group1.

    group1 : list
        A list of one or more single amnino acids compared for calculating
        delta.

    group2 : list
        A list of one or more single amnino acids compared for calculating
        delta.

    window_size : int
        Size of sliding window used to calculate the local sequence assymmetry

    Returns
    ----------
    float
        delta is returned as a positive float

    """

    delta = 0

    sigma = calculate_sigma(seq, group1, group2)
    nblobs = (len(seq)-window_size)+1

    for i in range(0, nblobs):
        blob = seq[i:i+window_size]

        blob_sigma = calculate_sigma(blob, group1, group2)

        delta = delta + ( ((sigma - blob_sigma)**2) / nblobs)

    return delta
    

    

# ....................................................................................................
#
def kappa_x(str seq, list group1, list group2, int window_size, int flatten=0):
    """
    User-facing high-performance implementation for generic calculation of 
    kappa_x. We use this for calculating real kappa (where group1 and 
    group2 are ['E','D'] and ['R','K'], respectively but the function can 
    be used to calculate arbitrary kappa-based patterning.

    NB1: kappa will return as -1 if 

    1. the sequece is shorter than the windowsize
    2. There are no residues from either group1 or group2

    The function will raise an exception if the windowsize is < 2

    NB2: kappa is defined as comparing the ratio of delta with deltamax, 
    where *in this implementation* deltamax refers to the delta associated
    with the most segregated sequece; e.g::

        (AAA)n-(XXX)m-(BBB)p
    
    Sometimes, when the charge asymmetry is VERY highly skewed, this most
    highly segregated sequence does not give the highest delta value, such
    that we can get a kappa greater than 1. This only occurs in situations
    where kappa is probably not a useful metric anyway (i.e 100x excess of 
    one group residue vs. another). We recommend setting the 'flatten'
    keyword to True, which means kappa values over 1 will be flatteed to 1.

    NB3: this implementation differs very slightly from the canonical 
    kappa reference implementation; it adds non-contributing 'wings' of 
    the windowsize onto the N- and C-termini of the sequence. This
    means residue clusters at the end contribute to the overall sequence 
    patterning as much as those in the middle, and also ensures we can 
    analytically determine the deltamax sequence for arbitrary    
    windowsizes.

    This both addresses a previous (subtle) limitation in kappa, but also 
    buys a ~100x speedup compared to previous reference implementations. 
    As a final note, I (Alex) wrote the original reference implementation 
    in localCIDER, so feel comfortable criticising it's flaws!

    NB4: If no residues are provided in group2 then the function assumes
    all residues not defined in group1 are in group2 and the function
    becomes a binary patterning function instead of a ternary pattering
    function.

    Parameters
    -----------
    seq : str
        Amino acid sequence (or any other string)

    group1 : list
        List of characters used to define patterning alphabet. Residues
        in this group are considered to be part of group 1. Must be 1
        or more elements.

    group2 : list
        List of characters used to define patterning alphabet. Residues
        in this group are considered to be part of group 2. If NOT provided
        then kappa converts into a binary parameter and every residue not 
        defined in group1 is considered to be group2.

    window_size : int
        Window size to be used when computing local patterning. Cannot be 
        larger than the sequence length or an exception will be raised.
        
    flatten : int
        Flag which, if set to 1, means if kappa is above 1 the function
        will flatten the value to 1. Default = 0 (non flatten).

    Returns
    ------------
    float 
        Returns a single float that lies between 0 and 1. The value reports on the likleihood of observing a 
        sequence more evenly mixed than The real sequence by chance. If this approaches 0 it says likelyhood of 
        observing a more well-mixed sequence becomes vanishingly small. As this approaches 1 it says likelihood 
        of observing a more well-mixed sequence is more and more likely.

    Raises
    -----------
    sparrow.sparrow_exceptions.PatterningException

    
    """


    # decare
    cdef int string_length, winged_string_length, i, count_1, count_2;
    cdef float wt_sigma, kappa_val


    # initialize
    string_length = len(seq)
    wing_size = int(window_size)    
    winged_string_length = wing_size*2 + string_length

    cdef array.array seq_ternary_winged = array.array('i', [0]*(winged_string_length))


    ## Block of sanity check
    # --------------------------------------------------------
    if window_size > len(seq):
        return -1

    if window_size < 2:
        raise sparrow_exceptions.PatterningException('window_size must be 2 or larger')

    if len(group1) == 0:
        raise sparrow_exceptions.PatterningException('Passed group1 is empty, group1 must contain one or more possible residues')


    # --------------------------------------------------------
    
    seq_ternary_winged = ternarize(seq, group1, group2, seq_ternary_winged, wing_size, string_length)

    count_1 = 0
    count_2 = 0
    for i in range(winged_string_length):
        if seq_ternary_winged[i] == 1:
            count_1 = count_1 + 1
        elif seq_ternary_winged[i] == -1:
            count_2 = count_2 + 1
            
    if count_1 == 0:
        if len(group1) > 0:
            return -1.0

    if count_2 == 0:
        if len(group2) > 0:
            return -1.0
        
    # --------------------------------------------------------
    wt_sigma = global_compositional_asymmetry(seq_ternary_winged, winged_string_length)

    delta  = patterning_asymmetry(seq_ternary_winged, window_size, wt_sigma, winged_string_length)

    delta_max = deltamax_patterning_asymmetry(seq_ternary_winged, wing_size, window_size, wt_sigma, winged_string_length, string_length)

    kappa_val = delta/delta_max

    # oldschool intbool - anything non-zero is true
    if flatten != 0:
        if kappa_val > 1:
            return 1.0

    return kappa_val
            



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

    If group2 is empty, then this function becomes a binarizing function, whereby we just
    set elements in ternarized_winged to zero or 1. 


    Parameters
    --------
    seq : str
        Amino acid string 

    group1 : list
        List of one or more residues for which membership of every position in the
        seq is compared against

    group2 : list
        List of zero or more residues for which membership of every position in the
        seq is compared against. If group2 is zero then every residue not defined
        in group1 is cosidered to be group2.

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

    # if no residues are passed for group2 then EVERY residue not in
    # group1is considered
    # to be in group 2
    if len(group2) == 0:

        # note in this scenario we're really doing a binerize operation        
        for i in range(string_length):
            if seq[i] in group1:
                ternarized_winged[i+wing_size] = 1
            else:
                ternarized_winged[i+wing_size] = -1

    # here's where we generate a ternaized sequence
    else:
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

    This calculates the parameter defined as 'sigma' in the original Das & Pappu paper.

    Parameter
    -----------
    ternary_array : array.arry of type int
        Array where each element is 0, 1, or -1.
 
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
        
        if ternary_array[i] == 1:
            count_A = count_A + 1
            
        elif ternary_array[i] == -1:
            count_B = count_B + 1
            
    # if none found return 0
    if (count_A + count_B) == 0:
        return 0
            
    # fraction of A and B
    FA = count_A/string_length
    FB = count_B/string_length



    # compute compositional asymmetry as (Fa - Fb)^2 / (Fa + Fb)
    return (FA-FB)*(FA-FB) / (FA + FB)

# ....................................................................................................
#
@cython.boundscheck(False)
@cython.wraparound(False) 
cdef float patterning_asymmetry(array.array seq_ternary_winged, 
                                int window_size, 
                                float overall_asymmetry, 
                                int winged_string_length): 
    """
    Function that takes a ternary sequence (ternary_seq) and computes the compositional 
    patterning compared to the overall sequence asymetry.

    This corresponds to the parameter 'delta' as described by Das and Pappu.
    
    Parameters
    ------------
    ternary_seq : array.array of type Int
        Integer array where each element is either 0 or 1

    window_size : int
        Window size over where local asymmetry is calculated

    overall_asymmetry : float
        Overall sequence compositional asymmetry, as calculated by 
        gobal_composition_asymmetry()

    string_length : int
        Pre-computed length of the array

    Returns
    -----------
    float
        Returns a float value that reports on the local patterning 
        for this binarized sequence. A lower value reflects a more 
        evenly distributed value
        

    """

    cdef int end, i, j, total;
    cdef float local_sum, window_asymmetry, FA, FB

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

        if total_A + total_B == 0:
            window_asymmetry = 0
        else:
            
            FA = total_A/window_size
            FB = total_B/window_size
                    
            # compute local window asymmetry
            window_asymmetry = (FA - FB)*(FA - FB) / (FA + FB)
    
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
        

            
                
