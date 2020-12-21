import numpy as np
cimport numpy as np
cimport cython 

from cpython cimport array
import array

from libc.stdlib cimport rand, srand, RAND_MAX

from . import sparrow_exceptions 



# ....................................................................................................
#
def patterning_percentile(str seq, list group, int window_size, int count, int seed):
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
        Seed used to initialize C random number generator

    Returns
    ------------
    float 
        Returns a single float that lies between 0 and 1. The value reports on the likleihood of observing a 
        sequence more evenly mixed than The real sequence by chance. If this approaches 0 it says likelyhood of 
        observing a more well-mixed sequence becomes vanishingly small. As this approaches 1 it says likelihood 
        of observing a more well-mixed sequence is more and more likely.

    """

    cdef int string_length
    cdef float wt_sigma
    string_length = len(seq)
    cdef array.array seq_binary = array.array('i', [0]*string_length)
    cdef array.array random_shuffle_pvals = array.array('f', [0]*count)

    ## Block of sanity check
    # --------------------------------------------------------
    if window_size > len(seq):
        raise sparrow_exceptions.PatterningException('window_size is larger than sequence length')

    if window_size < 2:
        raise sparrow_exceptions.PatterningException('window_size must be 2 or larger')

    # make sure seed is an integer
    try:
        seed = int(seed)
    except:
        raise sparrow_exceptions.PatterningException('Cannot convert seed into an integer')

    if len(group) == 0:
        raise sparrow_exceptions.PatterningException('Passed group is empty, group must contain one or more possible residues')

    # --------------------------------------------------------
     
    seed_C_rand(seed)

    seq_binary = binerize(seq, group, seq_binary, string_length)
        
    wt_sigma = global_compositional_asymmetry(seq_binary, string_length)

    real_delta  = patterning_asymmetry(seq_binary, window_size, wt_sigma, string_length)
    
    for i in range(count):
        random_shuffle_pvals[i] = patterning_asymmetry(shuffle_array(seq_binary, string_length), window_size, wt_sigma, string_length)
        
    # sort all patterning values
    np.asarray(random_shuffle_pvals).sort(kind='quick')    

    # find percentile 
    for i in range(count):
        if real_delta < random_shuffle_pvals[i]:
            return i/count

    return 1.0

# ....................................................................................................
#
cdef seed_C_rand(int seedval):
    """
    Function that initializes C's rand() function with
    a seed value. Without this the same seed is used every
    time..

    Parameters
    ------------
    seedval : int
       Non-negative integer seed

    Returns
    --------
    None
      No return but sets the seed!
    

    """
    srand(seedval)


# ....................................................................................................
#
# Turn off bounds checking for performance boosts!
@cython.boundscheck(False)
@cython.wraparound(False) 
cdef array.array shuffle_array(array.array binary_array, int string_length):
    """

    Cython function that takes an integer array (binary_array) of length
    string_length and shuffles it in place. This does not create a copy
    but literally shuffles the array that's passed in.

    Parameters:
    --------------
    binary_array : array.array of type int
        An array of type Int that will be shuffled

    string_length : int
        Length of the array - passed directly just so we don't have to
        recompute this every time

    Returns
    ---------:
    array.array of type int
        Returns an array of the same size as was originally passed in but
        has the positions shuffled. Note that actually returning this array
        is not strictly necessary because we shuffled the array in place,
        but returning it makes the code a bit easier to work with. 

    More info
    ----------
        This is a Cython function, which means it can only be called from
        within a .pyx file
    
    """

    # initialize
    cdef int i, r, val;

    # for each position in the string
    for i in range(string_length):
        
        # randomly select ANOTHER poistion in the array
        j = i + int(rand() / (int(RAND_MAX / ( string_length - i) + 1)))

        # extract j-th position
        val = binary_array[j]

        # assign jth position to what is currently ith position
        binary_array[j] = binary_array[i]

        # assign ith position to what WAS the jth position
        binary_array[i] = val

    return binary_array


# ....................................................................................................
#
cdef array.array binerize(str seq, list group, array.array binerized, int string_length):
    """
    Function that takes in an amino acid sequence and converts it into a binary integer 
    array. Specifically, each position is set to 0 depending on if the residue in question
    is present or absent from the group list.


    Parameters
    --------
    seq : str
        Amino acid string 

    group : list
        List of one or more residues for which membership of every position in the
        seq is compared against

    binerized : array.array
        Pre-allocated integer array of the same length as the string

    string_length : int 
        Length of the seq and binerized arrays

    Returns
    ------------
    array.array
       Returns a binary integer array where each position is either 1 or 0 
       as defined by the input list
        
    """
    
    cdef int i

    for i in range(string_length):
        if seq[i] in group:
            binerized[i] = 1
        else:
            binerized[i] = 0

    return binerized


# ....................................................................................................
#
@cython.boundscheck(False)
@cython.wraparound(False) 
cdef float global_compositional_asymmetry(array.array binary_array, int string_length):
    """
    Function that computes the global compositional asymmetry for the entire string. This value 
    tends to 1 when the two components are highly assymetric and to 0 when they are highly
    similar in abundance.

    Parameter
    -----------
    binary_array : array.arry of type int
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
    cdef int count = 0;
    cdef float FA, FB

    # for each element in the integer binary array ..
    for i in range(string_length):
        count = count + binary_array[i]
        
    # fraction of A and B
    FA = count/string_length
    FB = 1 - FA

    # compute compositional asymmetry as (Fa - Fb)^2
    return (FA-FB)*(FA-FB)

# ....................................................................................................
#
@cython.boundscheck(False)
@cython.wraparound(False) 
cdef float patterning_asymmetry(array.array binary_seq, int window_size, float overall_asymmetry, int string_length): 
    """
    Function that takes a binary sequence (binary_seq) and computes the compositional patterning compared to the
    overall sequence assymetry.

    Parameters
    ------------
    binary_seq : array.array of type Int
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

    end = (string_length - window_size) + 1

    local_sum = 0

    # for each possible window
    for i in range(end):

        # extract out the window sequence
        f = binary_seq[i:i+window_size]
        
        # compute the local fraction of hits within 
        # the window
        total = 0
        for j in range(window_size):
            total = total + f[j]
        FA = total/window_size
                    
        # compute local window asymmetry
        window_asymmetry = (FA - (1-FA))*(FA - (1-FA))
    
        # how much does local window asymmetry deviate from overall asymmetry 
        local_sum = (window_asymmetry - overall_asymmetry) * (window_asymmetry - overall_asymmetry) + local_sum

    return local_sum / end
        
        
