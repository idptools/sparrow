from sparrow import sparrow_exceptions, protein
from scipy.signal import savgol_filter



# .................................................................
#
def predefined_linear_track(seq, mode, window_size, end_mode, smooth=None):
    """
    Function that builds linear sequence profiles that can then be assigned to 
    Protein objects or used externally to sparrow 

    Parameters
    ----------------
    seq : str
        Amino acid sequence 

    mode : str
        Selector that defines one of a preset of possible track constructors. 
        
        'FCR'               : Fraction of charged residues
        
        'NCPR'              : Net charge per residue
          
        'aromatic'          : Fraction of aromatic residues

        'aliphatic'         : Fraction of aliphatic residues

        'polar'             : Fraction of polar residues

        'proline'           : Fraction of proline residues

        'positive'          : Fraction of positive residues
 
        'negative'          : Fraction of negative residues

        'hydrophobicity'    : Linear hydrophobicity (Kyte-Doolitle)

        'seg-complexity'    : Linear complexity

        'kappa'             : Linear charge patterning 

    

    window_size : int
        Number of residues over which local sequence properties are calculated. 
        A window stepsize of 1 is always used.
        

    end_mode : str
        Selector that defines how ends are dealt with. Empty string means nothing is
        done, but extend-ends and zero-ends ensure the track length equals the sequence
        length which can often be useful. Default is 'extend-ends'.

        'extend-ends'   |    The leading/lagging track values are copied from 
                             the first and last and values. 
    
        ''              |    Empty string means they're ignored,
    
        'zero-ends'     |    Means leading/lagging track values are set to zero.

    smooth : int or None
        Selector which allows you to smooth the data over a windowsize. Note window
        must be an odd number (applies a savgol_filter with a 3rd order polynomial
        which requires an odd number).
    


    Returns
    ----------
    list
        Returns a list of values as defined by the passed mode
    
    """

    # 
    # If additional linear analysis are useful they should be added here as a n
    # ew FX() function, and then added as allowed mode options in this function 
    # and in protein.build_linear_profile
    # 
    #
        
    
    if mode == 'FCR':
        def FX(s):
            return protein.Protein(s).FCR

    elif mode == 'NCPR':
        def FX(s):
            return protein.Protein(s).NCPR

    elif mode == 'aromatic':
        def FX(s):
            return protein.Protein(s).fraction_aromatic

    elif mode == 'aliphatic':
        def FX(s):
            return protein.Protein(s).fraction_aliphatic

    elif mode == 'polar':
        def FX(s):
            return protein.Protein(s).fraction_polar

    elif mode == 'proline':
        def FX(s):
            return protein.Protein(s).fraction_proline

    elif mode == 'positive':
        def FX(s):
            return protein.Protein(s).fraction_positive

    elif mode == 'negative':
        def FX(s):
            return protein.Protein(s).fraction_negative

    elif mode == 'hydrophobicity':
        def FX(s):
            return protein.Protein(s).hydrophobicity

    elif mode == 'seg-complexity':
        def FX(s):
            return protein.Protein(s).complexity

    elif mode == 'kappa':
        def FX(s):
            return protein.Protein(s).kappa
        
    else:
        raise sparrow_exceptions.SparrowException('Invalid mode passed to linear track: %s' %(mode))
                
    # finally build track using the specific function
    return build_track(seq, FX, window_size, end_mode, smooth)



# .................................................................
#
def linear_track_composition(seq, composition_list, window_size, end_mode, smooth=None):
    """

    Function that returns a vectorized representation of local composition
    as defined by the set of one or more residues passed in composition_list.
        
    Parameters
    ------------
        
    composition_list : list
        List where each element should be a valid amino acid

    window_size : int
        Number of residues over which local sequence properties are calculated. A 
        window stepsize of 1 is always used.

    end_mode : str
        Selector that defines how ends are dealt with. Empty string means nothing is
        done, but extend-ends and zero-ends ensure the track length equals the sequence
        length which can often be useful. Default is 'extend-ends'.

        'extend-ends'   |    The leading/lagging track values are copied from 
                             the first and last and values. 
    
        ''              |    Empty string means they're ignored,
    
        'zero-ends'     |    Means leading/lagging track values are set to zero.

    smooth : int or None
        Selector which allows you to smooth the data over a windowsize. Note window
        must be an odd number (applies a savgol_filter with a 3rd order polynomial
        which requires an odd number).


    Returns
    ----------
    list
        Returns a list of floating values that defines the density of the set of amino
        acids in the composition_list.
    
    """

    def FX(s):
        return protein.Protein(s).compute_residue_fractions(composition_list)

    return build_track(seq, FX, window_size, end_mode, smooth)



# .................................................................
#
def build_track(seq, track_function, window_size=7, end_mode='extend-ends', smooth=None):
    """
    Generic function for the construction of position-specific track parameters. Uses
    a customized function (track_function) that converts a sequence fragment to some 
    kind of value
    

    Parameters
    ------------
    seq : str
        Amino acid sequence

    track_function : fx
        Python function that takes a sequence and returns an appropriate track value

    window_size : int
        Size of sliding window to be used (stepsize is always 1). Default = 7

    end_mode : str
        Selector that defines how ends are dealt with. Empty string means nothing is
        done, but extend-ends and zero-ends ensure the track length equals the sequence
        length which can often be useful. Default is 'extend-ends'.

        'extend-ends'   |    The leading/lagging track values are copied from 
                             the first and last and values. 

        ''              |    Empty string means they're ignored,

        'zero-ends'     |    Means leading/lagging track values are set to zero.

    smooth : int or None
        Selector which allows you to smooth the data over a windowsize. Note window
        must be an odd number (applies a savgol_filter with a 3rd order polynomial
        which requires an odd number).
    

    Returns
    ------------
    list
        Returns a list of values generated by the track_function.

    """

    if len(seq) < window_size:
        raise sparrow_exceptions.SparrowException('Windowsize [%i] is larger than sequence [%i]' % (window_size,len(seq)))

    # validate smoothing param. Note we do this first because true is cast to 1 by int so have to check explicitly! 

    if smooth is not None:
        if type(smooth) is bool:
            raise sparrow_exceptions.SparrowException('If smooth provided, it should be a number that corresponds to the smoothing window')
        else:
            try:
                smooth = int(smooth)
            except Exception:
                raise sparrow_exceptions.SparrowException(f'Could not convert smooth parameter to an integer ({smooth}).')
            


    slen = len(seq)
    
    # run through window_size fragments and compute params using the
    # custom passed function
    end = (slen - window_size)+1
    track_vals = []
    for i in range(end):
        frag = seq[i:i+window_size]
        track_vals.append(track_function(frag))
    
    
    # deal with ends
    if end_mode == 'extend-ends':
        front = int(window_size/2)
        track_vals = [track_vals[0]]*front + track_vals

        end = len(seq) - len(track_vals) 
        track_vals = track_vals + [track_vals[-1]]*end 

    elif end_mode == 'zero-ends':
        front = int(window_size/2)
        track_vals = [0]*front + track_vals
        
        end = len(track_vals) - len(seq)
        track_vals = track_vals + [0]*end 
    elif end_mode == '':
        pass
    else:
        raise sparrow_exceptions.SparrowException('No valid end_mode passed')
        

    if smooth is not None:
        try:
            track_vals = savgol_filter(track_vals, smooth, 3)
        except Exception as e:
            raise sparrow_exceptions.SparrowException(f'Invalid smoothing parameter passed [smooth={smooth}]. Must be an integer between 3 and the length of the sequence')
        
    return track_vals
        
        
