"""
Snippit to build and adapted from localcider to get all phosphoisoforms 
of an amino acid sequence which is to integrated into sparrow

By : Garrett M. Ginell
2023-02-08

BASIC workflow is as followed:

  To get a list of run get_phosphoisoforms"

    phosphoSeqome = get_phosphoisoforms(sequence, mode='predict')
    # for options see various run variations in function header
    
  Then once you get the phosphoisoforms from the list above you can iterate
  the list and calculate a sequence parameter of choice and build a distribution: 

    parameter_list = []
    for s in phosphoSeqome:
        parameter_list.append(Protein(s).my_parameter_of_choice)
  
  This distribution can then be compared back to value of the original sequence:

    Protein(sequence).my_parameter_of_choice 
"""

from sparrow.predictors.phosphorylation import ser_phosphorylation_predictor
from sparrow.predictors.phosphorylation import thr_phosphorylation_predictor
from sparrow.predictors.phosphorylation import tyr_phosphorylation_predictor
import itertools


## -----------------------------------------
##
def _predict_all_phosphosites(sequence):
    """
    Gets list of predicted phosphosites 

    BASED OFF OF:
        predictors in sparrow:
        https://github.com/idptools/sparrow/tree/main/sparrow/predictors/phosphorylation

    Parameters
    ------------
    seq : str
        Valid amino acid sequence

    Returns:
    ----------
    list
        list of predicted positions of sites of phosphorylated T, S, and Y 
        Note positions are returned as indexed from 0

    """
    pS = ser_phosphorylation_predictor.SerPhosphorylationPredictor().predict_ser_phosphorylation(sequence, return_sites_only=True)
    pT = thr_phosphorylation_predictor.ThrPhosphorylationPredictor().predict_thr_phosphorylation(sequence, return_sites_only=True)
    pY = tyr_phosphorylation_predictor.TyrPhosphorylationPredictor().predict_tyr_phosphorylation(sequence, return_sites_only=True)

    return list(pS + pT + pY)

## ----------------------------------------
##
def _get_all_phosphosites(sequence):
    """

    BASED OFF OF:

     localcider.sequenceParameters.SequenceParameters().get_all_phosphorylatable_sites 

    Function which returns a list of all the positions which *could* be
    phosphorylated (i.e. are T/S/Y). NOTE this does not use any kind of
    smart lookup, metadata, or analysis. It's literally, where are the Y/T/S
    residues.
    Note positions are returned as indexed from 0 
    
    Parameters
    ------------
    sequence : str
        Valid amino acid sequence

    Returns:
    ----------
    list
        list of integers corresponding to S/T/Y positions in your sequence

    """ 
    sites = []
    idx = 0
    for i in sequence:
        if i in ["Y", "S", "T"]:
            sites.append(idx)
        idx = idx + 1
    return sites

## -----------------------------------
##
def _build_phosphoSeqome(sequence, phosphosites, phospho_rate=1):
    """
    Build all phospho-isoforms based on provided phosphosites

    Parameters
    ------------
    sequence : str
        Valid amino acid sequence

    phosphosites : list
        List of valid phosphosite positions

    phospho_rate : float 
        Value between 0 and 1 which defines the maximum percent of phosphosites 
        can be 'phosphorylated' a each sequence. Defult is 1 (IE all sites can be 
        phosphorylated)

    Returns
    ----------
    list
        list of sequences for all posible phospho-isoforms 
        based off of the provided inputed list of phosphosites

        When phospho_rate = 1 (100%) 
            the length of output list = 2^n where n=len(phosphosites)
    """

    _max_phospho_number = int(len(phosphosites)*phospho_rate)
    ## GET ALL phospho-sequence combinations 
    phosphoSeqome = []
    phosphoSeqome_info = []
    for phosphostatus in itertools.product("01", repeat=len(phosphosites)):
        
        if phosphostatus.count('1') > _max_phospho_number:
            continue
        newseq = list(sequence)

        count = 0
        indx = 0
        # look over each element in our phosphosite on/off list
        for i in phosphostatus:
            # if that element is ON
            if int(i) == 1:
                # set the AA at that position to a negative residue (we use E but
                # could be D)
                newseq[phosphosites[indx]] = "E"
                count+=1 
            indx = indx + 1
            
        # now we've replaced some number of T/Y/S with E representing a different
        # phosphostate
        newseq = "".join(newseq)
        phosphoSeqome.append(newseq)

    return phosphoSeqome

## -----------------------------------
##
def get_phosphoisoforms(sequence, mode="all", phospho_rate=1, phosphosites=None):
    """Phosphosites are replaced with the phosphomimetic 'E', enabling approximate calculation 
    of charge based sequence features with the presence of a phosphorylated residues.

    Parameters
    ----------
    sequence : str
        Valid amino acid sequence

    mode : str, optional
        Defition for how the phosphosites should be determined, by default "all"

        'all'       : Assumes all S/T/Y residues are potential phosphosites

        'predict'   : Leverages PARROT trained predictors via _predict_all_phosphosites
                        to predict phosphorylated sites based on sequence.
    
        'custom'    : uses the 'phosphosites' parameter as indices for phosphosites.
        
    phospho_rate : int, optional
        Value between 0 and 1 which defines the maximum percent of phosphosites 
        can be 'phosphorylated' a each sequence, by default 1 (IE all sites can be 
        phosphorylated)

    phosphosites : list, optional
        Custom list of indices for valid phosphosite positions, by default None

    Returns
    -------
    list
        list of sequences for the possible phosphoisoforms based off the selected method.
        Phosphorylatable amino acids are replaced with 'E'.    
    """

    # get phosphosite positions
    if mode == 'all':
        _phosphosites = _get_all_phosphosites(sequence)
    elif mode == 'predict':
        _phosphosites = _predict_all_phosphosites(sequence)
    elif mode == 'custom':
        if phosphosites != None:
            _phosphosites = phosphosites
        else:
            raise Exception('To use custom phosphosites must be defined')
    else:
        raise Exception('Please specify mode to compute phosphosites')

    # generate all phospho-Isoforms
    return _build_phosphoSeqome(sequence, _phosphosites, phospho_rate=phospho_rate)
