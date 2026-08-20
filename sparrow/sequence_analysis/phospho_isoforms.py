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

import itertools

from sparrow.sparrow_exceptions import SparrowException


## -----------------------------------------
##
def _predict_all_phosphosites(protein):
    """
    Gets list of predicted phosphosites

    BASED OFF OF:
        predictors in sparrow:
        https://github.com/idptools/sparrow/tree/main/sparrow/predictors/phosphorylation

    Parameters
    ------------
    protein : sparrow.Protein
        sparrow Protein object


    Returns:
    ----------
    list
        list of predicted positions of sites of phosphorylated T, S, and Y
        Note positions are returned as indexed from 0

    """

    # predict phosphosites
    pS = protein.predictor.serine_phosphorylation(return_sites_only=True)
    pT = protein.predictor.threonine_phosphorylation(return_sites_only=True)
    pY = protein.predictor.tyrosine_phosphorylation(return_sites_only=True)

    return list(pS + pT + pY)


## ----------------------------------------
##
def _get_all_phosphosites(sequence):
    """
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

    Raises
    ----------
    sparrow.sparrow_exceptions.SparrowException
        If phospho_rate is not between 0 and 1, or if a phosphosite index falls
        outside the sequence.
    """

    # without this check an out-of-range rate silently yields an empty list (for a
    # negative rate) or the full 2^n set (for a rate above 1)
    if phospho_rate < 0 or phospho_rate > 1:
        raise SparrowException(f"phospho_rate must be between 0 and 1 (got {phospho_rate})")

    for site in phosphosites:
        if site < 0 or site >= len(sequence):
            raise SparrowException(
                f"Phosphosite index {site} is outside the sequence "
                f"(length {len(sequence)}). Indices are expected to be 0-based."
            )

    _max_phospho_number = int(len(phosphosites) * phospho_rate)
    ## GET ALL phospho-sequence combinations
    phosphoSeqome = []
    for phosphostatus in itertools.product("01", repeat=len(phosphosites)):
        if phosphostatus.count("1") > _max_phospho_number:
            continue
        newseq = list(sequence)

        indx = 0
        # look over each element in our phosphosite on/off list
        for i in phosphostatus:
            # if that element is ON
            if int(i) == 1:
                # set the AA at that position to a negative residue (we use E but
                # could be D)
                newseq[phosphosites[indx]] = "E"
            indx = indx + 1

        # now we've replaced some number of T/Y/S with E representing a different
        # phosphostate
        newseq = "".join(newseq)
        phosphoSeqome.append(newseq)

    return phosphoSeqome


## -----------------------------------
##
def get_phosphoisoforms(protein, mode="all", phospho_rate=1, phosphosites=None):
    """Phosphosites are replaced with the phosphomimetic 'E', enabling approximate calculation
    of charge based sequence features with the presence of a phosphorylated residues.

    Parameters
    ----------
    protein : sparrow.Protein
        sparrow Protein object

    mode : str, optional
        Defition for how the phosphosites should be determined, by default "all"

        'all'       : Assumes all S/T/Y residues are potential phosphosites

        'predict'   : Leverages PARROT trained predictors via _predict_all_phosphosites
                        to predict phosphorylated sites based on sequence.

        'custom'    : uses the 'phosphosites' parameter as indices for phosphosites.

    phospho_rate : float, optional
        Value between 0 and 1 which defines the maximum percent of phosphosites
        can be 'phosphorylated' a each sequence, by default 1 (IE all sites can be
        phosphorylated)

    phosphosites : list, optional
        Custom list of 0-based indices for valid phosphosite positions. Only used
        when ``mode='custom'``, in which case it is required. Default is None.

    Returns
    -------
    list
        list of sequences for the possible phosphoisoforms based off the selected method.
        Phosphorylatable amino acids are replaced with 'E'.

    Raises
    -------
    sparrow.sparrow_exceptions.SparrowException
        If ``mode`` is not one of 'all', 'predict' or 'custom'; if
        ``mode='custom'`` but no ``phosphosites`` were given; if ``phospho_rate``
        is outside 0-1; or if a phosphosite index falls outside the sequence.
    """

    # get phosphosite positions
    if mode == "all":
        _phosphosites = _get_all_phosphosites(protein.sequence)
    elif mode == "predict":
        _phosphosites = _predict_all_phosphosites(protein)
    elif mode == "custom":
        if phosphosites is not None:
            _phosphosites = phosphosites
        else:
            raise SparrowException("To use mode='custom' the phosphosites keyword must be defined")
    else:
        raise SparrowException(f"Invalid mode passed [{mode}]. Must be one of: 'all', 'predict', 'custom'")

    # generate all phospho-Isoforms
    return _build_phosphoSeqome(protein.sequence, _phosphosites, phospho_rate=phospho_rate)
