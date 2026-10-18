"""Phosphoisoforms: every phosphorylation state of a sequence, as phosphomimetic sequences.

Each candidate phosphosite (S, T or Y) is either left alone or replaced with
glutamate (E), the usual phosphomimetic, and every combination is returned as a
sequence. Computing a property over all of them - its distribution of NCPR or
kappa, say - shows how far phosphorylation could move it, which you can compare
with the value for the unmodified sequence. The approach is adapted from
localCIDER's phosphorylation isoforms (Holehouse AS, et al. CIDER: Resources to
Analyze Sequence-Ensemble Relationships of Intrinsically Disordered Proteins.
Biophys J. 2017;112(1):16-21); this implementation is by Garrett M. Ginell.

A sequence with ``n`` candidate sites has ``2**n`` isoforms, so for a sequence
with more than about twenty S/T/Y residues restrict the sites - with
``mode='predict'`` or ``mode='custom'`` - or cap how many may be phosphorylated
at once with ``phospho_rate``.

From a :class:`~sparrow.protein.Protein`, ``Protein.generate_phosphoisoforms``
calls :func:`phosphoisoforms`.

Examples
--------
>>> from sparrow import Protein
>>> protein = Protein("MKSPEEKTPRYS")
>>> isoforms = phosphoisoforms(protein)
>>> len(isoforms)                  # four S/T/Y sites
16
>>> ncpr = [Protein(isoform).NCPR for isoform in isoforms]
>>> round(protein.NCPR, 3), round(min(ncpr), 3)
(0.083, -0.25)
"""

import itertools
from typing import Any

from sparrow.sparrow_exceptions import SparrowException
from sparrow.tools.utilities import warn_renamed

# functions that lost their get_ prefix: old name -> new name. The old names
# still work, with a DeprecationWarning, for one release.
_RENAMED_FUNCTIONS = {"get_phosphoisoforms": "phosphoisoforms"}


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
def _all_phosphosites(sequence):
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
def phosphoisoforms(protein, mode="all", phospho_rate=1, phosphosites=None):
    """Every phosphoisoform of a protein, with phosphorylated sites replaced by E.

    Replacing a phosphorylated residue with the phosphomimetic glutamate gives an
    approximate way to compute charge-dependent sequence properties of the
    phosphorylated protein.

    Parameters
    ----------
    protein : sparrow.protein.Protein
        The protein.
    mode : {'all', 'predict', 'custom'}, optional
        How the candidate phosphosites are chosen. Default ``'all'``.

        * ``'all'`` - every S, T and Y residue.
        * ``'predict'`` - the sites sparrow's serine, threonine and tyrosine
          phosphorylation predictors call (``Protein.predictor``).
        * ``'custom'`` - the positions in ``phosphosites``.
    phospho_rate : float, optional
        The largest fraction of the candidate sites phosphorylated in any one
        isoform, between 0 and 1: isoforms have at most
        ``int(n_sites * phospho_rate)`` sites changed. Default 1, which allows
        every combination.
    phosphosites : list of int, optional
        0-indexed positions of the candidate sites, used only (and required) with
        ``mode='custom'``. They need not be S, T or Y. Default None.

    Returns
    -------
    list of str
        One sequence per phosphoisoform; the first is the unmodified sequence.
        With ``n`` candidate sites and ``phospho_rate=1`` there are ``2**n``.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``mode`` is not one of 'all', 'predict' or 'custom'; if
        ``mode='custom'`` but no ``phosphosites`` were given; if ``phospho_rate``
        is outside 0-1; or if a phosphosite index falls outside the sequence.

    Examples
    --------
    >>> from sparrow import Protein
    >>> phosphoisoforms(Protein("MSAPTEEY"))[:4]
    ['MSAPTEEY', 'MSAPTEEE', 'MSAPEEEY', 'MSAPEEEE']
    >>> len(phosphoisoforms(Protein("MSAPTEEY"), phospho_rate=0.5))   # at most one site
    4
    >>> phosphoisoforms(Protein("MSAPTEEY"), mode="custom", phosphosites=[1])
    ['MSAPTEEY', 'MEAPTEEY']
    """

    # get phosphosite positions
    if mode == "all":
        _phosphosites = _all_phosphosites(protein.sequence)
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


def __getattr__(name: str) -> Any:
    """Resolve a function's old ``get_`` name, with a ``DeprecationWarning``.

    Parameters
    ----------
    name : str
        The attribute asked for.

    Returns
    -------
    Any
        The renamed function.

    Raises
    ------
    AttributeError
        If ``name`` is not one of the renamed functions.
    """
    if name not in _RENAMED_FUNCTIONS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    new_name = _RENAMED_FUNCTIONS[name]
    warn_renamed(f"{__name__}.{name}", f"{__name__}.{new_name}")
    return globals()[new_name]
