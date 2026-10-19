## For adding a new predictor, please see the documentation in
#
#
#
# New predictor functions must have the following pattern
#
# 1. Simple all lower_case name separated by underscores that defines the
#    name of the thing being predicted. Do not include 'scores' or 'values' or
#    anything like that. The SPARROW team reserves the right to change the
#    public-facing predictor name if it will be confusing or overlap with existing
#    or planned future features.
#
# 2. It should be callable without any arguments, and use the Predictors.__protein
#    object to obtain sequence information OR other sequence information (e.g.
#    information from other predictors). In this way, a predictor can in fact make
#    use of predictions made by other predictors for hierachical or complex predictions.
#
# 3. Ideally, once a prediction has been computed once it should be saved within the
#    self.__precomputed dictionary using key that matches the function name. The goal
#    here is that if we had some really expensive predictor, this memoization means we
#    only need to compute once. This is not strictly required, but if at all possible
#    we recommend this.
#
# 4. If the predicted value is saved in the self.__precomputed dictionary
#
#

import functools
import importlib
import warnings
from collections.abc import Callable, Hashable
from dataclasses import dataclass

import numpy as np

from sparrow.data.configs import MIN_LENGTH_ALBATROSS_RE_RG
from sparrow.sparrow_exceptions import SparrowException
from sparrow.tools.utilities import copy_of, validate_keyword_option

#: Every network a Predictor runs, by the name batch_predict knows it by, with the
#: module and class of the predictor that loads it. A network's prediction is
#: cached under its name, so sparrow.batch can store a prediction made for a
#: whole batch where the methods below look for it.
NETWORK_PREDICTORS = {
    "rg": ("sparrow.predictors.rg.radius_of_gyration_predictor", "RgPredictor"),
    "scaled_rg": ("sparrow.predictors.scaled_rg.scaled_radius_of_gyration_predictor", "ScaledRgPredictor"),
    "re": ("sparrow.predictors.e2e.end_to_end_distance_predictor", "RePredictor"),
    "scaled_re": ("sparrow.predictors.scaled_re.scaled_end_to_end_distance_predictor", "ScaledRePredictor"),
    "asphericity": ("sparrow.predictors.asphericity.asphericity_predictor", "AsphericityPredictor"),
    "prefactor": ("sparrow.predictors.prefactor.prefactor_predictor", "PrefactorPredictor"),
    "scaling_exponent": ("sparrow.predictors.scaling_exponent.scaling_exponent_predictor", "ScalingExponentPredictor"),
    "dssp": ("sparrow.predictors.dssp.dssp_predictor", "DSSPPredictor"),
    "transmembrane": ("sparrow.predictors.transmembrane.transmembrane_predictor", "TransmembranePredictor"),
    "mitochondrial_targeting": (
        "sparrow.predictors.mitochondrial_targeting.mitochondrial_targeting_predictor",
        "MitochondrialTargetingPredictor",
    ),
    "nes": ("sparrow.predictors.nes.nuclear_export_signal_predictor", "NESPredictor"),
    "nls": ("sparrow.predictors.nls.nuclear_import_signal_predictor", "NLSPredictor"),
    "tad": ("sparrow.predictors.tad.transactivation_domain_predictor", "TADPredictor"),
    "pscore": ("sparrow.predictors.pscore.pscore_predictor", "PScorePredictor"),
    "ser_phosphorylation": (
        "sparrow.predictors.phosphorylation.ser_phosphorylation_predictor",
        "SerPhosphorylationPredictor",
    ),
    "thr_phosphorylation": (
        "sparrow.predictors.phosphorylation.thr_phosphorylation_predictor",
        "ThrPhosphorylationPredictor",
    ),
    "tyr_phosphorylation": (
        "sparrow.predictors.phosphorylation.tyr_phosphorylation_predictor",
        "TyrPhosphorylationPredictor",
    ),
}

#: The Predictor methods that read a network's prediction, each with the network.
#: The radius of gyration and the end-to-end distance choose between this network
#: and its scaled counterpart for each sequence (see Predictor._network).
METHOD_NETWORKS = {
    "radius_of_gyration": "rg",
    "end_to_end_distance": "re",
    "asphericity": "asphericity",
    "prefactor": "prefactor",
    "scaling_exponent": "scaling_exponent",
    "dssp_helicity": "dssp",
    "dssp_extended": "dssp",
    "dssp_coil": "dssp",
    "transmembrane_regions": "transmembrane",
    "mitochondrial_targeting_sequence": "mitochondrial_targeting",
    "nuclear_export_signal": "nes",
    "nuclear_import_signal": "nls",
    "transactivation_domains": "tad",
    "pscore": "pscore",
    "serine_phosphorylation": "ser_phosphorylation",
    "threonine_phosphorylation": "thr_phosphorylation",
    "tyrosine_phosphorylation": "tyr_phosphorylation",
}

#: The Predictor methods that read a metapredict prediction
METAPREDICT_METHODS = ("disorder", "disorder_domains", "binary_disorder", "pLDDT")

#: Every Predictor method sparrow.batch can predict for a whole batch at once
BATCH_METHODS = tuple(METHOD_NETWORKS) + METAPREDICT_METHODS

# The methods that also have a network trained on the value divided by sqrt(N),
# named with this prefix, which they use by default
SCALED_NETWORK_METHODS = ("radius_of_gyration", "end_to_end_distance")
SCALED_NETWORK_PREFIX = "scaled_"

# The column of each DSSP method's class in the DSSP network's probabilities
# (DSSPPredictor.predict_dssp_probabilities): helix, extended, coil
_DSSP_COLUMNS = {"dssp_helicity": 0, "dssp_extended": 1, "dssp_coil": 2}

# The PScore network predicts on a 0-6 scale, which pscore() rescales onto 0-1
_PSCORE_MINIMUM = 0
_PSCORE_MAXIMUM = 6.0

# Where the metapredict predictions are cached: the disorder domains with
# metapredict's own defaults, and the pLDDT. binary_disorder caches the domains
# for each set of its arguments, under this prefix (see _binary_disorder_key).
_DISORDER_KEY = "metapredict-disorder"
_PLDDT_KEY = "metapredict-pLDDT"

# The name Protein.low_complexity_mask had here, before it moved to Protein since
# it is not a prediction. The old name still works, with a DeprecationWarning,
# for one release.
_OLD_LOW_COMPLEXITY_MASK_NAME = "low_complexity_domains"


def network_predictor_class(network):
    """The class of the predictor that loads one of sparrow's networks.

    Parameters
    ----------
    network : str
        One of :data:`NETWORK_PREDICTORS`.

    Returns
    -------
    type
        A subclass of :class:`~sparrow.predictors.base.BaseNetworkPredictor`.
        Its module, which imports torch, is imported now.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``network`` is not one of sparrow's networks.
    """
    if network not in NETWORK_PREDICTORS:
        raise SparrowException(f"{network!r} is not one of sparrow's networks; these are {list(NETWORK_PREDICTORS)}")
    module_name, class_name = NETWORK_PREDICTORS[network]
    return getattr(importlib.import_module(module_name), class_name)


def load_network_predictor(network, version=None, device=None):
    """Load the predictor for one of sparrow's networks.

    Parameters
    ----------
    network : str
        One of :data:`NETWORK_PREDICTORS`.
    version : str or None, optional
        The network version. Default None, the predictor's default.
    device : torch.device, str or None, optional
        The device to load it onto. Default None, the CPU.

    Returns
    -------
    sparrow.predictors.base.BaseNetworkPredictor
        The loaded predictor.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``network`` is not one of sparrow's networks, or its weights file
        does not exist.
    """
    return network_predictor_class(network)(version=version, device=device)


def _binary_disorder_key(disorder_threshold, minimum_IDR_size, minimum_folded_domain, gap_closure, override):
    """Where binary_disorder caches the disorder domains for one set of its arguments.

    The arguments are part of the key: the plain disorder key holds domains
    found with metapredict's defaults, so reading that here would silently
    ignore any other values.

    Returns
    -------
    str
    """
    return f"{_DISORDER_KEY}-{disorder_threshold}-{minimum_IDR_size}-{minimum_folded_domain}-{gap_closure}-{override}"


def _batch_options(batch_size, gpuid, show_progress):
    """The options to pass on to a batch function: only those given.

    Returns
    -------
    dict
    """
    options = {}
    if batch_size is not None:
        options["batch_size"] = batch_size
    if gpuid is not None:
        options["gpuid"] = gpuid
    options["show_progress_bar"] = show_progress
    return options


def _predict_network(network, sequences, *, batch_size=None, gpuid=None, show_progress=False):
    """One of sparrow's networks' prediction for each of many sequences, in batches.

    Returns
    -------
    list
        One prediction per sequence, as the network's predictor makes it.
    """
    from sparrow.predictors.batch_predictor import predict_many

    return predict_many(network, sequences, **_batch_options(batch_size, gpuid, show_progress))


def _predict_disorder_domains(sequences, *, batch_size=None, gpuid=None, show_progress=False, **domain_arguments):
    """metapredict's disorder and disorder domains for each of many sequences, in batches.

    metapredict chooses its own device, so ``gpuid`` is not passed to it.

    Returns
    -------
    list of metapredict DisorderObject
        One per sequence, as ``meta.predict_disorder_domains`` makes it for one.
    """
    from metapredict import meta

    options = {"batch_size": batch_size} if batch_size is not None else {}
    return meta.predict_disorder(
        sequences,
        return_domains=True,
        return_numpy=False,
        show_progress_bar=show_progress,
        **options,
        **domain_arguments,
    )


def _predict_plddt(sequences, *, batch_size=None, gpuid=None, show_progress=False):
    """metapredict's predicted pLDDT for each of many sequences, in batches.

    Returns
    -------
    list of numpy.ndarray
        One per sequence, as ``meta.predict_pLDDT`` makes it for one.
    """
    from metapredict import meta

    options = {"batch_size": batch_size} if batch_size is not None else {}
    predictions = meta.predict_pLDDT(sequences, normalized=True, show_progress_bar=show_progress, **options)
    return [scores for _, scores in predictions]


def _dssp_result(probability, mode, threshold, minimum_length):
    """A DSSP method's result, from the probability of its class at each residue.

    Parameters
    ----------
    probability : numpy.ndarray
        The probability of the class at each residue, as cached.
    mode : {'class', 'probability', 'both'}
        What to return.
    threshold : float
        As for :func:`~sparrow.predictors.dssp.dssp_predictor.classes_from_probability`.
    minimum_length : int
        As for :func:`~sparrow.predictors.dssp.dssp_predictor.classes_from_probability`.

    Returns
    -------
    numpy.ndarray or tuple
        The classes, a copy of the probabilities, or both.
    """
    from sparrow.predictors.dssp.dssp_predictor import classes_from_probability

    if mode == "probability":
        return probability.copy()

    classes = classes_from_probability(probability, threshold, minimum_length)
    if mode == "class":
        return classes
    return (classes, probability.copy())


@dataclass(frozen=True)
class _Source:
    """A prediction a Predictor method reads, for sparrow.batch.

    Attributes
    ----------
    key : Hashable
        Where the prediction is cached.
    predict_many : callable
        Makes it for a list of sequences at once:
        ``predict_many(sequences, batch_size=..., gpuid=..., show_progress=...)``
        returns one prediction per sequence.
    """

    key: Hashable
    predict_many: Callable[..., list]


class Predictor:
    """Programmatic access to sparrow's sequence-based predictors.

    A ``Predictor`` is not created directly; it is reached as the cached
    ``Protein.predictor`` accessor and operates on that protein's sequence::

        from sparrow import Protein
        p = Protein("MEEEKKKKSSSTTTDDD")
        p.predictor.disorder()
        p.predictor.radius_of_gyration()

    Each method loads its underlying network lazily on first use and memoizes
    the result, so repeated calls are cheap and unused predictors incur no cost.
    The available predictions span disorder and secondary structure, polymer
    dimensions (Rg, Re, asphericity, prefactor, scaling exponent), residue-level
    phosphorylation, localization signals (NLS/NES), transactivation domains,
    transmembrane regions, mitochondrial targeting, and phase-separation pscore.

    See the per-method documentation below for inputs, outputs, and options.
    """

    # .................................................................
    #
    def __init__(self, protein_obj):
        """
        Parent object for wrangling the various sequence predictors implemented in
        sparrow.

        This object is accessible under teh Protein.predictor dot operator. Public
        facing functions should be a single name that refers to the predictor type,
        and those functions should return a np.array of length equal to the
        sequence with per-residue predicted values

        """

        # note the predictor object obtains a reference to the Protein object
        self.__protein = protein_obj

        # the predictor of each network this Predictor has run, by the network's
        # name, each loaded on first use
        self.__network_predictors = {}

        # every prediction made, so that none is made twice: a network's under
        # its name (see _prediction), and metapredict's under its key
        self.__precomputed = {}

    def __getattr__(self, name: str) -> Callable[..., list[int]]:
        """Resolve ``low_complexity_domains``, now ``Protein.low_complexity_mask``, with a ``DeprecationWarning``.

        Python calls this only for a name the object does not otherwise have,
        so the old name is not listed among the predictions, by ``dir`` or by
        :func:`sparrow.batch`.

        Parameters
        ----------
        name : str
            The attribute asked for.

        Returns
        -------
        Callable[..., list[int]]
            :meth:`_old_low_complexity_mask`, which takes the old method's
            arguments.

        Raises
        ------
        AttributeError
            If ``name`` is not ``low_complexity_domains``.
        """
        if name != _OLD_LOW_COMPLEXITY_MASK_NAME:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        warnings.warn(
            "Protein.predictor.low_complexity_domains has moved to Protein.low_complexity_mask, since it is not a "
            "prediction: call p.low_complexity_mask(residue_selector=...) instead. The old name still works, but "
            "will be removed in a future release of sparrow.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._old_low_complexity_mask

    def _old_low_complexity_mask(
        self, residue_selector, minimum_length=15, max_interruption=5, fractional_threshold=0.25
    ):
        """``Protein.low_complexity_mask``, called as the old ``Predictor.low_complexity_domains`` was.

        The old method took the chosen residues as its first argument and had
        only the ``'holt'`` mode, so this passes them on by name with that mode.
        Its defaults are the old method's, which are also those of
        :func:`~sparrow.sequence_analysis.sequence_complexity.low_complexity_domains_holt`.

        Parameters
        ----------
        residue_selector : str
            The chosen residues, as one-letter codes, such as ``'Q'`` or ``'ED'``.
        minimum_length : int
            The shortest domain, in residues. Default is 15.
        max_interruption : int
            The longest run of other residues allowed inside a domain. Default
            is 5.
        fractional_threshold : float
            The fraction of a domain's length its chosen residues must exceed.
            Default is 0.25.

        Returns
        -------
        list of int
            1 for each residue inside a low-complexity domain and 0 for one
            outside, as ``Protein.low_complexity_mask`` gives it.
        """
        return self.__protein.low_complexity_mask(
            mode="holt",
            residue_selector=residue_selector,
            minimum_length=minimum_length,
            max_interruption=max_interruption,
            fractional_threshold=fractional_threshold,
        )

    # .................................................................
    #
    def transmembrane_regions(self, recompute=False):
        """
        Predictor that returns per-residue predictions for transmembrane regions.

        Value = 1 : transmembrane region
        Value = 0 : non transmembrane region

        Predictor trained by Alex on transmembrane regions from uniprot

        Parameters
        --------------
        recompute : bool
            Flag which, of set to true, means the predictor re-runs regardless of if
            the prediction has run already

        Returns
        -----------
        list
            A list of len(seq) with per-residue predictions

        """

        return copy_of(self._prediction("transmembrane", recompute))

    # .................................................................
    #
    def dssp_helicity(self, mode="class", threshold=0.5, minimum_helical_length=5):
        """
        Predictor that returns a binary list with 1 or 0 for helicity
        or not, as predicted from a DSSP prediction.

        Value = 0 : non-helical
        Value = 1 : helical

        Predictor trained by Stephen Plassmeyer on DSSP scores from AlphaFold2. The
        network predicts helix, extended and coil at once, and its prediction is
        cached, so the three DSSP methods, with any arguments, run it once.

        Parameters
        --------------
        mode : str
            Selector which defines how helicity is represented. Default is 'class' but
            the other options are 'probability' and 'both'. For more information on this
            see the Return type info.

        threshold :float
            If class is requested, this defines the threshold at which a residue is
            considered to be helical or not. Default =0.5

        minimum_helical_length : int
            If class is requested, this is the short a region can be and be designated
            as a helix. Default = 5.

        Returns
        -----------
        np.ndarray or tuple
            Return data depends on mode selector

            * class : An np.ndarray of length equal to the sequence where each
              element is a 1 or a 0 (1=helical, 0=non-helical).
            * probability : An np.ndarray of length equal to the sequence where
              each element is between 0 and 1 and reports on the probability that
              the residue is in a helix.
            * both : A tuple where the first element is the class np.ndarray and
              the second element is the probability np.ndarray.

        """

        # ensure a valid keyword was passed
        validate_keyword_option(mode, ["class", "probability", "both"], "mode")

        # the network predicts all three DSSP classes at once; this is one column
        probability = self._prediction("dssp")[:, _DSSP_COLUMNS["dssp_helicity"]]
        return _dssp_result(probability, mode, threshold, minimum_helical_length)

    # .................................................................
    #
    def dssp_coil(self, mode="class", threshold=0.5, minimum_coil_length=1):
        """
        Prediction as to whethere a residue will be found in a coil-state or not (i.e.
        not a helix or an extended/beta structure).

        Predictor trained by Stephen Plassmeyer on DSSP scores from AlphaFold2. The
        network predicts helix, extended and coil at once, and its prediction is
        cached, so the three DSSP methods, with any arguments, run it once.

        Parameters
        --------------
        mode : str
            Selector which defines how coil is represented. Default is 'class' but
            the other options are 'probability' and 'both'. For more information on this
            see the Return type info.

        threshold :float
            If class is requested, this defines the threshold at which a residue is
            considered to be coil or not. Default = 0.5

        minimum_coil_length : int
            If class is requested, this is the short a region can be and be designated
            as a coil region. Default = 1.


        Returns
        -----------
        np.ndarray or tuple
            Return data depends on mode selector

            * class : An np.ndarray of length equal to the sequence where each
              element is a 1 or a 0 (1=coil, 0=non-coil).
            * probability : An np.ndarray of length equal to the sequence where
              each element is between 0 and 1 and reports on the probability that
              the residue is in a coil.
            * both : A tuple where the first element is the class np.ndarray and
              the second element is the probability np.ndarray.

        """

        # ensure a valid keyword was passed
        validate_keyword_option(mode, ["class", "probability", "both"], "mode")

        # the network predicts all three DSSP classes at once; this is one column
        probability = self._prediction("dssp")[:, _DSSP_COLUMNS["dssp_coil"]]
        return _dssp_result(probability, mode, threshold, minimum_coil_length)

    # .................................................................
    #
    def dssp_extended(self, mode="class", threshold=0.5, minimum_extended_length=5):
        """
        Prediction as to whethere a residue will be found in an extended/beta-state or not (i.e.
        not a helix or a coil region).

        Predictor trained by Stephen Plassmeyer on DSSP scores from AlphaFold2. The
        network predicts helix, extended and coil at once, and its prediction is
        cached, so the three DSSP methods, with any arguments, run it once.

        Parameters
        --------------
        mode : str
            Selector which defines how extended is represented. Default is 'class' but
            the other options are 'probability' and 'both'. For more information on this
            see the Return type info.

        threshold :float
            If class is requested, this defines the threshold at which a residue is
            considered to be in an extended region or not. Default = 0.5

        minimum_extended_length : int
            If class is requested, this is the short a region can be and be designated
            as an extended region. Default = 5.


        Returns
        -----------
        np.ndarray or tuple
            Return data depends on mode selector

            * class : An np.ndarray of length equal to the sequence where each
              element is a 1 or a 0 (1=extended, 0=non-extended).
            * probability : An np.ndarray of length equal to the sequence where
              each element is between 0 and 1 and reports on the probability that
              the residue is in an extended state.
            * both : A tuple where the first element is the class np.ndarray and
              the second element is the probability np.ndarray.

        """

        # ensure a valid keyword was passed
        validate_keyword_option(mode, ["class", "probability", "both"], "mode")

        # the network predicts all three DSSP classes at once; this is one column
        probability = self._prediction("dssp")[:, _DSSP_COLUMNS["dssp_extended"]]
        return _dssp_result(probability, mode, threshold, minimum_extended_length)

    # .................................................................
    #
    def mitochondrial_targeting_sequence(self, recompute=False):
        """
        Returns per-residue binary classification as to if the sequence
        includes a mitochondrial targeting sequence. Note this can ONLY
        be found in the N-terminal

        Parameters
        --------------
        recompute : bool
            Flag which, of set to true, means the predictor re-runs regardless of if
            the prediction has run already

        Returns
        -------------
        list
            Returns a list with  per-residue binary score for whether
            or not a residue is predicted to be in a mitochondrial targeting
            sequence or not

        """

        return copy_of(self._prediction("mitochondrial_targeting", recompute))

    # .................................................................
    #
    def nuclear_export_signal(self, recompute=False):
        """
        Returns per-residue probability score as to if the sequence
        includes a nuclear export sequence sequence.

        Parameters
        --------------
        recompute : bool
            Flag which, of set to true, means the predictor re-runs regardless of if
            the prediction has run already

        Returns
        -------------
        list
            Returns a list with  per-residue probability scores for whether
            or not a residue is predicted to be in a nuclear export
            signal

        """

        return copy_of(self._prediction("nes", recompute))

    # .................................................................
    #
    def transactivation_domains(self, recompute=False):
        """
        Returns per-residue probability score as to if the sequence
        includes a transactivation domain.

        Parameters
        --------------
        recompute : bool
            Flag which, of set to true, means the predictor re-runs regardless of if
            the prediction has run already

        Returns
        -------------
        list
            Returns a list with per-residue probability scores for whether
            or not a residue is predicted to be in a transactivation domain

        """

        return copy_of(self._prediction("tad", recompute))

    # .................................................................
    #
    def nuclear_import_signal(self, recompute=False):
        """
        Returns per-residue probability score as to if the sequence
        includes a nuclear import sequence sequence.

        Parameters
        --------------
        recompute : bool
            Flag which, of set to true, means the predictor re-runs regardless of if
            the prediction has run already

        Returns
        -------------
        list
            Returns a list with  per-residue probability scores for whether
            or not a residue is predicted to be in a nuclear import
            signal

        """

        return copy_of(self._prediction("nls", recompute))

    # .................................................................
    #
    def disorder(self, recompute=False):
        """
        Returns per-residue disorder prediction using metapredict (V2)

        Parameters
        --------------
        recompute : bool
            Flag which, of set to true, means the predictor re-runs regardless of if
            the prediction has run already

        Returns
        -------------
        list
            Returns a list with  per-residue disorder score as per
            calculated by metapredict.

        """

        selector = _DISORDER_KEY
        # local import
        from metapredict import meta

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = meta.predict_disorder_domains(self.__protein.sequence, return_numpy=False)

        return copy_of(self.__precomputed[selector].disorder)

    # .................................................................
    #
    def disorder_domains(self, recompute=False, return_boundaries=False):
        """
        Returns per-residue binary classification for if a residue is
        predicted to be in an IDR (1) or not 0. Note this uses metapredicts'
        disorder domain prediction which is more sophisticated than just
        taking the binary classification of each

        Parameters
        --------------
        recompute : bool
            Flag which, of set to true, means the predictor re-runs regardless of if
            the prediction has run already

        return_boundaries : bool
            Flag which, if set to True, means this function returns the IDR
            boundaries instead of a per-residue mask. Default is False.

        Returns
        -------------
        list
            By default, a list of len(seq) with 1s or 0s indicating whether each
            residue is in an IDR or not.

            If ``return_boundaries`` is True, a nested list is returned instead,
            where each sub-element gives the start and end indices of one IDR in
            the sequence, e.g. ``[[0, 10], [40, 80]]`` (start 0-indexed, end
            exclusive).

        See Also
        -------------
        binary_disorder : Same per-residue classification, but exposes
            metapredict's IDR-definition parameters.

        """

        selector = _DISORDER_KEY

        # local import
        from metapredict import meta

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = meta.predict_disorder_domains(self.__protein.sequence, return_numpy=False)

        if return_boundaries:
            return copy_of(self.__precomputed[selector].disordered_domain_boundaries)

        # this is where we construct our binary mask - may be a more efficient way to do this, but I think this scales
        # as O(n*m) where n=number of residues in sequence and m is number domains and given m is typically < 10 this
        # basically scales as a for loop over n which is not bad...
        idr_pos = []
        for i in range(0, len(self.__protein.sequence)):
            found = False
            for d in self.__precomputed[selector].disordered_domain_boundaries:
                # if i is within this IDR
                if i >= d[0] and i < d[1]:
                    idr_pos.append(1)
                    found = True
                    break

                # domains are ordered so if the position i is smaller than
                # the end of the current domain the remaining domains are
                # irrelevant
                if i < d[1]:
                    break

            if found is False:
                idr_pos.append(0)

        return idr_pos

    # .................................................................
    #
    def pLDDT(self, recompute=False):
        """
        Returns per-residue predicted pLDDT score - a prediction of what
        AlphaFold2 will predict

        Parameters
        --------------
        recompute : bool
            Flag which, of set to true, means the predictor re-runs regardless of if
            the prediction has run already

        Returns
        -------------
        np.ndarray
            Returns an array of len(seq) with per-residue pLDDT scores. Note
            these are normalized to fall between 0 and 1 rather than 0 and 100.

        """

        selector = _PLDDT_KEY
        # local import
        from metapredict import meta

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = meta.predict_pLDDT(
                self.__protein.sequence,
                normalized=True,
            )

        return copy_of(self.__precomputed[selector])

    # .................................................................
    #
    def binary_disorder(
        self,
        recompute=False,
        disorder_threshold=0.5,
        minimum_IDR_size=12,
        minimum_folded_domain=50,
        gap_closure=10,
        override_folded_domain_minsize=False,
    ):
        """
        Returns binary classification as to if a residue is in an IDR or not
        using metapredict's IDR definition algorithm. The same parameters
        that are fed into metapredicts' predict_disorder_domains() function
        can be provided here.

        Results are memoized per unique combination of the parameters below, so
        calling this function again with different values recomputes correctly
        without needing recompute=True.

        Parameters
        --------------
        recompute : bool
            Flag which, of set to true, means the predictor re-runs regardless of
            if the prediction has run already

        disorder_threshold : float
            Value that defines what 'disordered' is based on the input predictor
            score. The higher the value the more stringent the cutoff.
            Default = 0.5.

        minimum_IDR_size : int
            Defines the smallest possible IDR. This is a hard limit - i.e. we
            CANNOT get IDRs smaller than this.
            Default = 12.

        minimum_folded_domain : int
            Defines where we expect the limit of small folded domains to be. This
            is NOT a hard limit and functions to modulate the removal of large gaps
            (i.e. gaps less than this size are treated less strictly). Note that, in
            addition, gaps < 35 are evaluated with a threshold of
            0.35*disorder_threshold and gaps < 20 are evaluated with a threshold
            of 0.25*disorder_threshold. These two lengthscales were decided based on
            the fact that coiled-coiled regions (which are IDRs in isolation) often
            show up with reduced apparent disorder within IDRs, and but can be as
            short as 20-30 residues. The folded_domain_threshold is used based on
            the idea that it allows a 'shortest reasonable' folded domain to be
            identified. Default = 50.

        gap_closure : int
            Defines the largest gap that would be 'closed'. Gaps here refer to a
            scenario in which you have two groups of disordered residues seprated
            by a 'gap' of un-disordered residues. In general large gap sizes will
            favour larger contigous IDRs. It's worth noting that gap_closure
            becomes  relevant only when minimum_region_size becomes very small
            (i.e. < 5)  because  really gaps emerge when the smoothed disorder
            fit is "noisy", but when smoothed gaps are increasingly rare.
            Default = 10.

        override_folded_domain_minsize : bool
            DEPRECATED and ignored. Current versions of metapredict's
            ``predict_disorder_domains()`` no longer expose this option, so
            passing it has no effect on the returned classification. Setting it
            to anything other than False raises a ``DeprecationWarning``. It will
            be removed in a future release. Default = False.


        Returns
        -------------
            list
            returns a list with per-residue classification, where 1
            means IDR and 0 means not IDR.


        """

        # this option is no longer supported by metapredict's IDR-definition
        # algorithm; warn rather than silently ignoring a value the user set
        if override_folded_domain_minsize:
            warnings.warn(
                "override_folded_domain_minsize is deprecated and has no effect: "
                "metapredict's predict_disorder_domains() no longer supports it. "
                "The argument will be removed in a future release of sparrow.",
                DeprecationWarning,
                stacklevel=2,
            )

        # NB: the IDR-definition parameters are part of the cache key. The plain
        # "metapredict-disorder" key used by disorder()/disorder_domains() holds a
        # result computed with metapredict's defaults, so reusing it here would
        # silently ignore any non-default values passed to this function.
        selector = _binary_disorder_key(
            disorder_threshold, minimum_IDR_size, minimum_folded_domain, gap_closure, override_folded_domain_minsize
        )
        # local import
        from metapredict import meta

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = meta.predict_disorder_domains(
                self.__protein.sequence,
                disorder_threshold=disorder_threshold,
                minimum_IDR_size=minimum_IDR_size,
                minimum_folded_domain=minimum_folded_domain,
                gap_closure=gap_closure,
                return_numpy=False,
            )

        # algorithm below reconstructrs a binary trace of disorder/order based on
        # boundary definitions in, I think, as few operations as possible, if we know
        # that both the FD and IDR boundaries appear in N-to-C order and we know that
        # every residue is either one or other

        # extract out boundaries
        idr_boundaries = self.__precomputed[selector].disordered_domain_boundaries
        fd_boundaries = self.__precomputed[selector].folded_domain_boundaries

        # define if an IDR or an FD comes first

        try:
            # if first position of first IDR is 0 then we start with an IDR
            if idr_boundaries[0][0] == 0:
                first = idr_boundaries
                second = fd_boundaries
                first_val = 1
                second_val = 0

            # if not start with a folded domain
            else:
                first = fd_boundaries
                second = idr_boundaries
                first_val = 0
                second_val = 1

        # or if NO IDRs start with a folded domain
        except IndexError:
            first = fd_boundaries
            second = idr_boundaries
            first_val = 0
            second_val = 1

        # step through boundaries and extend
        binary_scores = []
        for i in range(max(len(fd_boundaries), len(idr_boundaries))):
            try:
                binary_scores.extend((first[i][1] - first[i][0]) * [first_val])
            except IndexError:
                pass

            try:
                binary_scores.extend((second[i][1] - second[i][0]) * [second_val])
            except IndexError:
                pass

        return binary_scores

    # .................................................................
    #
    def pscore(self, recompute=False, normalized=True):
        """
        Predictor that returns per-residue predictions for PScore scores.

        Parameters
        --------------
        recompute : bool
            Flag which, of set to true, means the predictor re-runs regardless of if
            the prediction has run already

        normalized : bool
            Flag which, if set to true (default), means the returned pscore is
            rescaled from its native 0-6 range onto 0-1 and clipped to those
            bounds. Set to False for the raw network output.

        Returns
        -----------
        np.ndarray
            An array of len(seq) with per-residue predictions

        """

        score = self._prediction("pscore", recompute)

        if normalized:
            # from the network's native 0-6 range onto 0-1, clipped to those bounds
            return np.clip((score - _PSCORE_MINIMUM) / (_PSCORE_MAXIMUM - _PSCORE_MINIMUM), 0.0, 1.0)

        return copy_of(score)

    # .................................................................
    #
    def serine_phosphorylation(self, recompute=False, raw_values=False, return_sites_only=False):
        """
        Function for predicting serine phosphorylation from sequence

        Parameters
        --------------
        recompute : bool
            Flag which, of set to true, means the predictor re-runs regardless of if
            the prediction has run already

        raw_values : bool
            By default, this function takes an initial per-residue probility profile and
            breaks that into dicrete residues that are predicted to be phosphorylated.
            If you just want the raw per-residue phosphoprofile then raw_values can be
            set to true, and the function will return an array with values between 0
            and 1 per residue

        return_sites_only : bool
            Flag which, if set to true, means the function returns a list of sites
            (using 0 indexing) instead of a residue mask

        Returns
        ------------
        np.ndarray or list

            By default, returns an np.ndarray mask where 1 = predicted phosphosite
            and 0 is not.

            If raw_values is set to true, instead the array elements contain
            probabilities between 0 and 1

            if retrn_sites_only is set to True then instead a list with index
            positions of phosphosites is provided.

        """

        if raw_values and return_sites_only:
            raise SparrowException(
                "When calling the serine_phosphorylation() predictor you cannot request both raw_values and return_sites_only"
            )

        return self.__phosphorylation("ser_phosphorylation", recompute, raw_values, return_sites_only)

    # .................................................................
    #
    def threonine_phosphorylation(self, recompute=False, raw_values=False, return_sites_only=False):
        """
        Function for predicting threonine phosphorylation from sequence

        Parameters
        --------------
        recompute : bool
            Flag which, of set to true, means the predictor re-runs regardless of if
            the prediction has run already

        raw_values : bool
            By default, this function takes an initial per-residue probility profile and
            breaks that into dicrete residues that are predicted to be phosphorylated.
            If you just want the raw per-residue phosphoprofile then raw_values can be
            set to true, and the function will return an array with values between 0
            and 1 per residue

        return_sites_only : bool
            Flag which, if set to true, means the function returns a list of sites
            (using 0 indexing) instead of a residue mask

        Returns
        ------------
        np.ndarray or list

            By default, returns an np.ndarray mask where 1 = predicted phosphosite
            and 0 is not.

            If raw_values is set to true, instead the array elements contain
            probabilities between 0 and 1

            if retrn_sites_only is set to True then instead a list with index
            positions of phosphosites is provided.

        """

        if raw_values and return_sites_only:
            raise SparrowException(
                "When calling the threonine_phosphorylation() predictor you cannot request both raw_values and return_sites_only"
            )

        return self.__phosphorylation("thr_phosphorylation", recompute, raw_values, return_sites_only)

    # .................................................................
    #
    def tyrosine_phosphorylation(self, recompute=False, raw_values=False, return_sites_only=False):
        """
        Function for predicting tyrosine phosphorylation from sequence

        Parameters
        --------------
        recompute : bool
            Flag which, of set to true, means the predictor re-runs regardless of if
            the prediction has run already

        raw_values : bool
            By default, this function takes an initial per-residue probility profile and
            breaks that into dicrete residues that are predicted to be phosphorylated.
            If you just want the raw per-residue phosphoprofile then raw_values can be
            set to true, and the function will return an array with values between 0
            and 1 per residue

        return_sites_only : bool
            Flag which, if set to true, means the function returns a list of sites
            (using 0 indexing) instead of a residue mask

        Returns
        ------------
        np.ndarray or list

            By default, returns an np.ndarray mask where 1 = predicted phosphosite
            and 0 is not.

            If raw_values is set to true, instead the array elements contain
            probabilities between 0 and 1

            if retrn_sites_only is set to True then instead a list with index
            positions of phosphosites is provided.

        """

        if raw_values and return_sites_only:
            raise SparrowException(
                "When calling the tyrosine_phosphorylation() predictor you cannot request both raw_values and return_sites_only"
            )

        return self.__phosphorylation("tyr_phosphorylation", recompute, raw_values, return_sites_only)

    def radius_of_gyration(self, use_scaled=True, recompute=False, safe=True):
        """
        Returns the predicted radius of gyration of the sequence. Note that this
        prediction can be done using one of two independently-trained networks. If
        use_scaled = True then the predictor uses a network that was trained on
        sequences after normalizing out the length contribution, whereas if
        use_scaled=False then the prediction is done on a network trained using
        rg data directly.

        In principle both networks should give near-identical predictions. However,
        in general we have found the scaled networks are slightly more accurate.

        Note that for short sequences use_scaled performs much more accurately. As
        such, if the sequence is shorter than
        ``sparrow.data.configs.MIN_LENGTH_ALBATROSS_RE_RG`` residues (35 at the
        time of writing), the predictor automatically invokes the scaled network
        even if the non-scaled version was requested. This is the recommended and
        safe behavior; however, if you want to override this, you can do so by
        setting safe=False. We do NOT recommend this, but, it can be done.

        Parameters
        --------------
        use_scaled : bool
            Flag which selects the network used. If True (default) the
            ``scaled_rg`` network - trained on Rg/sqrt(N) - is used and the
            prediction is multiplied back up by sqrt(N). If False the ``rg``
            network, trained directly on Rg, is used. We recommend leaving this
            as True. Default = True.

        recompute : bool
            Flag which, of set to true, means the predictor re-runs regardless of if
            the prediction has run already

        safe : bool
            Flag which, if set to False, means the requested network will be used
            for rg prediction regardless of the sequence length. NOT RECOMMENDED.
            Default = True.

        Returns
        -------------
        float
            Returns the predicted radius of gyration of the sequence, in Angstroms

        """

        # a sequence shorter than MIN_LENGTH_ALBATROSS_RE_RG uses the scaled
        # network unless safe is False; see _network
        network = self._network("radius_of_gyration", use_scaled=use_scaled, safe=safe)
        return copy_of(self._prediction(network, recompute))

    def end_to_end_distance(self, use_scaled=True, recompute=False, safe=True):
        """
        Returns the predicted end-to-end distance of the sequence. Note that this
        prediction can be done using one of two independently-trained networks. If
        use_scaled = True then the predictor uses a network that was trained on
        sequences after normalizing out the length contribution, whereas if
        use_scaled=False then the prediction is done on a network trained using
        end-to-end distance data directly.

        In principle both networks should give near-identical predictions. However,
        in general we have found the scaled networks are slightly more accurate.

        Note that for short sequences scaled networks performs much more accurately. As
        such, if the sequence is shorter than
        ``sparrow.data.configs.MIN_LENGTH_ALBATROSS_RE_RG`` residues (35 at the
        time of writing), the predictor automatically invokes the scaled network
        even if the non-scaled version was requested. This is the recommended and
        safe behavior; however, if you want to override this, you can do so by
        setting safe=False. We do NOT recommend this, but, it can be done.

        Parameters
        --------------
        use_scaled : bool
            Flag which selects the network used. If True (default) the
            ``scaled_re`` network - trained on Re/sqrt(N) - is used and the
            prediction is multiplied back up by sqrt(N). If False the ``re``
            network, trained directly on Re, is used. We recommend leaving this
            as True. Default = True.

        recompute : bool
            Flag which, of set to true, means the predictor re-runs regardless of if
            the prediction has run already

        safe : bool
            Flag which, if set to False, means the requested network will be used
            for re prediction regardless of the sequence length. NOT RECOMMENDED.
            Default = True.

        Returns
        -------------
        float
            Returns the predicted end-to-end distance of the sequence, in Angstroms

        """

        # a sequence shorter than MIN_LENGTH_ALBATROSS_RE_RG uses the scaled
        # network unless safe is False; see _network
        network = self._network("end_to_end_distance", use_scaled=use_scaled, safe=safe)
        return copy_of(self._prediction(network, recompute))

    def asphericity(self, recompute=False):
        """
        Returns the predicted asphericity of the sequence

        Parameters
        --------------
        recompute : bool
            Flag which, of set to true, means the predictor re-runs regardless of if
            the prediction has run already

        Returns
        -------------
        float
            Returns the predicted asphericity of the sequence

        """

        return copy_of(self._prediction(self._network("asphericity"), recompute))

    def prefactor(self, recompute=False):
        """
        Returns the predicted prefactor of the sequence

        Parameters
        --------------
        recompute : bool
            Flag which, of set to true, means the predictor re-runs regardless of if
            the prediction has run already

        Returns
        -------------
        float
            Returns the predicted prefactor of the sequence

        """

        return copy_of(self._prediction(self._network("prefactor"), recompute))

    def scaling_exponent(self, recompute=False):
        """
        Returns the predicted scaling exponent of the sequence

        Parameters
        --------------
        recompute : bool
            Flag which, of set to true, means the predictor re-runs regardless of if
            the prediction has run already

        Returns
        -------------
        float
            Returns the predicted scaling exponent of the sequence

        """

        return copy_of(self._prediction(self._network("scaling_exponent"), recompute))

    # .................................................................
    #
    def _prediction(self, network, recompute=False):
        """The prediction of one of sparrow's networks for this sequence, made on first use.

        Each network's prediction is cached under its name, as its predictor
        makes it (:meth:`~sparrow.predictors.base.BaseNetworkPredictor.predict`),
        and every method that reads the network makes its result from that. So a
        network runs once for this sequence, whatever the methods are asked,
        and sparrow.batch can store a prediction made in a batch (see
        :meth:`_store`). The cached object itself is returned; a method copies
        what it hands back.

        Parameters
        ----------
        network : str
            One of :data:`NETWORK_PREDICTORS`.
        recompute : bool, optional
            Predict again even if a prediction is cached. Default False.

        Returns
        -------
        Any
            The prediction.
        """
        if network in self.__precomputed and not recompute:
            return self.__precomputed[network]

        if network not in self.__network_predictors:
            self.__network_predictors[network] = load_network_predictor(network)
        self.__precomputed[network] = self.__network_predictors[network].predict(self.__protein.sequence)
        return self.__precomputed[network]

    def __phosphorylation(self, network, recompute, raw_values, return_sites_only):
        """A phosphorylation method's result, made from its network's cached probabilities.

        Parameters
        ----------
        network : str
            The phosphorylation network.
        recompute : bool
            Predict again even if a prediction is cached.
        raw_values, return_sites_only : bool
            As for :meth:`serine_phosphorylation`.

        Returns
        -------
        numpy.ndarray or list
            As for :meth:`serine_phosphorylation`.
        """
        from sparrow.predictors.phosphorylation.phospho_predictor_utils import phosphorylation_result

        target_residue = network_predictor_class(network).TARGET_RESIDUE
        return phosphorylation_result(
            self.__protein.sequence,
            self._prediction(network, recompute),
            target_residue,
            raw_values=raw_values,
            return_sites_only=return_sites_only,
        )

    def _network(self, method, use_scaled=True, safe=True):
        """The network a method reads for this sequence, which is also its prediction's cache key.

        sparrow.batch calls it to decide which network to run for each sequence
        of a batch, and :meth:`_store` to put each prediction where the method
        will look.

        Parameters
        ----------
        method : str
            One of the methods in :data:`METHOD_NETWORKS`.
        use_scaled : bool, optional
            As for :meth:`radius_of_gyration` and :meth:`end_to_end_distance`;
            ignored for the other methods. Default True.
        safe : bool, optional
            As for :meth:`radius_of_gyration` and :meth:`end_to_end_distance`;
            ignored for the other methods. Default True.

        Returns
        -------
        str
            The network's name, as
            :func:`~sparrow.predictors.batch_predictor.batch_predict` knows it.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If ``method`` does not read a network.
        """
        if method not in METHOD_NETWORKS:
            raise SparrowException(f"{method!r} does not read a network; these do: {list(METHOD_NETWORKS)}")

        network = METHOD_NETWORKS[method]
        if method not in SCALED_NETWORK_METHODS:
            return network

        # the scaled network is much more accurate for short sequences, so they
        # use it whatever was asked for, unless safe is False
        if safe and len(self.__protein) < MIN_LENGTH_ALBATROSS_RE_RG:
            use_scaled = True

        if use_scaled:
            return SCALED_NETWORK_PREFIX + network
        return network

    def _source(self, method, arguments):
        """The prediction a method reads, for sparrow.batch: where it is cached, and how to make it at once for many.

        Parameters
        ----------
        method : str
            The method's name.
        arguments : dict
            Every argument the method is called with, defaults included.

        Returns
        -------
        _Source or None
            None for a method that reads no prediction.
        """
        if method in METHOD_NETWORKS:
            network_arguments = {name: arguments[name] for name in ("use_scaled", "safe") if name in arguments}
            network = self._network(method, **network_arguments)
            return _Source(network, functools.partial(_predict_network, network))

        if method in ("disorder", "disorder_domains"):
            return _Source(_DISORDER_KEY, _predict_disorder_domains)

        if method == "binary_disorder":
            key = _binary_disorder_key(
                arguments["disorder_threshold"],
                arguments["minimum_IDR_size"],
                arguments["minimum_folded_domain"],
                arguments["gap_closure"],
                arguments["override_folded_domain_minsize"],
            )
            predict_many = functools.partial(
                _predict_disorder_domains,
                disorder_threshold=arguments["disorder_threshold"],
                minimum_IDR_size=arguments["minimum_IDR_size"],
                minimum_folded_domain=arguments["minimum_folded_domain"],
                gap_closure=arguments["gap_closure"],
            )
            return _Source(key, predict_many)

        if method == "pLDDT":
            return _Source(_PLDDT_KEY, _predict_plddt)

        return None

    def _has(self, key):
        """Whether a prediction is cached under a key.

        Parameters
        ----------
        key : Hashable
            A key, as :meth:`_source` gives it.

        Returns
        -------
        bool
        """
        return key in self.__precomputed

    def _store(self, key, prediction):
        """Adopt a prediction made outside this Predictor.

        Used by sparrow.batch, which makes a prediction for a whole batch at
        once - with the batch harness for a network
        (:func:`~sparrow.predictors.batch_predictor.predict_many`), or with
        metapredict's own batch functions - and stores each sequence's here,
        under the key :meth:`_source` gives, where the methods look. A
        prediction made in a batch can differ from one made for the sequence
        alone by about 1e-7 relative for a network, the precision of its 32-bit
        arithmetic, because the batch is summed in a different order, and by
        one in the last rounded place for metapredict's; so a prediction this
        Predictor has already made is kept, and a value already read does not
        change.

        Parameters
        ----------
        key : Hashable
            Where the prediction is cached: a network's name, or a metapredict key.
        prediction : Any
            The prediction, as the network's predictor or metapredict makes it.

        Returns
        -------
        None

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If ``key`` is neither a network nor a metapredict key.
        """
        is_metapredict = key in (_DISORDER_KEY, _PLDDT_KEY) or str(key).startswith(f"{_DISORDER_KEY}-")
        if key not in NETWORK_PREDICTORS and not is_metapredict:
            raise SparrowException(f"{key!r} is neither one of sparrow's networks nor a metapredict prediction")

        self.__precomputed.setdefault(key, prediction)
