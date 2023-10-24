
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

from sparrow.data.configs import MIN_LENGTH_ALBATROSS_RE_RG

import numpy as np
from sparrow.sparrow_exceptions import SparrowException
from sparrow.tools.utilities import validate_keyword_option

class Predictor:


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
        self.__protein =  protein_obj


        # predictors that operate using overaching Predictor objects needs to be initialized 
        # as a hidden variable and set to None here. This is the recommended approach for
        # parrot-based predictors, although we note this is not strictly necessary if the
        # predictor operates using some stateless master function (e.g. metapredict)
        self.__transmembrane_predictor_object = None
        self.__dssp_predictor_object = None
        self.__mitochondrial_targeting_predictor_object  = None
        self.__nes_predictor_object  = None
        self.__nls_predictor_object  = None
        self.__pscore_predictor_object = None
        self.__ser_phos_predictor_object = None
        self.__thr_phos_predictor_object = None
        self.__tyr_phos_predictor_object = None
        self.__tad_predictor_object = None

        # ALBATROSS predictors set here
        self.__rg_predictor_object = None
        self.__re_predictor_object = None

        self.__scaled_rg_predictor_object = None
        self.__scaled_re_predictor_object = None

        self.__prefactor_predictor_object = None
        self.__scaling_exponent_predictor_object = None
        self.__asphericity_predictor_object = None

        # this __precomputed dictionary is where predictions made can be scored so that
        # the Predictor avoids needing to recompute (potentially expensive) predictions
        self.__precomputed = {}
        

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
            An list of len(seq) with per-residue predictions

        """

        selector = 'transmembrane'

        if self.__transmembrane_predictor_object is None:
            from .transmembrane.transmembrane_predictor import TransmembranePredictor
            self.__transmembrane_predictor_object = TransmembranePredictor()

        
        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__transmembrane_predictor_object.predict_transmebrane_regions(self.__protein.sequence)

        
        return self.__precomputed[selector] 

    # .................................................................
    #
    def dssp_helicity(self, mode='class', threshold=0.5, minimum_helical_length=5):
        """
        Predictor that returns a binary list with 1 or 0 for helicity
        or not, as predicted from a DSSP prediction.

        Value = 0 : non-helical
        Value = 1 : helical

        Predictor trained by Stephen Plassmeyer on DSSP scores from AlphaFold2. Note
        that because several options are available for dssp helicity predictions we
        do not offer a recompute option and we always recompute.

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
            
            * class : An np.ndarray of length equal to the sequence where each element
            is a 1 or a 0 (1=helical, 0 non-helocal)

            * probability : An np.ndarray of length equal to the sequence where each 
            element is between 0 and 1 and reports on the probability that the residue
            is in a helix

            * both : A tuple where first element is the class np.ndarray and the second
            element is the probability np.ndarray

        """

        # ensure a valid keyword was passed
        validate_keyword_option(mode, ['class','probability', 'both'], 'mode')
        

        if self.__dssp_predictor_object is None:
            from .dssp.dssp_predictor import DSSPPredictor
            self.__dssp_predictor_object = DSSPPredictor()


        if mode == 'class':
            return self.__dssp_predictor_object.predict_helicity_smart(self.__protein.sequence, threshold=threshold, minlen=minimum_helical_length)
        
        elif mode == 'probability':
            return self.__dssp_predictor_object.predict_helical_probability(self.__protein.sequence)

        elif mode == 'both':
            return self.__dssp_predictor_object.predict_helicity_smart(self.__protein.sequence, threshold=threshold, minlen=minimum_helical_length, return_probability=True)
        

    # .................................................................
    #
    def dssp_coil(self, mode='class', threshold=0.5, minimum_coil_length=1):
        """
        Prediction as to whethere a residue will be found in a coil-state or not (i.e.
        not a helix or an extended/beta structure).

        Predictor trained by Stephen Plassmeyer on DSSP scores from AlphaFold2. Note
        that because several options are available for dssp predictions, we
        do not offer a recompute option and we always recompute.

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
            
            * class : An np.ndarray of length equal to the sequence where each element
            is a 1 or a 0 (1=coil, 0=non-coil)

            * probability : An np.ndarray of length equal to the sequence where each 
            element is between 0 and 1 and reports on the probability that the residue
            is in a coil

            * both : A tuple where first element is the class np.ndarray and the second
            element is the probability np.ndarray

        """

        # ensure a valid keyword was passed
        validate_keyword_option(mode, ['class','probability', 'both'], 'mode')
        

        if self.__dssp_predictor_object is None:
            from .dssp.dssp_predictor import DSSPPredictor
            self.__dssp_predictor_object = DSSPPredictor()

        if mode == 'class':
            return self.__dssp_predictor_object.predict_coil_smart(self.__protein.sequence, threshold=threshold, minlen=minimum_coil_length)
        
        elif mode == 'probability':
            return self.__dssp_predictor_object.predict_coil_probability(self.__protein.sequence)

        elif mode == 'both':
            return self.__dssp_predictor_object.predict_coil_smart(self.__protein.sequence, threshold=threshold, minlen=minimum_coil_length, return_probability=True)
        

    # .................................................................
    #
    def dssp_extended(self, mode='class', threshold=0.5, minimum_extended_length=5):
        """
        Prediction as to whethere a residue will be found in an extended/beta-state or not (i.e.
        not a helix or a coil region).

        Predictor trained by Stephen Plassmeyer on DSSP scores from AlphaFold2. Note
        that because several options are available for dssp predictions, we
        do not offer a recompute option and we always recompute.

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
            
            * class : An np.ndarray of length equal to the sequence where each element
            is a 1 or a 0 (1=extended, 0=non-coil)

            * probability : An np.ndarray of length equal to the sequence where each 
            element is between 0 and 1 and reports on the probability that the residue
            is in an extended state

            * both : A tuple where first element is the class np.ndarray and the second
            element is the probability np.ndarray

        """

        # ensure a valid keyword was passed
        validate_keyword_option(mode, ['class','probability', 'both'], 'mode')
        

        if self.__dssp_predictor_object is None:
            from .dssp.dssp_predictor import DSSPPredictor
            self.__dssp_predictor_object = DSSPPredictor()

        if mode == 'class':
            return self.__dssp_predictor_object.predict_extended_smart(self.__protein.sequence, threshold=threshold, minlen=minimum_extended_length)
        
        elif mode == 'probability':
            return self.__dssp_predictor_object.predict_extended_probability(self.__protein.sequence)

        elif mode == 'both':
            return self.__dssp_predictor_object.predict_extended_smart(self.__protein.sequence, threshold=threshold, minlen=minimum_extended_length, return_probability=True)
        

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

        selector = 'mitochondrial-targeting-sequence'

        
        if self.__mitochondrial_targeting_predictor_object is None:
            from .mitochondrial_targeting.mitochondrial_targeting_predictor import MitochondrialTargetingPredictor
            self.__mitochondrial_targeting_predictor_object = MitochondrialTargetingPredictor()


        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__mitochondrial_targeting_predictor_object.predict_mitochondrial_targeting(self.__protein.sequence)

        return self.__precomputed[selector]


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

        selector = 'nes'

        
        if self.__nes_predictor_object is None:
            from .nes.nuclear_export_signal_predictor import NESPredictor
            self.__nes_predictor_object = NESPredictor()


        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__nes_predictor_object.predict_nuclear_export_signal(self.__protein.sequence)

        return self.__precomputed[selector]


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
            Returns a list with  per-residue probability scores for whether
            or not a residue is predicted to be in a nuclear export 
            signal

        """

        selector = 'tad'

        
        if self.__tad_predictor_object is None:
            from .tad.transactivation_domain_predictor import TADPredictor
            self.__tad_predictor_object = TADPredictor()


        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__tad_predictor_object.predict_transactivation_domains(self.__protein.sequence)

        return self.__precomputed[selector]
    

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

        selector = 'nls'
        
        if self.__nls_predictor_object is None:
            from .nls.nuclear_import_signal_predictor import NLSPredictor
            self.__nls_predictor_object = NLSPredictor()


        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__nls_predictor_object.predict_nuclear_import_signal(self.__protein.sequence)

        return self.__precomputed[selector]
    

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

        selector = 'metapredict-disorder'
        # local import
        from metapredict import meta

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = meta.predict_disorder_domains(self.__protein.sequence, return_numpy=False)

        return self.__precomputed[selector].disorder




    # .................................................................
    #
    def low_complexity_domains(self, residue_selector, minimum_length=15, max_interruption=5, fractional_threshold=0.25):
        """
        Function that returns a binary classifier as to if a residue is in a 
        low-complexity regions. Specifically, low-complexity domains here are
        defined based on their amino acid composition, defined by the residues
        passed in the residue_selector.

        Parameters
        -------------
        
        residue_selector : str
            A string of one or more one-letter amino acid codes used to define
            the type of residues to find in LCD. (str). For example 'Q' or 'ED' for
            aspartic acid and glutamic acid.

        minimum_length : int
            An integer that defines the shortes possible LCD (int). Must be a 
            positive integer greater than 0. Default=15
            
        max_interruption : int
            An integer that defines the longest possible interruption allowed between
            any two residues defined by the residue selector within the LCD. This is 
            related to but independent of the fractional threshold. Default = 5.

        fractional_threshold : float
            A fraction between 0 and 1 that defines the minimum fraction of amino 
            acids  found in the residue selector that can be tolerated in the LCD.
            Default = 0.25.

        return domains  : bool
         

        Return
        -------------
        list 

        """
        
        
        boundaries = self.__protein.low_complexity_domains(mode='holt',
                                                           residue_selector=residue_selector,
                                                           minimum_length=minimum_length,
                                                           max_interruption=max_interruption,
                                                           fractional_threshold=fractional_threshold)
        
        # this is where we construct our binary mask - may be a more efficient way to do this, but I think this scales
        # as O(n*m) where n=number of residues in sequence and m is number domains and given m is typically < 10 this
        # basically scales as a for loop over n which is not bad...
        lcd_pos = []
        for i in range(0,len(self.__protein.sequence)):

            found = False
            for d in boundaries:

                # if i is within this IDR
                if i >= d[1] and i < d[2]:
                    lcd_pos.append(1)
                    found = True
                    break

                # domains are ordered so if the position i is smaller than
                # the end of the current domain the remaining domains are
                # irrelevant
                if i < d[2]:
                    break

            if found is False:
                lcd_pos.append(0)

        return lcd_pos


        




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
            Flag which, if provdied, means this function returns a nested list, where
            each sub-element contains the start and end indices of IDRs in the
            sequence, e.g. [[0,10], [40,80]]. 

        Returns
        -------------
        list
            Returns a list with 1s or 0s as to if a region is in an IDR or not. NOTE
            if you want the actual boundaries these can be obtained in a hacky way 

        """

        selector = 'metapredict-disorder'

        # local import
        from metapredict import meta

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = meta.predict_disorder_domains(self.__protein.sequence, return_numpy=False)

        if return_boundaries:
            return self.__precomputed[selector].disordered_domain_boundaries

        # this is where we construct our binary mask - may be a more efficient way to do this, but I think this scales
        # as O(n*m) where n=number of residues in sequence and m is number domains and given m is typically < 10 this
        # basically scales as a for loop over n which is not bad...
        idr_pos = []
        for i in range(0,len(self.__protein.sequence)):

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
        list
            Returns a list with per-residue pLDDT scores

        """

        selector = 'metapredict-pLDDT'
        # local import
        from metapredict import meta

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = meta.predict_pLDDT(self.__protein.sequence, normalized=True, )

        return self.__precomputed[selector]
    

    # .................................................................
    #
    def binary_disorder(self, 
                        recompute=False,  
                        disorder_threshold=0.5, 
                        minimum_IDR_size=12, 
                        minimum_folded_domain=50,
                        gap_closure=10,
                        override_folded_domain_minsize=False):
    

        """
        Returns binary classification as to if a residue is in an IDR or not 
        using metapredict's IDR definition algorithm. The same parameters
        that are fed into metapredicts' predict_disorder_domains() function
        can be provided here.

        If non-default values are used we recommend selecting recompute=True.

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

   
        Returns
        -------------
            list
            returns a list with per-residue classification, where 1
            means IDR and 0 means not IDR.


        """

        selector = 'metapredict-disorder'
        # local import
        from metapredict import meta

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = meta.predict_disorder_domains(self.__protein.sequence, 
                                                                         disorder_threshold=disorder_threshold,
                                                                         minimum_IDR_size=minimum_IDR_size,
                                                                         minimum_folded_domain=minimum_folded_domain,
                                                                         gap_closure=gap_closure,
                                                                         return_numpy=False)

                                                                         
                                                                         
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
        for i in range(max(len(fd_boundaries),len(idr_boundaries))):
            
            try:
                binary_scores.extend((first[i][1]-first[i][0])*[first_val])
            except IndexError:
                pass

            try:
                binary_scores.extend((second[i][1]-second[i][0])*[second_val])
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
            Flag which, if set to true, means returned pscore falls between 0 and 1

        Returns
        -----------
        list
            An list of len(seq) with per-residue predictions

        """

        if normalized:
            selector = 'pscore-normalized'
        else:
            selector = 'pscore'


        if self.__pscore_predictor_object is None:
            from .pscore.pscore_predictor import PScorePredictor
            self.__pscore_predictor_object = PScorePredictor()

        if selector not in self.__precomputed or recompute is True:
            if normalized:

                
                min_pscore = 0
                max_pscore = 6.0

                score = self.__pscore_predictor_object.predict_pscore(self.__protein.sequence)
                self.__precomputed[selector] = np.clip((score - min_pscore)/(max_pscore - min_pscore), 0.0, 1.0)

            else:
                self.__precomputed[selector] = self.__pscore_predictor_object.predict_pscore(self.__protein.sequence)

        return self.__precomputed[selector] 


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
            raise SparrowException('When calling the serine_phosphorylation() predictor you cannot request both raw_valus and return_sites only')
        
        if raw_values is True:
            selector = 'ser-phosphorylation-raw-vals'
        elif return_sites_only is True:
            selector = 'ser-phosphorylation-sites'
        else:
            selector = 'ser-phosphorylation'
        
        if self.__ser_phos_predictor_object is None:
            from .phosphorylation.ser_phosphorylation_predictor import SerPhosphorylationPredictor
            self.__ser_phos_predictor_object = SerPhosphorylationPredictor()


        if selector not in self.__precomputed or recompute is True:
            if selector == 'ser-phosphorylation':
                self.__precomputed[selector]  = self.__ser_phos_predictor_object.predict_ser_phosphorylation(self.__protein.sequence)
                
            elif selector == 'ser-phosphorylation-sites':
                self.__precomputed[selector]  = self.__ser_phos_predictor_object.predict_ser_phosphorylation(self.__protein.sequence, return_sites_only=True)
                
            elif selector == 'ser-phosphorylation-raw-vals':
                self.__precomputed[selector]  = self.__ser_phos_predictor_object.predict_ser_phosphorylation(self.__protein.sequence, raw_values=True)

        return self.__precomputed[selector] 



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
            raise SparrowException('When calling the serine_phosphorylation() predictor you cannot request both raw_valus and return_sites only')
        
        if raw_values is True:
            selector = 'thr-phosphorylation-raw-vals'
        elif return_sites_only is True:
            selector = 'thr-phosphorylation-sites'
        else:
            selector = 'thr-phosphorylation'
        
        if self.__thr_phos_predictor_object is None:
            from .phosphorylation.thr_phosphorylation_predictor import ThrPhosphorylationPredictor
            self.__thr_phos_predictor_object = ThrPhosphorylationPredictor()

        if selector not in self.__precomputed or recompute is True:
            if selector == 'thr-phosphorylation':
                self.__precomputed[selector]  = self.__thr_phos_predictor_object.predict_thr_phosphorylation(self.__protein.sequence)
                
            elif selector == 'thr-phosphorylation-sites':
                self.__precomputed[selector]  = self.__thr_phos_predictor_object.predict_thr_phosphorylation(self.__protein.sequence, return_sites_only=True)
                
            elif selector == 'thr-phosphorylation-raw-vals':
                self.__precomputed[selector]  = self.__thr_phos_predictor_object.predict_thr_phosphorylation(self.__protein.sequence, raw_values=True)

        return self.__precomputed[selector] 



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
            raise SparrowException('When calling the serine_phosphorylation() predictor you cannot request both raw_valus and return_sites only')
        
        if raw_values is True:
            selector = 'tyr-phosphorylation-raw-vals'
        elif return_sites_only is True:
            selector = 'tyr-phosphorylation-sites'
        else:
            selector = 'tyr-phosphorylation'
        
        if self.__tyr_phos_predictor_object is None:
            from .phosphorylation.tyr_phosphorylation_predictor import TyrPhosphorylationPredictor
            self.__tyr_phos_predictor_object = TyrPhosphorylationPredictor()

        if selector not in self.__precomputed or recompute is True:
            if selector == 'tyr-phosphorylation':
                self.__precomputed[selector]  = self.__tyr_phos_predictor_object.predict_tyr_phosphorylation(self.__protein.sequence)
                
            elif selector == 'tyr-phosphorylation-sites':
                self.__precomputed[selector]  = self.__tyr_phos_predictor_object.predict_tyr_phosphorylation(self.__protein.sequence, return_sites_only=True)
                
            elif selector == 'tyr-phosphorylation-raw-vals':
                self.__precomputed[selector]  = self.__tyr_phos_predictor_object.predict_tyr_phosphorylation(self.__protein.sequence, raw_values=True)

        return self.__precomputed[selector] 

    
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
        such, if the sequence is less than 25 residues in length, the predictor
        automatically invokes the scaled network even if the non-scaled version
        was requested. This is the recommended and save behavior; however, if you
        want to override this, you can do so by setting safe=False. We do NOT
        recommend this, but, it can be done.

        Parameters
        --------------
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
            Returns the predicted radius of gyration of the sequence

        """

        # if this is a short sequence and safe=True
        if safe:
            if len(self.__protein) < MIN_LENGTH_ALBATROSS_RE_RG:
                use_scaled = True
        
        if use_scaled:
            selector = 'scaled_rg'
            if self.__scaled_rg_predictor_object is None or recompute is True:
                from .scaled_rg.scaled_radius_of_gyration_predictor import ScaledRgPredictor
                self.__scaled_rg_predictor_object = ScaledRgPredictor()

            if selector not in self.__precomputed or recompute is True:
                self.__precomputed[selector] = self.__scaled_rg_predictor_object.predict_scaled_rg(self.__protein.sequence) * np.sqrt(len(self.__protein.sequence))

        else:
            selector = 'rg'
            if self.__rg_predictor_object is None or recompute is True:
                from .rg.radius_of_gyration_predictor import RgPredictor    
                self.__rg_predictor_object = RgPredictor()

            if selector not in self.__precomputed or recompute is True:
                self.__precomputed[selector] = self.__rg_predictor_object.predict_rg(self.__protein.sequence)
        
        return self.__precomputed[selector]



    def end_to_end_distance(self, use_scaled=True, recompute=False, safe=True):
        """
        Returns the predicted end-to-end distance of the sequence. Note that this 
        prediction can be done using one of two independently-trained networks. If
        use_scaled = True then the predictor uses a network that was trained on 
        sequences after normalizing out the length contribution, whereas if 
        use_scaled=False then the prediction is done on a network trained using
        end-to-end distance data directly data directly.

        In principle both networks should give near-identical predictions. However,
        in general we have found the scaled networks are slightly more accurate.

        Note that for short sequences scaled networks performs much more accurately. As
        such, if the sequence is less than 25 residues in length, the predictor
        automatically invokes the scaled network even if the non-scaled version
        was requested. This is the recommended and save behavior; however, if you
        want to override this, you can do so by setting safe=False. We do NOT
        recommend this, but, it can be done.

        Parameters
        --------------
        recompute : bool
            Flag which, of set to true, means the predictor re-runs regardless of if
            the prediction has run already

        safe : bool
            Flag which, if set to False, means the requested network will be used 
            for re prediction regardless of the sequence length. NOT RECOMMENDED.
            Default = True.

        Parameters
        --------------
        recompute : bool
            Flag which, of set to true, means the predictor re-runs regardless of if
            the prediction has run already
   
        Returns
        -------------
        float
            Returns the predicted end-to-end distance of the sequence

        """
        
        if safe:
            if len(self.__protein) < MIN_LENGTH_ALBATROSS_RE_RG:
                use_scaled = True
        
        if use_scaled:
            selector = 'scaled_re'
            if self.__scaled_re_predictor_object is None or recompute is True:
                from .scaled_re.scaled_end_to_end_distance_predictor import ScaledRePredictor
                self.__scaled_re_predictor_object = ScaledRePredictor()

            if selector not in self.__precomputed or recompute is True:
                self.__precomputed[selector] = self.__scaled_re_predictor_object.predict_scaled_re(self.__protein.sequence) * np.sqrt(len(self.__protein.sequence))
                
        else:
            selector = 're'
            if self.__re_predictor_object is None or recompute is True:
                from .e2e.end_to_end_distance_predictor import RePredictor    
                self.__re_predictor_object = RePredictor()

            if selector not in self.__precomputed or recompute is True:
                self.__precomputed[selector] = self.__re_predictor_object.predict_re(self.__protein.sequence)
        
        return self.__precomputed[selector]

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

        selector = 'asph'
        
        if self.__asphericity_predictor_object is None:
            from .asphericity.asphericity_predictor import AsphericityPredictor
            
            self.__asphericity_predictor_object = AsphericityPredictor()


        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__asphericity_predictor_object.predict_asphericity(self.__protein.sequence)

        return self.__precomputed[selector]
    
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

        selector = 'prefactor'
        
        if self.__prefactor_predictor_object is None:
            from .prefactor.prefactor_predictor import PrefactorPredictor
            
            self.__prefactor_predictor_object = PrefactorPredictor()


        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__prefactor_predictor_object.predict_prefactor(self.__protein.sequence)

        return self.__precomputed[selector]

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

        selector = 'scaling_exponent'
        
        if self.__scaling_exponent_predictor_object is None:
            from .scaling_exponent.scaling_exponent_predictor import ScalingExponentPredictor
            
            self.__scaling_exponent_predictor_object = ScalingExponentPredictor()


        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__scaling_exponent_predictor_object.predict_scaling_exponent(self.__protein.sequence)

        return self.__precomputed[selector]
