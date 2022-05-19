
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

import numpy as np

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
    def dssp(self, recompute=False):
        """
        Predictor that returns per-residue predictions for DSSP scores.


        Value = 0 : helical
        Value = 1 : extended
        Value = 2 : coil

        Predictor trained by Ryan on DSSP scores from AF2

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

        selector = 'dssp'


        if self.__dssp_predictor_object is None:
            from .dssp.dssp_predictor import DSSPPredictor
            self.__dssp_predictor_object = DSSPPredictor()

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__dssp_predictor_object.predict_dssp(self.__protein.sequence)

        return self.__precomputed[selector] 


    # .................................................................
    #
    def dssp_helicity(self, recompute=False):
        """
        Predictor that returns a binary list with 1 or 0 for helicity
        or not, as predicted from a DSSP prediction.

        Value = 0 : non-helical
        Value = 1 : helical

        Predictor trained by Ryan on DSSP scores from AF2

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

        selector = 'dssp'


        if self.__dssp_predictor_object is None:
            from .dssp.dssp_predictor import DSSPPredictor
            self.__dssp_predictor_object = DSSPPredictor()

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__dssp_predictor_object.predict_dssp(self.__protein.sequence)

        return np.array(np.array(self.__precomputed[selector]) == 0, dtype=int).tolist()

    # .................................................................
    #
    def dssp_coil(self, recompute=False):
        """
        Predictor that returns a binary list with 1 or 0 for helicity
        or not, as predicted from a DSSP prediction.

        Value = 0 : non-coil
        Value = 1 : coil

        Predictor trained by Ryan on DSSP scores from AF2

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

        selector = 'dssp'


        if self.__dssp_predictor_object is None:
            from .dssp.dssp_predictor import DSSPPredictor
            self.__dssp_predictor_object = DSSPPredictor()

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__dssp_predictor_object.predict_dssp(self.__protein.sequence)

        return np.array(np.array(self.__precomputed[selector]) == 2, dtype=int).tolist()

    # .................................................................
    #
    def dssp_extended(self, recompute=False):
        """
        Predictor that returns a binary list with 1 or 0 for extended
        or not, as predicted from a DSSP prediction.

        Value = 0 : non-extended
        Value = 1 : extended

        Predictor trained by Ryan on DSSP scores from AF2

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

        selector = 'dssp'


        if self.__dssp_predictor_object is None:
            from .dssp.dssp_predictor import DSSPPredictor
            self.__dssp_predictor_object = DSSPPredictor()

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__dssp_predictor_object.predict_dssp(self.__protein.sequence)

        return np.array(np.array(self.__precomputed[selector]) == 1, dtype=int).tolist()



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

