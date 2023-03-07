from .tools import general_tools, track_tools, io
from protfasta import utilities 
from . import sparrow_exceptions
from . import calculate_parameters
from .visualize import sequence_visuals
from sparrow import data
from .sequence_analysis import sequence_complexity
from .sequence_analysis import phospho_isoforms
from .sequence_analysis import physical_properties
from .sequence_analysis import elm

import numpy as np
import torch
from .patterning import kappa, iwd, scd
from .data import amino_acids
from sparrow.predictors import Predictor
from sparrow.polymer import Polymeric


class Protein:

    def __init__(self, s, validate=False):
        """
        Construct for Protein object. Requires only a single sequence as 
        input. Note that construction does not trigger any sequence 
        parameters to be calculated, all of which are caculated as needed.
        
        Parameters
        -------------
        s :  str
            Amino acid sequence

        validate : bool
            Flag that can be set if sequence should be validated to ensure 
            it's a good amino acid sequence. Generally not necessary to 
            sometimes useful. If set, the function constructor will 
            automatically convert non-standard amino acids to standard
            
            amino acids according to the standard rule.

            * ``B -> N``
            * ``U -> C``
            * ``X -> G``
            * ``Z -> Q``
            * ``* -> <empty string>``
            * ``- -> <empty string>``
        
        Returns
        -----------
            Protein object

        """

        # If validation is needed...
        if validate:
            s = s.upper()
            if general_tools.is_valid_protein_sequence(s) is False:
                
                fixed = utilities.convert_to_valid(s)

                if general_tools.is_valid_protein_sequence(fixed) is False:                
                    raise sparrow_exceptions.SparrowException('Invalid amino acid')

                self.__seq = fixed
            else:
                self.__seq = s.upper()
                
        else:
            self.__seq = s.upper()


        # all sequence parameters are initialized as unset class variables
        self.__aa_fracts = None
        self.__FCR = None
        self.__NCPR = None
        self.__hydrophobicity = None
        self.__aro = None
        self.__ali = None
        self.__polar = None
        self.__IDP_check = None
        self.__f_positive = None
        self.__f_negative = None
        self.__complexity = None
        self.__kappa = None
        self.__scd = None
        self.__shd = None
        self.__kappa_x = {}
        self.__linear_profiles = {}
        self.__molecular_weight = None
        self.__predictor_object = None
        self.__polymeric_object = None
        self.__elms = None

        self.__gpu = ["cuda" if torch.cuda.is_available() else "cpu"][0]
        
    # .................................................................
    #
    @property
    def molecular_weight(self):
        """
        Returns the molecular weight of the the protein.

        Returns
        ---------
        float 
            The molecular weight
        
        """

        if self.__molecular_weight is None:
            self.__molecular_weight = physical_properties.calculate_molecular_weight(self.sequence)

        return self.__molecular_weight
        
            
        
    # .................................................................
    #
    @property
    def amino_acid_fractions(self):
        """
        Returns a dictionary where keys are each of the twenty standard 
        amino acids and the values are the fraction of that amino acid 
        in the sequence.
        
        Returns
        ---------
        dict 
            Returns a 20-position dictionary that includes single-letter 
            codes for each amino acid the corresponding fraction of those 
            residues
            
        
        """

        if self.__aa_fracts is None:
            self.__aa_fracts = calculate_parameters.calculate_aa_fractions(self.__seq)

        return self.__aa_fracts
        
    # .................................................................
    #            
    @property
    def FCR(self):
        """
        Returns the fraction of charged residues (FCR) in the sequence. 
        Charged residues are Asp, Glu, Lys and Arg.

        Returns
        --------
        float
            Float between 0 and 1

        """
        if self.__FCR is None:
            self.__FCR = 0
            for i in data.amino_acids.CHARGE:                
                self.__FCR = self.amino_acid_fractions[i] + self.__FCR
        return self.__FCR


    # .................................................................
    #            
    @property
    def fraction_positive(self):
        """
        Returns the fraction of positively charged residues in the sequence. 
        Positive residues are Arg and Lys (not His at physiological pH). 

        Returns
        --------
        float
            Float between 0 and 1

        """
        if self.__f_positive is None:
            self.__f_positive = 0 
            for i in data.amino_acids.POS:
                self.__f_positive = self.__f_positive + self.amino_acid_fractions[i]
            

        return self.__f_positive

    # .................................................................
    #            
    @property
    def fraction_negative(self):
        """
        Returns the fraction of negatively charged residues in the 
        sequence. Negative residues are Asp and Glu.

        Returns
        --------
        float
            Float between 0 and 1
        """
        if self.__f_negative is None:
            self.__f_negative = 0 
            for i in data.amino_acids.NEG:
                self.__f_negative = self.__f_negative + self.amino_acid_fractions[i]
            

        return self.__f_negative


    # .................................................................
    #
    @property
    def NCPR(self):
        """
        Returns the net charge per residue of the sequence. Net 
        charge is defined as (fraction positive) - (fraction negative)

        Returns
        --------
        float
            Float between -1 and +1
        """

        if self.__NCPR is None:
            self.__NCPR = self.fraction_positive - self.fraction_negative

        return self.__NCPR


    # .................................................................
    #
    @property
    def kappa(self):
        """
        Returns the charge segregation parameter kappa for the sequence. 
        If kappa cannot be calculated (e.g. sequence is shorter than
        5 residues or a sequence with no charged residues) returns a -1.

        Note that kappa defaults to flattening kappa values above 1 to
        1, but this can be turned off with calculate_kappa_x() functio

        Returns
        --------
        float
            Float between 0 and 1, or -1
        """

        if self.__kappa is None:
            if len(self.sequence) < 6:
                self.__kappa = -1
            else:

                k5 = kappa.kappa_x(self.sequence, ['R','K'], ['E','D'], 5, 1)
                k6 = kappa.kappa_x(self.sequence, ['R','K'], ['E','D'], 6, 1)
                self.__kappa = (k5+k6)/2

        return self.__kappa

    @property
    def SCD(self):
        """
        Returns the default sequence charge decoration (SCD) parameter 
        as defined by Sawle and Ghosh [1]

        Returns
        --------
        float
            Returns a float that reports on the sequence charge decoration 

        Reference
        --------
        Sawle, L., & Ghosh, K. (2015). A theoretical method to compute sequence 
        dependent configurational properties in charged polymers and proteins. 
        The Journal of Chemical Physics, 143(8), 085101.

        

        """
        if self.__scd is None:
            self.__scd = scd.compute_scd_x(self.sequence, group1=['E','D'], group2=['R','K'])
            
        return self.__scd 


    @property
    def SHD(self):
        """
        Returns the default sequence charge decoration (SCD) parameter 
        as defined by Sawle and Ghosh [1]

        Returns
        --------
        float
            Returns a float that reports on the sequence charge decoration 

        Reference
        --------
        Sawle, L., & Ghosh, K. (2015). A theoretical method to compute sequence 
        dependent configurational properties in charged polymers and proteins. 
        The Journal of Chemical Physics, 143(8), 085101.

        

        """
        if self.__shd is None:
            self.__shd = scd.compute_shd(self.sequence, hydro_dict=False)
            
        return self.__shd 
    
    # .................................................................
    #
    @property
    def fraction_aromatic(self):
        """
        Returns the fraction of aromatic residues in the sequence. Aromatic
        residues are Tyr, Phe, Trp.

        Returns
        --------
        float
            Float between 0 and 1
        """

        if self.__aro is None:
            self.__aro = 0
            for i in data.amino_acids.ARO:                
                self.__aro = self.amino_acid_fractions[i] + self.__aro

        return self.__aro

    # .................................................................
    #
    @property
    def fraction_aliphatic(self):
        """
        Returns the fraction of aliphatic residues in the sequence. 
        Aliphatic residues are Ala, Leu, Ile, Met, Val. 

        Returns
        --------
        float
            Float between 0 and 1
        """

        if self.__ali is None:
            self.__ali = 0
            for i in data.amino_acids.ALI:                
                self.__ali = self.amino_acid_fractions[i] + self.__ali


        return self.__ali


    # .................................................................
    #
    @property
    def fraction_polar(self):
        """
        Returns the fraction of polar residues in the sequence. 
        Aliphatic residues are Gly, Ser, Thr, Gln, Asn, His.

        Returns
        --------
        float
            Float between 0 and 1
        """

        if self.__polar is None:
            self.__polar = 0
            for i in data.amino_acids.POLAR:                
                self.__polar = self.amino_acid_fractions[i] + self.__polar


        return self.__polar

    # .................................................................
    #
    @property
    def fraction_proline(self):
        """
        Returns the fraction of proline residues.

        Returns
        --------
        float
            Float between 0 and 1
        """
        return self.amino_acid_fractions['P']



    # .................................................................
    #
    @property
    def hydrophobicity(self):
        """
        Returns the linear hydrophobicity from sequence 
        using the Kyte-Doolitle scale.

        Returns
        ----------
        list 
            List of values that correspond to per-residue
            hydrophobicity based on a given hydrophobicity scale
        """
        if self.__hydrophobicity is None:
            self.__hydrophobicity = calculate_parameters.calculate_hydrophobicity(self.__seq)

        return self.__hydrophobicity 


    # .................................................................
    #
    @property
    def complexity(self):
        """
        Calculates the Wootton-Federhen complexity of a sequence (also called
        seg complexity, as this the theory used in the classic SEG algorithm.

        Returns
        ----------
        float
            Returns a float that corresponds to the compositional complexity 
            associated with the passed sequence.
        """
        if self.__complexity is None:
            self.__complexity = calculate_parameters.calculate_seg_complexity(self.__seq)

        return self.__complexity
            

    # .................................................................
    #
    def compute_residue_fractions(self, residue_selector):
        """
        residue_selector is a list of one or more residue types 
        which are used to query the sequence        
        
        """

        f = 0.0

        for i in residue_selector:            
            if i in self.amino_acid_fractions:
                f = f + self.amino_acid_fractions[i]

        return f

    # .................................................................
    #
    def compute_kappa_x(self, group1, group2=None, window_size=6, flatten=True):
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
        -------------
        group1 : str
            Must be a string of valid amino acid one letter codes. 
            This defines one set of residues for which patterning 
            is computed against. If a second set is not provided, 
            patterning is done via group1 vs. all other residues.

        group2 : str
            If provided, this defines the SECOND set of residues, 
            such that patterning is done as residues in group1 vs. 
            group2 in the background of everything else.
        
        window_size : int
            The window size used for the computation of kappa, by default 6

        window_size : int
            Size over which local sequence patterning will be 
            calculated. Default = 6.

        flatten : bool
            Flag which, if set to True, means if kappa is above 1 the 
            function will flatten the value to 1. Default = True.

        Returns
        -------------
        float
            Returns a value associated with the generalized kappa. If 
            flatten is True this is guarenteed to be between 0 and 1 
            unless it's -1 (see above). If flatten is set to False its 
            VERY likeli this will be between 0 and 1 and if it's above
            1 the parameter is probably not useful to use.

        """

        for i in group1:
            if i not in amino_acids.VALID_AMINO_ACIDS:
                raise sparrow_exceptions.ProteinException(f'Amino acid {i} (in group 1) is not a standard amino acid')

            # make sure order is always consistent
            group1 = "".join(sorted(group1))


        # now deal with group 2
        if group2 is None:
            group2 = ''
        else:
            for i in group2:
                if i not in amino_acids.VALID_AMINO_ACIDS:
                    raise sparrow_exceptions.ProteinException(f'Amino acid {i} (in group 2) is not a standard amino acid')
                
            # make sure order is always consistent
            group2 = "".join(sorted(group2))

        if flatten:
            kappa_x_name = group1 + "-" + group2 + str(window_size) + 'flat'
        else:
            kappa_x_name = group1 + "-" + group2 + str(window_size)

        # after set up, calculate kappa_x
        if kappa_x_name not in self.__kappa_x:
            
            if flatten:
                self.__kappa_x[kappa_x_name] = kappa.kappa_x(self.sequence, list(group1), list(group2), window_size, 1)
            else:
                self.__kappa_x[kappa_x_name] = kappa.kappa_x(self.sequence, list(group1), list(group2), window_size, 0)
                
        return self.__kappa_x[kappa_x_name]


    # .................................................................
    #
    def compute_iwd(self, target_residues):
        """
        Returns the inverse weighted distance (IWD), a metric for 
        residue clustering

        Parameters
        -------------
        target_residues : list
            Must be a list of valid amino acid one letter codes. 
            Sanity checking is not performed here (maybe add this?). 
            This defines one set of residues for which patterning 
            is computed against. If a second set is not provided, 
            patterning is done via group1 vs. all other residues.

        Returns
        --------
        float
            Float that is positive

        """

        # ensure valid amino acids are used
        for i in target_residues:
            if i not in amino_acids.VALID_AMINO_ACIDS:
                raise sparrow_exceptions.ProteinException(f'Amino acid {i} (in target_residues) is not a standard amino acid')

        return iwd.calculate_average_inverse_distance_from_sequence(self.sequence, target_residues)
    
    # .................................................................
    #
    def compute_SCD_x(self, group1, group2):
        """
        Function that computes the sequence charge decoration (SCD) 
        parameter of Sawle and Ghosh. This is an alternative sequence 
        patterning parameter which we provide here generalized such 
        that it determines the patterning between any two groups of
        residues. 

        Parameters
        --------------

        group1 : str or list
            Collection of amino acids to be used for defining "negatively
            charged" residues.

        group2 : str or list
            Collection of amino acids to be used for defining "positively
            charged" residues.

        Returns
        -----------
        float
            Returns the custom sequence charge decoration.

        See also
        ---------
        sparrow.protein.scd

        Reference
        -------------
        Sawle, L., & Ghosh, K. (2015). A theoretical method to compute 
        sequence dependent configurational properties in charged polymers 
        and proteins. The Journal of Chemical Physics, 143(8), 085101.
        """

        return scd.compute_scd_x(self.sequence, group1=group1, group2=group2)


    # .................................................................
    #
    def compute_SHD_custom(self, hydro_dict):
        """
        Function takes in a sequence and returns Sequence 
        Hydropathy Decoration (SHD), IE. patterning of hydrophobic 
        residues in the sequence. This is computed as define in ref 1

        To define the hydrophobicity values used the user should pass
        a hydrophobicity dictionary (hydro_dict) which maps amino 
        acid residues to hydrophobicity scores.

        Parameters
        --------------

        hydro : dict
            Dictionary that maps amino acid to hydrophobicity score. 
            Note that every amino acid in the sequence must exist in the
            hydro dict or the function raise an exception.

        Returns
        -----------
        float
            Returns the customized sequence hydroph

        Reference
        -------------
        Zheng, W., Dignon, G. L., Brown, M., Kim, Y. C., & Mittal, J. (2020). 
        Hydropathy Patterning Complements Charge Patterning to Describe 
        Conformational Preferences of Disordered Proteins. Journal of 
        Physical Chemistry Letters. https://doi.org/10.1021/acs.jpclett.0c00288

        """

        scd.compute_shd(self.sequence, hydro_dict=hydro_dict)
        
    # .................................................................
    #
    def compute_iwd_charged_weighted(self, charge=['-','+']):
        """
        Returns the weighted inverse weighted distance (IWD) for either Positive 
        or Negative residues in the sequence. This is a metric for residue clustering
        weighted by the NCPR of each target residue.  

        Parameters
        -------------

        charge : ['-','+']
            Pass '-' to quantify the clustering of negitive residues.
            Pass '+' to quantify the clustering of positive residues.

        Returns
        --------
        float
            Float that is positive
        """

        # ensure valid charge is passed 
        if charge not in ['-','+']:
            raise sparrow_exceptions.ProteinException(f'Passed charge {charge} is not a valid option. Pass "-" for negitive residues and "+" for positive residues.')

        # calculate or retrieve mask of NCPR for sequence
        if 'NCPR-8-extend-ends' not in self.__linear_profiles:
            self.__linear_profiles['NCPR-8-extend-ends'] = track_tools.predefined_linear_track(self.__seq, 'NCPR', 8, 'extend-ends', None)

        linear_NCPR = self.__linear_profiles['NCPR-8-extend-ends']

        return iwd.calculate_average_inverse_distance_charge(linear_NCPR, self.sequence, charge)

    # .................................................................
    #
    def compute_bivariate_iwd_charged_weighted(self):
        """
        Returns the a weighted bivariate inverse weighted distance (IWD) for
        between Possitive and Negative residues in the sequence, a metric for 
        residue clustering weighted by the difference in NCPR of the target residues.  

        Returns
        --------
        float
            Float that is positive
        """

        # calculate or retrieve mask of NCPR for sequence
        if 'NCPR-8-extend-ends' not in self.__linear_profiles:
            self.__linear_profiles['NCPR-8-extend-ends'] = track_tools.predefined_linear_track(self.__seq, 'NCPR', 8, 'extend-ends', None)

        linear_NCPR = self.__linear_profiles['NCPR-8-extend-ends']

        return iwd.calculate_average_bivariate_inverse_distance_charge(linear_NCPR, self.sequence)

    ## .................................................................
    ##
    def generate_phosphoisoforms(self, mode='all', phospho_rate=1, phosphosites=None):
        """
        Calls sequence_analysis.phospho_isoforms module to get a list
        of possible phosphoisoforms sequences. See module header for more documentation.

        Phosphosites are replaced with the phosphomimetic 'E', enabling approximate calculation 
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
        return phospho_isoforms.get_phosphoisoforms(self.sequence, mode=mode, phospho_rate=phospho_rate, 
            phosphosites=phosphosites)
        
    # .................................................................
    #            
    def linear_sequence_profile(self, mode, window_size=8, end_mode='extend-ends', smooth=None):
        """
        Function that returns a vectorized representation of local composition/sequence properties, as defined
        by the passed 'mode', which acts as a selector toggle for a large set of pre-defined analyses types.


        Parameters
        -----------
        mode : str
        
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
            Number of residues over which local sequence properties are calculated. A window stepsize of 1
            is always used

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
            Returns a list with values that correspond to the passed mode
        """

        io.validate_keyword_option(mode, ['FCR','NCPR','aromatic','aliphatic','polar','proline','positive','negative','hydrophobicity', 'seg-complexity','kappa'], 'mode')

        if smooth is not None:
            name = '%s-%i-%s-%i' %(mode, window_size, end_mode, smooth)
        else:
            name = '%s-%i-%s' %(mode, window_size, end_mode)


        if name not in self.__linear_profiles:
            self.__linear_profiles[name] = track_tools.predefined_linear_track(self.__seq,  mode, window_size, end_mode, smooth)
        
        return self.__linear_profiles[name]


    # .................................................................
    #            
    def linear_composition_profile(self, composition_list, window_size=8, end_mode='extend-ends', smooth=None):
        """
        Function that returns a vectorized representation of local composition/sequence properties, as defined
        by the set of one or more residues passed in composition_list.

        Parameters
        ------------
        
        composition_list : list
            List where each element should be a valid amino acid

        window_size : int
            Number of residues over which local sequence properties are calculated. A window stepsize of 1
            is always used

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
            Returns a list with values that correspond to the passed mode

        """

        io.validate_keyword_option(end_mode, ['extend-ends', 'zero-ends', ''], 'end_mode')
        
        # we sort the composition list to unify how it is saved for memoization
        try:
            composition_list = list(set(composition_list))
            composition_list.sort()
            
        except AttributeError:
            raise sparrow_exceptions.ProteinException('Unable to sort composition_list (%s) - this should be a list'%(str(composition_list)))

        
            
        name = "-".join(composition_list) + "-window_size=%i"%(window_size) + "-end_mode=%s"%(end_mode) + "smooth=%s"%(smooth)

        if name not in self.__linear_profiles:
            self.__linear_profiles[name] = track_tools.linear_track_composition(self.__seq,  composition_list, window_size=window_size, end_mode=end_mode, smooth=smooth)
        
        return self.__linear_profiles[name]


    # .................................................................
    #            
    def low_complexity_domains(self, mode='holt', **kwargs):
        """
        Function that extracts low complexity domains from a protein sequence. 
        The arguments passed depend on the mode 
        of extract, as defined below. For now, only 'Holt' is allowed. 

        mode : 'holt'
            Function which returns the set of low-complexity tracts as defined by Gutierrez et al [1].
            Specifically, this returns low complexity sequences defined by four key parameters outlined
            below. The 

            Additional keyword arguments:

                **residue_selector**  : a string of one or more one-letter amino acid codes used to define
                                        the type of residues to find in LCD. (str). For example 'Q' or 'ED' for
                                        aspartic acid and glutamic acid.

                **minimum_length**    : an integer that defines the shortes possible LCD (int). Default = 15. Must 
                                        be positive.
   
                **max_interruption**  : an integer that defines the longest possible interruption allowed between
                                        any two residues defined by the residue selector within the LCD. This is 
                                        related to but independent of the fractional threshold.
                                        For Gutierrez et al this was 17 (!). Default  = 5. 

                **fractional_threshold**  : a fraction between 0 and 1 that defines the minimum fraction of amino 
                                            acids  found in the residue selector that can be tolerated in the LCD.
                                            Default = 0.25.


        mode : 'holt-permissive'

            DO NOT USE FOR NOW!
            Function which returns the set of low-complexity tracts in a slightly more permissive manner than 
            was defied by Gutierrez et al.

            Specifically, this returns regions of a sequence where there is a run that contains minimum_length
            residues which conform to a contigous stretch of the residue(s) defined in residue_selector without
            more that max_interruption intervening residues.
                    
            For example, if residue_selector = 'Q', minimum_length = 10 and max_interuption = 2, then
            QQQQQAAQQQQQ and QAQAQAQAQAQAQAQAQAQ would count but QQQQQAAAQQQQQ would not. 

            Unlike 'holt' mode, now AAAQAQAQAQAQAQAQAAAA WOULD count (while there are 8 Qs here, there is a 
            region of 13 residues that sit in a Q-rich region with no more than 2 interruptions.

            Additional keyword arguments:

                **residue_selector**  : a string of one or more one-letter amino acid codes used to define
                                        the type of residues to find in LCD. (str)

                **minimum_length**    : an integer that defines the shortes possible LCD (int). Default = 15

                **max_interruption**  : an integer that defines the longest possible interruption allowed.
                                        Default  = 2

                **fractional_threshold**  : an fraction between 0 and 1 that defines the longest possible interruption 
                                        allowed. Default = 0.25

        

        Returns
        ---------------
        list of LCDs:
            All modes return the same data-structure, a list of lists with zero or more sublists that are defined 
            as:

            [0] - sequence
            [1] - start position in sequence (slice numbering, indexing from 0)
            [2] - end position in sequence (slice numbering, such that seq[start:end] gives you the LCD)
        

        """

        #io.validate_keyword_option(mode, ['holt', 'holt-permissive'], 'mode')
        io.validate_keyword_option(mode, ['holt'], 'mode')

        if mode == 'holt':
            return sequence_complexity.low_complexity_domains_holt(self.sequence, **kwargs)

        if mode == 'holt-permissive':
            return sequence_complexity.low_complexity_domains_holt_permissive(self.sequence, **kwargs)
            
        

    def show_sequence(self,                   
                      blocksize=10, 
                      newline=50, 
                      fontsize=14, 
                      font_family='Courier', 
                      colors={},
                      header=None,
                      bold_positions=[],
                      bold_residues=[],
                      return_raw_string=False):

        """
        Function that generates an HTML colored string that either renders in the browser or returns the 
        html string. Contains various customizable components.

        Parameters
        -------------

        blocksize : int
            Defines how big blocks of residues are. Blocks are equal to blocksize or the newline parameter, whicever is smaller. 
            Default=10. If set to -1 uses length of the sequence.

        newline : int
            Defines how many residues are shown before a newline is printed. Default is 50. If set to -1 uses the length of
            the sequence.

        fontsize : int
            Fontsize used. Default is 14

        font_family : str
            Which font family (from HTML fonts) is used. Using a non-monospace font makes no sense as columns will be 
            unaligned. Default is Courier. 

        colors : dict
            Dictionary that allows overiding of default color scheme. Should be of format key-value as 'residue'-'color' where 
            residue is a residue in the string and color is a valid HTML color (which can be a Hexcode, standard HTML color name). 
            Note that this also lets you define colors for non-standard amino acids should these be useful. Default is an empty 
            dictionary. Note also that the standard amino acid colorings are defined at sparrow.data.amino_acids.AA_COLOR


        header : str
            If provided, this is a string that provides a FASTA-style header (with a leading carrett included). Default None.

        bold_positions : list
            List of positions (indexing from 1 onwards) which will be bolded. Useful for highlighting specific regions. Note that this
            defines individual residues so (for example) to bold residues 10 to 15 would require bold_positions=[10,11,12,13,14,15]. 
            Default is an empty list.

        bold_residues : list
            List of residue types that can be bolded. Useful for highlighting specific residue groups.  Default is an empty list.

        return_raw_string : bool
            If set to true, the function returns the actual raw HTML string, as opposed to an in-notebook rendering. 
            Default is False

        Returns
        ----------
        None or str
            If return_raw_string is set to True then an HTML-compatible string is returned.


        Raises
        -------
        sparrow.sparrow_exceptions.SparrowException
            Raises a sparrow exception if invalid input is provided (within reason).

        """


        r_val = sequence_visuals.show_sequence(self.sequence, 
                                               blocksize=blocksize,
                                               newline=newline,
                                               fontsize=fontsize,
                                               font_family=font_family,
                                               colors=colors,
                                               header=header,
                                               bold_positions=bold_positions,
                                               bold_residues=bold_residues,
                                               return_raw_string=return_raw_string)
        
        if return_raw_string:
            return r_val

        
    @property
    def predictor(self):
        """
        Returns a sparrow.Predictor object which provides programatic access 
        to the various different sequence-based predictors implemented in
        sparrow.

        Note that each predictor performance necessary imports at runtime on
        the first execution for the first protein, minimizing unecessary 
        overhead.

        Currently available predictors are:

            * disorder : predict per-residue disorder
            * dssp : predict per-residue DSSP score (0,1,or 2)
            * nes : nuclear export signal
            * nis : nuclear import signal
            * phosphorylation
            * pscore
            * tad
            * mitochondrial targeting
            * rg : radius of gyration
            * transmembrane_region : predict binary classification of transmembrane region 
        
        """
        if self.__predictor_object is None:
            self.__predictor_object = Predictor(self)
        return self.__predictor_object

    @property
    def polymeric(self):
        """
        Returns a sparrow.Polymeric object which provides programatic access 
        to the various different predicted polymer properties for the sequence.

        Note that, of course, many of these would only be valid if the sequence
        behaved as an intrinsically disordered or unfolded polypeptide. 
        
        """
        if self.__polymeric_object is None:
            self.__polymeric_object = Polymeric(self)
        return self.__polymeric_object
    
    @property
    def elms(self):
        """Returns a list of NamedTuples containing each of the 
        elm annotations for the given sequence.

        Returns
        -------
        List[NamedTuple]
            A list of NamedTuples containing all possible elms in a given sequence.
    
        """
        if self.__elms is None:
            self.__elms = elm.find_all_elms(self.sequence)
        return self.__elms
        
    @property
    def sequence(self):
        return self.__seq

    def __len__(self):
        return len(self.__seq)
        
    def __repr__(self):
        s = self.__seq[0:5]
        if len(s) < 5:
            s = s + "."*(5-len(s))
        return f"Protein|L = {len(self)}]|{s}..."


