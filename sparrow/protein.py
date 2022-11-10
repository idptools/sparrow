from .tools import general_tools, track_tools, io
from protfasta import utilities 
from . import sparrow_exceptions
from . import calculate_parameters
from .visualize import sequence_visuals
from sparrow import data
from .sequence_analysis import sequence_complexity
from .sequence_analysis import physical_properties
import numpy as np
from .patterning import kappa, iwd
from .data import amino_acids
from sparrow.predictors import Predictor

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
        self.__kappa_x = {}
        self.__linear_profiles = {}
        self.__molecular_weight = None
        self.__predictor_object = None
        
        
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
        If kappa cannot be calculated (i.e. sequence lacks both positively
        and negatively charged residues, or sequence length is less than 6, 
        then this function returns -1)

        Returns
        --------
        float
            Float between 0 and 1
        """

        if self.__kappa is None:
            if self.fraction_positive == 0:
                self.__kappa = -1
            elif self.fraction_negative == 0:
                self.__kappa = -1
            elif len(self.sequence) < 6:
                self.__kappa = -1
            else:

                k5 = kappa.kappa_x(self.sequence, ['R','K'], ['E','D'], 5)
                k6 = kappa.kappa_x(self.sequence, ['R','K'], ['E','D'], 6)
                self.__kappa = (k5+k6)/2

        return self.__kappa

    @property
    def SCD(self):

        return self.compute_SCD_x()

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
    def compute_kappa_x(self, group1, group2=None, window_size=6):
        """
        Returns kappa for an arbitrary set of residues with an arbitrary window size.

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

        """

        for i in group1:
            if i not in amino_acids.VALID_AMINO_ACIDS:
                raise sparrow_exceptions.ProteinException(f'Amino acid {i} (in group 1) is not a standard amino acid')

            # make sure order is always consistent
            group1 = "".join(sorted(group1))
                
        if group2 is None:
            kappa_x_name = group1 + "-" + str(window_size)
        else:
            for i in group2:
                if i not in amino_acids.VALID_AMINO_ACIDS:
                    raise sparrow_exceptions.ProteinException(f'Amino acid {i} (in group 2) is not a standard amino acid')
                
            # make sure order is always consistent
            group2 = "".join(sorted(group2))
                
            kappa_x_name = group1 + "-" + group2 + str(window_size)

        # after set up, calculate kappa_x
        if kappa_x_name not in self.__kappa_x:
            if group2 is None:
                group2=[]
                for i in amino_acids.VALID_AMINO_ACIDS:
                    if i in group1:
                        pass
                    else:
                        group2.append(i)

            
            self.__kappa_x[kappa_x_name] = kappa.kappa_x(self.sequence, list(group1), list(group2), window_size)

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



    # .................................................................
    #
    def compute_SCD_x(self, group1=['E','D'], group2=['R','K']):
        
        total = 0
        
        for m in range(1, len(self)):
            
            m_val = m + 1
            
            for n in range(0, m-1):
                
                n_val = n+1

                cur_m_res = self.sequence[m]
                cur_n_res = self.sequence[n]
                
                if cur_m_res in group1:
                    cur_m_charge = -1
                    
                elif cur_m_res in group2:
                    cur_m_charge = 1
                    
                else:
                    cur_m_charge = 0

                if cur_n_res in group1:
                    cur_n_charge = -1
                    
                elif cur_n_res in group2:
                    cur_n_charge = 1
                    
                else:
                    cur_n_charge = 0

                charge_val = cur_m_charge * cur_n_charge
                
                final_val = charge_val * (np.sqrt((m_val)-(n_val)))

                total = total + final_val

        return round(total * (1/len(self)), 5)


        
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
            * transmembrane_region : predict binary classification of transmembrane region 
        
        """
        if self.__predictor_object is None:
            self.__predictor_object = Predictor(self)
        return self.__predictor_object

        


        
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


