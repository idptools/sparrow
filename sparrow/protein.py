from .tools import general_tools, track_tools
from protfasta import utilities 
from . import sparrow_exceptions
from . import calculate_parameters
from sparrow import data
import numpy as np

class Protein:

    def __init__(self, s, validate=False):
        """
        Construct for Protein object. Requires only a single sequence as input. Note that construction
        does not trigger any sequence parameters to be calculated, all of which are caculated as needed.
        Parameters
        -------------
        s :  str
            Amino acid sequence

        validate : bool
            Flag that can be set if sequence should be validated to ensure it's a good
            amino acid sequence. Generally not necessary to sometimes useful. If set, the
            function constructor will automatically convert non-standard amino acids to standard
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
                    raise sparrow_exception('Invalid amino acid')

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
        self.__disorder = None
        self.__IDP_check = None
        self.__f_positive = None
        self.__f_negative = None
        self.__linear_profiles = {}
        
            
        
    # .................................................................
    #
    @property
    def amino_acid_fractions(self):
        """
        Returns a dictionary where keys are each of the twenty standard amino
        acids and the values are the fraction of that amino acid in the sequence.

        Returns
        ---------
        dict 
            Returns a 20-position dictionary that includes single-letter codes for
            each amino acid the corresponding fraction of those residues
        
        """

        if self.__aa_fracts is None:
            self.__aa_fracts = calculate_parameters.calculate_aa_fractions(self.__seq)

        return self.__aa_fracts
        
    # .................................................................
    #            
    @property
    def FCR(self):
        """
        Returns the fraction of charged residues (FCR) in the sequence. Charged
        residues are Asp, Glu, Lys and Arg.

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
        Returns the fraction of positively charges residues in the sequence. Positive
        residues are Arg and Lys (not His at physiological pH). 

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
        Returns the fraction of positively charges residues in the sequence. Negative
        residues are Asp and Glu.

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
        Returns the net charge per residue of the sequence. Net charge is 
        defined as (fraction positive) - (fraction negative)

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
    def aromatic_fractions(self):
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
    def aliphatic_fractions(self):
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
    def polar_fractions(self):
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
    def proline_fractions(self):
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
    def disorder(self):
        if self.__disorder is None:
            self.__disorder = calculate_parameters.calculate_disorder(self.__seq)

        return self.__disorder



    # .................................................................
    #
    @property
    def hydrophobicity(self):
        if self.__hydrophobicity is None:
            self.__hydrophobicity = calculate_parameters.calculate_hydrophobicity(self.__seq)

        return self.__hydrophobicity 


    # .................................................................
    #
    @property
    def is_IDP(self):
        if self.__IDP_check is None:
            if np.mean(self.disorder) >= data.configs.DISORDER_THRESHOLD:
                self.__IDP_check = True
            else:
                self.__IDP_check = False

        return self.__IDP_check


    # .................................................................
    #
    def compute_residue_fractions(self, residue_selector):
        """
        residue_selector is a list of one or more residue types which are used to query
        the sequence
        
        """
        f = 0
        for i in residue_selector:
            if i in self.amino_acid_fractions:
                f = f + self.amino_acid_fractions[i]
        return f
            
        


    # .................................................................
    #            
    def build_linear_profile(self, mode, window_size=8, end_mode='extend-ends'):
        """
        Function that returns a vectorized representation of local composition/sequence properties, as defined
        by the passed 'mode', which acts as a selector toggle for a large set of pre-defined analyses types.


        Parameters
        -----------
        mode : str
        
            'FCR'               : Fraction of charged residues
        
            'NCPR'              : Net charge per residue
         
            'aromatic'         : Fraction of aromatic residues

            'aliphatic'        : Fraction of aliphatic residues

            'polar'             : Fraction of polar residues

            'proline'           : Fraction of proline residues

            'positive'          : Fraction of positive residues
 
            'negative'          : Fraction of negative residues

            'hydrophobicity'    : Linear hydrophobicity (Kyte-Doolitle)

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

        Returns
        ----------
        list
            Returns a list with values that correspond to the passed mode
        """

        name = '%s-%i-%s' %(mode, window_size, end_mode)
        if name not in self.__linear_profiles:
            self.__linear_profiles[name] = track_tools.predefined_linear_track(self.__seq,  mode, window_size, end_mode)
        
        return self.__linear_profiles[name]


    # .................................................................
    #            
    @property
    def build_linear_profile_composition(self, composition_list, window_size=8, end_mode='extend-ends'):
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

        Returns
        ----------
        list
            Returns a list with values that correspond to the passed mode

        """
        
        name = "-".join(composition_list)

        if name not in self.__linear_profiles:
            self.__linear_profiles[name] = track_tools.linear_track_composition(self.__seq,  composition_list, window_size, end_mode)
        
        return self.__linear_profiles[name]
        
    @property
    def sequence(self):
        return self.__seq

    def __len__(self):
        return len(self.__seq)
        
    def __repr__(self):
        return "Protein = %i  " % (len(self))


