from .tools import general_tools
from protfasta import utilities 
from . import sparrow_exceptions
from . import calculate_parameters
from .data import configs



class Protein:


    def __init__(self, s, validate=False):

        # If validation is needed...
        if validate:
            if general_tools.is_valid_protein_sequence(s) is False:

                s = s.upper()
                fixed = utilities.convert_to_valid(s)

                if general_tools.is_valid_protein_sequence(fixed) is False:                
                    raise sparrow_exception('Invalid amino acid')

                self.__seq = fixed
                

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
        
            
        
    # .................................................................
    #
    @property
    def amino_acid_fractions(self):
        """
        Returns a dictionary of amino acid fractions
        """
        if self.__aa_fracts is None:
            self.__aa_fracts = calculate_parameters.calculate_aa_fractions(self.__seq)

        return self.__aa_fracts
        
    # .................................................................
    #            
    @property
    def FCR(self):
        if self.__FCR is None:
            self.__FCR = self.amino_acid_fractions['K'] + self.amino_acid_fractions['R'] + self.amino_acid_fractions['E'] + self.amino_acid_fractions['D']

        return self.__FCR

    # .................................................................
    #
    @property
    def NCPR(self):
        if self.__NCPR is None:
            self.__NCPR = (self.amino_acid_fractions['K'] + self.amino_acid_fractions['R']) - (self.amino_acid_fractions['E'] + self.amino_acid_fractions['D'])

        return self.__NCPR

    # .................................................................
    #
    @property
    def aromatic_fractions(self):
        if self.__aro is None:
            self.__aro = 0
            for i in amino_acids.ARO:                
                self.__aro = self.amino_acid_fractions[i] + self.__aro

        return self.__aro

    # .................................................................
    #
    @property
    def aliphatic_fractions(self):
        if self.__ali is None:
            self.__ali = 0
            for i in amino_acids.ALI:                
                self.__ali = self.amino_acid_fractions[i] + self.__ali


        return self.__ali

    @property
    def polar_fractions(self):
        if self.__polar is None:
            self.__polar = 0
            for i in amino_acids.POLAR:                
                self.__polar = self.amino_acid_fractions[i] + self.__polar


        return self.__polar

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
    def is_IDP(self):
        if self.__IDP_check is None:
            if np.mean(self.disorder) >= data.configs.DISORDER_THRESHOLD:
                self.__IDP_check = True
            else:
                self.__IDP_check = False

        return self.__IDP_check

    # .................................................................
    #
    def hydrophobicity(self):
        if self.__hydrophobicity is None:
            self.__hydrophobicity = calculate_parameters.calculate_hydrophobicity(self.__seq)

        return self.__hydrophobicity 


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
            
        
        

    @property
    def sequence(self):
        return self.__seq

    def __len__(self):
        return len(self.__seq)
        
    def __repr__(self):
        return "Protein = %i  " % (len(self))


