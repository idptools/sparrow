'''
Like the Protein functionality but for batches of proteins.
'''

import numpy as np
from sparrow.batch_calculations.batch_tools import seqs_to_matrix
from sparrow.batch_calculations import batch_properties

class ProteinBatch:
    def __init__(self, s_list, 
                 convert_invalid_aas=False):
        """
        Construct for ProteinBatch object. Requires a list of sequences
        input. Note that construction does not trigger any sequence
        parameters to be calculated, all of which are calculated as needed.

        Parameters
        -------------
        s_list :  list
            list of protein sequence

        convert_invalid_aas : bool
            Flag that can be set to convert invalid aas
            to valid aas. This is done by converting the invalid aas
            to the first valid aa in the list. The invalid aas
            are converted as follows:
            * ``B -> N``
            * ``U -> C``
            * ``X -> G``
            * ``Z -> Q``

        Returns
        -----------
            ProteinBatch object
        """
        # make sure that input is a list. 
        if not isinstance(s_list, list):
            raise TypeError("s_list must be a list of sequences")
        
        # sort the sequences by length
        s_list = sorted(s_list, key=len)

        # make sure that all sequences are strings
        self.seq_matrix_dict = seqs_to_matrix(s_list, tolerance=1, convert_aas=convert_invalid_aas)

        # all sequence parameters are initialized as unset class variables
        self.__aa_fracts = None
        self.__FCR = None
        self.__NCPR = None
        self.__hydrophobicity = None
        self.__aro = None
        self.__ali = None
        self.__polar = None
        self.__f_proline = None
        self.__f_positive = None
        self.__f_negative = None
        self.__complexity = None
        self.__kappa = None
        self.__scd = None
        self.__shd = None
        self.__kappa_x = {}
        self.__molecular_weight = None
        self.__IWD = None
    
    # .................................................................
    #
    @property
    def molecular_weight(self):
        """
        Returns the molecular weight of the protein batch.

        Returns
        ---------
        list of float
            The molecular weight
        """

        if self.__molecular_weight is None:
            vals=[]
            for seq_length in self.seq_matrix_dict:
                seq_matrix, valid_mask, seq_lengths = self.seq_matrix_dict[seq_length]
                vals.extend(batch_properties.batch_molecular_weight(
                    seq_matrix, valid_mask, seq_lengths
                ))
            self.__molecular_weight = vals
        return self.__molecular_weight    
    
    @property
    def amino_acid_fractions(self):
        """
        Returns the amino acid fractions of the protein batch.

        Returns
        ---------
        list of dict
            The amino acid fractions
        """
        if self.__aa_fracts is None:
            vals=[]
            for seq_length in self.seq_matrix_dict:
                seq_matrix, valid_mask, seq_lengths = self.seq_matrix_dict[seq_length]
                vals.extend(batch_properties.batch_aa_fracts(
                seq_matrix, seq_lengths)
            )
            self.__aa_fracts = vals
        return self.__aa_fracts
    
    @property
    def hydrophobicity(self):
        """
        Returns the hydrophobicity of the protein batch.

        Returns
        ---------
        list of float
            The hydrophobicity values
        """
        if self.__hydrophobicity is None:
            vals = []
            for seq_length in self.seq_matrix_dict:
                seq_matrix, valid_mask, seq_lengths = self.seq_matrix_dict[seq_length]
                vals.extend(batch_properties.batch_hydrophobicity(
                    seq_matrix, valid_mask))
            self.__hydrophobicity = vals
        return self.__hydrophobicity
    
    @property
    def FCR(self):
        """
        Returns the FCR (Fraction of Charged Residues) of the protein batch.

        Returns
        ---------
        list of float
            The FCR values
        """
        if self.__FCR is None:
            vals = []
            for seq_length in self.seq_matrix_dict:
                seq_matrix, valid_mask, seq_lengths = self.seq_matrix_dict[seq_length]
                vals.extend(batch_properties.batch_FCR(
                    seq_matrix, valid_mask))
            self.__FCR = vals
        return self.__FCR
    
    @property
    def NCPR(self):
        """
        Returns the NCPR (Net Charge Per Residue) of the protein batch.

        Returns
        ---------
        list of float
            The NCPR values
        """
        if self.__NCPR is None:
            vals = []
            for seq_length in self.seq_matrix_dict:
                seq_matrix, valid_mask, seq_lengths = self.seq_matrix_dict[seq_length]
                vals.extend(batch_properties.batch_NCPR(
                    seq_matrix, valid_mask))
            self.__NCPR = vals
        return self.__NCPR
    
    @property
    def fraction_aromatic(self):
        """
        Returns the aromaticity of the protein batch.

        Returns
        ---------
        list of float
            The aromaticity values
        """
        if self.__aro is None:
            vals = []
            for seq_length in self.seq_matrix_dict:
                seq_matrix, valid_mask, seq_lengths = self.seq_matrix_dict[seq_length]
                vals.extend(batch_properties.batch_frac_aro(
                    seq_matrix, valid_mask))
            self.__aro = vals
        return self.__aro
    
    @property
    def fraction_polar(self):
        """
        Returns the polarity of the protein batch.

        Returns
        ---------
        list of float
            The polarity values
        """
        if self.__polar is None:
            vals = []
            for seq_length in self.seq_matrix_dict:
                seq_matrix, valid_mask, seq_lengths = self.seq_matrix_dict[seq_length]
                vals.extend(batch_properties.batch_frac_polar(
                    seq_matrix, valid_mask))
            self.__polar = vals
        return self.__polar
    
    @property
    def fraction_aliphatic(self):
        """
        Returns the aliphaticity of the protein batch.

        Returns
        ---------
        list of float
            The aliphaticity values
        """
        if self.__ali is None:
            vals = []
            for seq_length in self.seq_matrix_dict:
                seq_matrix, valid_mask, seq_lengths = self.seq_matrix_dict[seq_length]
                vals.extend(batch_properties.batch_frac_ali(
                    seq_matrix, valid_mask))
            self.__ali = vals
        return self.__ali
    
    @property
    def fraction_positive(self):
        """
        Returns the fraction of positive residues in the protein batch.

        Returns
        ---------
        list of float
            The fraction of positive residues
        """
        if self.__f_positive is None:
            vals = []
            for seq_length in self.seq_matrix_dict:
                seq_matrix, valid_mask, seq_lengths = self.seq_matrix_dict[seq_length]
                vals.extend(batch_properties.batch_frac_positive(
                    seq_matrix, valid_mask))
            self.__f_positive = vals
        return self.__f_positive

    @property
    def fraction_negative(self):
        """
        Returns the fraction of negative residues in the protein batch.

        Returns
        ---------
        list of float
            The fraction of negative residues
        """
        if self.__f_negative is None:
            vals = []
            for seq_length in self.seq_matrix_dict:
                seq_matrix, valid_mask, seq_lengths = self.seq_matrix_dict[seq_length]
                vals.extend(batch_properties.batch_frac_negative(
                    seq_matrix, valid_mask))
            self.__f_negative = vals
        return self.__f_negative
    
    @property
    def fraction_proline(self):
        """
        Returns the fraction of proline residues in the protein batch.

        Returns
        ---------
        list of float
            The fraction of proline residues
        """
        if self.__f_proline is None:
            vals = []
            for seq_length in self.seq_matrix_dict:
                seq_matrix, valid_mask, seq_lengths = self.seq_matrix_dict[seq_length]
                vals.extend(batch_properties.batch_frac_proline(
                    seq_matrix, valid_mask))
            self.__f_proline = vals
        return self.__f_proline
    
    @property
    def complexity(self):
        """
        Returns the complexity of the protein batch.

        Returns
        ---------
        list of float
            The complexity values
        """
        if self.__complexity is None:
            vals = []
            for seq_length in self.seq_matrix_dict:
                seq_matrix, valid_mask, seq_lengths = self.seq_matrix_dict[seq_length]
                vals.extend(batch_properties.batch_complexity(
                    seq_matrix, seq_lengths))
            self.__complexity = vals
        return self.__complexity

    @property
    def kappa(self):
        """
        Returns the Kappa values of the protein batch.

        Returns
        ---------
        list of float
            The Kappa values
        """
        if self.__kappa is None:
            vals = []
            for seq_length in self.seq_matrix_dict:
                seq_matrix, valid_mask, seq_lengths = self.seq_matrix_dict[seq_length]
                vals.extend(batch_properties.batch_kappa(
                    seq_matrix, valid_mask, seq_lengths))
            self.__kappa = vals
        return self.__kappa
    
    def kappa_x(self, group1, group2, window_size=6):
        """
        Returns the Kappa values for a specific residue x in the protein batch.

        Parameters
        ----------
        group1 : str
            First group of residues
        group2 : str
            Second group of residues
        window_size : int
            Window size for calculation

        Returns
        ---------
        list of float
            The Kappa values for the specified groups
        """
        key = (group1, group2, window_size)
        if key not in self.__kappa_x:
            vals = []
            for seq_length in self.seq_matrix_dict:
                seq_matrix, valid_mask, seq_lengths = self.seq_matrix_dict[seq_length]
                vals.extend(batch_properties.batch_kappa_x(
                    seq_matrix, group1, group2, 
                    window_size=window_size, valid_mask=valid_mask, 
                    seq_lengths=seq_lengths))
            self.__kappa_x[key] = vals
        return self.__kappa_x[key]
    
    def compute_iwd(self, target_residues, weights=None):
        """
        Returns the IWD (Inter-Residue Distance) values of the protein batch.

        Parameters
        ----------
        target_residues : list
            List of residues for calculation
        weights : list, optional
            Weights for the residues

        Returns
        ---------
        list of float
            The IWD values
        """
        key = (tuple(target_residues), tuple(weights) if weights else None)
        if key not in getattr(self, '_ProteinBatch__IWD_cache', {}):
            if not hasattr(self, '_ProteinBatch__IWD_cache'):
                self.__IWD_cache = {}
            vals = []
            for seq_length in self.seq_matrix_dict:
                seq_matrix, valid_mask, seq_lengths = self.seq_matrix_dict[seq_length]
                vals.extend(batch_properties.batch_iwd(
                    seq_matrix, valid_mask, target_residues, weights=weights))
            self.__IWD_cache[key] = vals
        return self.__IWD_cache[key]

    @property
    def SCD(self):
        """
        Returns the SCD (Sequence Charge Distribution) values of the protein batch.

        Returns
        ---------
        list of float
            The SCD values
        """
        if self.__scd is None:
            vals = []
            for seq_length in self.seq_matrix_dict:
                seq_matrix, valid_mask, seq_lengths = self.seq_matrix_dict[seq_length]
                vals.extend(batch_properties.batch_scd(
                    seq_matrix, valid_mask, seq_lengths))
            self.__scd = vals
        return self.__scd
    
    @property
    def SHD(self):
        """
        Returns the SHD values of the protein batch.

        Returns
        ---------
        list of float
            The SHD values
        """
        if self.__shd is None:
            vals = []
            for seq_length in self.seq_matrix_dict:
                seq_matrix, valid_mask, seq_lengths = self.seq_matrix_dict[seq_length]
                vals.extend(batch_properties.batch_shd(
                    seq_matrix, valid_mask, seq_lengths))
            self.__shd = vals
        return self.__shd
    
    def __repr__(self):
        """
        Returns a string representation of the ProteinBatch object.
        """
        return f"ProteinBatch with {len(self.seq_matrix)} sequences"
    
    def __len__(self):
        """
        Returns the number of sequences in the ProteinBatch.
        """
        return len(self.seq_matrix)
