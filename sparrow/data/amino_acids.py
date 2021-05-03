##
## Data on individual amino acids
##
##

VALID_AMINO_ACIDS = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']


ARO   = ['Y','W','F']
ALI   = ['A','L','M','I','V']
POLAR = ['Q','N','S','T','H','G']
CHARGE = ['E','D','R','K']
POS = ['R','K']
NEG = ['E','D']

AA_COLOR = {'Y':'#ff9d00',
            'W':'#ff9d00',
            'F':'#ff9d00',
            'A':'#171616',
            'L':'#171616',
            'M':'#171616',
            'I':'#171616',
            'V':'#171616',
            'Q':'#04700d',
            'N':'#04700d',
            'S':'#04700d',
            'T':'#04700d',
            'H':'#04700d',
            'G':'#04700d',
            'E':'#ff0d0d',
            'D':'#ff0d0d',
            'R':'#2900f5',
            'K':'#2900f5',
            'C':'#ffe70d',
            'P':'#cf30b7'}
            

# KYTE-DOOLITTLE SCALES
# References
# A simple method for displaying the hydropathic character of a protein.
# Kyte J, Doolittle RF. J Mol Biol. 1982 May 5;157(1):105-32.
# Why are "natively unfolded" proteins unstructured under physiological conditions?
# Valdimir N. Uversky, Joel R. Gillespie, and Anthony L. Frink
# Protines: Structure, function, and genetics 41:415-427 (2000)
# Main hydrophobicity scale

AA_hydro_KD = {"A": 6.3,
               "R": 0.0,
               "N": 1.0,
               "D": 1.0,
               "C": 7.0,
               "Q": 1.0,
               "E": 1.0,
               "G": 4.1,
               "H": 1.3,
               "I": 9.0,
               "L": 8.3,
               "K": 0.6,
               "M": 6.4,
               "F": 7.3,
               "P": 2.9,
               "S": 3.7,
               "T": 3.8,
               "W": 3.6,
               "Y": 3.2,
               "V": 8.7}

AA_hydro_KD_normalized = {'A': 0.7, 
                          'R': 0.0, 
                          'N': 0.111, 
                          'D': 0.111, 
                          'C': 0.778, 
                          'Q': 0.111, 
                          'E': 0.111, 
                          'G': 0.456, 
                          'H': 0.144, 
                          'I': 1.0, 
                          'L': 0.922, 
                          'K': 0.067, 
                          'M': 0.711, 
                          'F': 0.811, 
                          'P': 0.322, 
                          'S': 0.411, 
                          'T': 0.422, 
                          'W': 0.4, 
                          'Y': 0.356,
                          'V': 0.967}

