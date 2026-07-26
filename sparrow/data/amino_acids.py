##
## Data on individual amino acids
##
##

VALID_AMINO_ACIDS =      ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
VALID_AMINO_ACIDS_PHYS = ['W','Y','F','H','Q','N','T','S','C','G','A','I','L','V','M','E','D','K','R','P']


ARO   = ['Y','W','F']
ALI   = ['A','L','M','I','V']
POLAR = ['Q','N','S','T','H','G']
CHARGE = ['E','D','R','K']
POS = ['R','K']
NEG = ['E','D']

AA_THREE_TO_ONE = {'ALA':'A',
                   'CYS':'C',
                   'ASP':'D',
                   'GLU':'E',
                   'PHE':'F',
                   'GLY':'G',
                   'HIS':'H',
                   'ILE':'I',
                   'LYS':'K',
                   'LEU':'L',
                   'MET':'M',
                   'ASN':'N',
                   'PRO':'P',
                   'GLN':'Q',
                   'ARG':'R',
                   'SER':'S',
                   'THR':'T',
                   'VAL':'V',
                   'TRP':'W',
                   'TYR':'Y'}

AA_ONE_TO_THREE = {}
for x in AA_THREE_TO_ONE:
    AA_ONE_TO_THREE[AA_THREE_TO_ONE[x]] = x

    
# acetyl groups have 1C-2O, 4H prior to
# peptide bond formation
# 
AA_MOLECULAR_WEIGHT = {'A': 89.1,
                       'C': 121.2,
                       'D': 133.1,
                       'E': 147.1,
                       'F': 165.2,
                       'G': 75.1,
                       'H': 155.2,
                       'I': 131.2,
                       'K': 146.2,
                       'L': 130.2,
                       'M': 149.2,
                       'N': 132.1,
                       'P': 115.1,
                       'Q': 146.2,
                       'R': 174.2,
                       'S': 105.1,
                       'T': 119.1,
                       'V': 117.1,
                       'W': 204.2,
                       'Y': 181.2,
                       '<': 48,
                       '>': 48}


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

# Residues classed as 'disorder promoting'.
# Reference
# Campen A, Williams RM, Brown CJ, Meng J, Uversky VN, Dunker AK.
# TOP-IDP-scale: a new amino acid scale measuring propensity for intrinsic
# disorder. Protein Pept Lett. 2008;15(9):956-63.
# This is the set used by localCIDER's get_fraction_disorder_promoting().
DISORDER_PROMOTING = ['T', 'A', 'G', 'R', 'D', 'H', 'Q', 'K', 'S', 'E', 'P']

# Residues classed as 'chain expanding' - the charged residues plus proline.
# Used by localCIDER's get_fraction_expanding(); at neutral pH the fraction of
# expanding residues is simply FCR + the proline fraction.
EXPANDING = ['E', 'D', 'R', 'K', 'P']

# Polyproline II (PPII) propensity scale.
# Reference
# Elam WA, Schrank TP, Campagnolo AJ, Hilser VJ. Evolutionary conservation of
# the polyproline II conformation surrounding intrinsically disordered
# phosphorylation sites. Protein Sci. 2013;22(4):405-17.
AA_PPII_HILSER = {'A': 0.37,
                  'C': 0.25,
                  'D': 0.30,
                  'E': 0.42,
                  'F': 0.17,
                  'G': 0.13,
                  'H': 0.20,
                  'I': 0.39,
                  'K': 0.56,
                  'L': 0.24,
                  'M': 0.36,
                  'N': 0.27,
                  'P': 1.00,
                  'Q': 0.53,
                  'R': 0.38,
                  'S': 0.24,
                  'T': 0.32,
                  'V': 0.39,
                  'W': 0.25,
                  'Y': 0.25}

# Side-chain pKa values for the titratable residues, as used for pH-dependent
# charge and isoelectric-point calculations. These are the EMBOSS values, and
# match those used by localCIDER.
AA_PKA = {'C': 8.5,
          'Y': 10.1,
          'H': 6.5,
          'E': 4.1,
          'D': 3.9,
          'K': 10.0,
          'R': 12.5}

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

