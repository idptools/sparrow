"""
Empty init file in case you choose a package besides PyTest such as Nose which may look for such a file
"""

import numpy as np

from sparrow.data.amino_acids import VALID_AMINO_ACIDS


def build_seq(min_count=10,max_count=50):

    # how many residues
    n_res = np.random.randint(4,20)

    s = ''
    for i in range(n_res):
        aa_idx = np.random.randint(0,20)
        s = s + VALID_AMINO_ACIDS[aa_idx]*np.random.randint(min_count, max_count)
        
    s = list(s)
    np.random.shuffle(s)
    s = "".join(s)
    return s
