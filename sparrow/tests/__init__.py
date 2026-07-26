"""
Empty init file in case you choose a package besides PyTest such as Nose which may look for such a file
"""

import os

import numpy as np

from sparrow.data.amino_acids import VALID_AMINO_ACIDS

# Reference data lives next to the tests. Resolve it from __file__ rather than the
# working directory so `pytest` works from the repository root as well as from
# inside sparrow/tests.
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "test_data")


def reference_data_path(*parts):
    """Return an absolute path to a file inside the tests' test_data directory.

    Parameters
    ----------
    *parts : str
        Path components below ``sparrow/tests/test_data``.

    Returns
    -------
    str
        Absolute path to the requested reference-data file.
    """
    return os.path.join(TEST_DATA_DIR, *parts)


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
