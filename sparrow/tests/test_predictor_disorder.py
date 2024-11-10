# Import package, test suite, and other packages as needed
import sparrow
import pytest
import sys
import numpy as np
from sparrow.protein import Protein

def test_protein_code_coverage():

    P = Protein('MKASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQ')


    # V2
    # assert np.isclose(np.mean(P.predictor.disorder()), 0.8636131147540983)
    
    assert np.isclose(np.mean(P.predictor.disorder()), 0.92875415)
