"""
Unit and regression test for the sparrow package.
"""

# Import package, test suite, and other packages as needed
import sparrow
import pytest
import sys
import numpy as np
from sparrow.protein import Protein

def test_sparrow_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "sparrow" in sys.modules


def test_protein_code_coverage():

        
    s = 'MKASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQ'
    # constructor
    P = Protein(s)
    assert len(P) == 61

    P = Protein(s, validate=True)
    assert len(P) == 61

    s_broken = 'MkASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSXYGQSSYSSYXQ'
    P = Protein(s_broken, validate=True)
    assert len(P) == 61
    assert s == P.sequence

    

    assert len(P.amino_acid_fractions) == 20
    assert P.FCR == 0.04918032786885246 
    assert P.fraction_positive == 0.01639344262295082
    assert P.fraction_negative == 0.03278688524590164
    assert P.NCPR == -0.01639344262295082
    assert P.aromatic_fractions == 0.16393442622950818
    assert P.aliphatic_fractions == 0.06557377049180328
    assert P.polar_fractions == 0.6721311475409836
    assert P.proline_fractions == 0.04918032786885246
    assert np.mean(P.disorder) == 0.8121311475409836
    assert P.hydrophobicity == 3.052459016393442
    assert P.compute_residue_fractions(['P','E','K','R','D']) == 0.09836065573770492
    assert P.is_IDP == True

    assert np.mean(P.build_linear_profile('FCR')) == 0.04918032786885246
    assert np.mean(P.build_linear_profile('NCPR')) == -0.02459016393442623
    assert np.mean(P.build_linear_profile('aromatic')) == 0.1680327868852459
    assert np.mean(P.build_linear_profile('aliphatic')) == 0.05737704918032787
    assert np.mean(P.build_linear_profile('polar')) == 0.6762295081967213
    assert np.mean(P.build_linear_profile('proline')) == 0.04918032786885246
    assert np.mean(P.build_linear_profile('positive')) == 0.012295081967213115
    assert np.mean(P.build_linear_profile('negative')) == 0.036885245901639344
    assert np.mean(P.build_linear_profile('hydrophobicity')) == 3.0450819672131146
