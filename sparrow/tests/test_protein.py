"""
Unit and regression test for the sparrow package.
"""

# Import package, test suite, and other packages as needed
import sparrow
import pytest
import sys
import numpy as np
from sparrow.protein import Protein
from sparrow.data.amino_acids import VALID_AMINO_ACIDS
import random


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
    assert P.fraction_aromatic == 0.16393442622950818
    assert P.fraction_aliphatic == 0.06557377049180328
    assert P.fraction_polar == 0.6721311475409836
    assert P.fraction_proline == 0.04918032786885246

    assert np.mean(P.predictor.disorder()) == 0.8636131147540983
    assert P.hydrophobicity == 3.052459016393442
    assert P.compute_residue_fractions(['P','E','K','R','D']) == 0.09836065573770492

    assert np.mean(P.linear_sequence_profile('FCR')) == 0.04918032786885246
    assert np.mean(P.linear_sequence_profile('NCPR')) == -0.02459016393442623
    assert np.mean(P.linear_sequence_profile('aromatic')) == 0.1680327868852459
    assert np.mean(P.linear_sequence_profile('aliphatic')) == 0.05737704918032787
    assert np.mean(P.linear_sequence_profile('polar')) == 0.6762295081967213
    assert np.mean(P.linear_sequence_profile('proline')) == 0.04918032786885246
    assert np.mean(P.linear_sequence_profile('positive')) == 0.012295081967213115
    assert np.mean(P.linear_sequence_profile('negative')) == 0.036885245901639344
    assert np.isclose(np.mean(P.linear_sequence_profile('hydrophobicity')),3.0450819672131146)
    assert np.mean(P.linear_composition_profile(['E','K'])) == 0.012295081967213115

    P = Protein("KRRARKRRARKRRARKRRAR")
    elms = P.elms
    func_sites = []
    elm_sequences = []
    start, end = [],[]
    for elm in elms:
        start.append(elm.start)
        end.append(elm.end)
        elm_sequences.append(elm.sequence)
        func_sites.append(elm.functional_site_name)
    func_sites = list(set(func_sites))
    assert func_sites == ['PCSK cleavage site',
                          'CendR Motif Binding to Neuropilin Receptors',
                           'NRD cleavage site',
                           'di Arginine retention/retrieving signal',
                           'N-degron',
                           'NLS classical Nuclear Localization Signals']
    assert start == [1, 6, 11, 16, 4, 9, 14, 0, 5, 10, 15, 0, 5, 10, 15, 1, 11, 0, 16, 1, 6, 11, 16, 0, 3, 13, 4, 14, 1, 9]
    assert end == [4, 9, 14, 19, 9, 14, 19, 3, 8, 13, 18, 3, 8, 13, 18, 8, 18, 3, 20, 5, 10, 15, 20, 20, 9, 19, 10, 20, 9, 15]
    assert elm_sequences == ['RRA',
                            'RRA',
                            'RRA',
                            'RRA',
                            'RKRRA',
                            'RKRRA',
                            'RKRRA',
                            'KRR',
                            'KRR',
                            'KRR',
                            'KRR',
                            'KRR',
                            'KRR',
                            'KRR',
                            'KRR',
                            'RRARKRR',
                            'RRARKRR',
                            'KRR',
                            'RRAR',
                            'RRAR',
                            'RRAR',
                            'RRAR',
                            'RRAR',
                            'KRRARKRRARKRRARKRRAR',
                            'ARKRRA',
                            'ARKRRA',
                            'RKRRAR',
                            'RKRRAR',
                            'RRARKRRA',
                            'RKRRAR']

        
        
