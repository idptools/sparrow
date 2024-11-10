"""
Unit and regression test for the sparrow package.
"""

# Import package, test suite, and other packages as needed
import random
import sys

import numpy as np
import pytest

import sparrow
from sparrow.data.amino_acids import VALID_AMINO_ACIDS
from sparrow.protein import Protein
from sparrow.sequence_analysis.elm import (
    ELM,
    compute_gained_elms,
    compute_lost_elms,
    compute_retained_elms,
)


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

    s_broken = 'MKASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSXYGQSSYSSYXQ'
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

    # V2
    # assert np.mean(P.predictor.disorder()) == 0.8636131147540983
    
    assert np.isclose(np.mean(P.predictor.disorder()), 0.92875415)
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
    for func_site in func_sites:
        assert func_site in ['di Arginine retention/retrieving signal',
                            'CendR Motif Binding to Neuropilin Receptors',
                            'NLS classical Nuclear Localization Signals',
                            'N-degron',
                            'NRD cleavage site',
                            'PCSK cleavage site']
    assert sorted(start) == sorted([1, 6, 11, 16, 4, 9, 14, 0, 5, 10, 15, 0, 5, 10, 15, 1, 11, 0, 16, 1, 6, 11, 16, 0, 3, 13, 4, 14, 1, 9])
    assert sorted(end) == sorted([4, 9, 14, 19, 9, 14, 19, 3, 8, 13, 18, 3, 8, 13, 18, 8, 18, 3, 20, 5, 10, 15, 20, 20, 9, 19, 10, 20, 9, 15])
    assert sorted(elm_sequences) == sorted(['RRA',
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
                            'RKRRAR'])

def test_elm_comparisons():
    wt = sparrow.Protein("MKKK")
    mut = sparrow.Protein("MRKK")

    wt_elms = wt.elms
    mut_elms = mut.elms

    assert wt.elms == {
                        ELM(regex='^M{0,1}[RK][^P].', functional_site_name='N-degron', start=0, end=3, sequence='KKK')
                    }
    assert mut.elms == {
                        ELM(regex='(.RK)|(RR[^KR])', functional_site_name='NRD cleavage site', start=0, end=3, sequence='MRK'),
                        ELM(regex='^M{0,1}[RK][^P].', functional_site_name='N-degron', start=0, end=4, sequence='MRKK')
                    }

    assert wt.elms - mut.elms == set()
    assert wt.elms & mut.elms == {ELM(regex='^M{0,1}[RK][^P].', functional_site_name='N-degron', start=0, end=4, sequence='MKKK')}
    
    assert compute_lost_elms(wt,[2,"K"]) == set()
    assert compute_retained_elms(wt,"p.K1R") == {ELM(regex='^M{0,1}[RK][^P].', functional_site_name='N-degron', start=0, end=3, sequence='RKK')}
    assert compute_gained_elms(wt,"p.K2R") == {ELM(regex='(.RK)|(RR[^KR])', functional_site_name='NRD cleavage site', start=0, end=3, sequence='MRK')}
    
    assert compute_retained_elms(mut,"p.M1K") == {ELM(regex='(.RK)|(RR[^KR])', functional_site_name='NRD cleavage site', start=0, end=3, sequence='MRK'),
                                                 ELM(regex='^M{0,1}[RK][^P].', functional_site_name='N-degron', start=0, end=4, sequence='MRKK')}
    
    assert compute_gained_elms(mut,"p.M1K") == {ELM(regex='KR.', functional_site_name='PCSK cleavage site', start=0, end=3, sequence='KRK'),
                                                 ELM(regex='[KR]R.', functional_site_name='PCSK cleavage site', start=0, end=3, sequence='KRK')}
    assert compute_lost_elms(mut, "p.M1G") == {ELM(regex='^M{0,1}[RK][^P].', functional_site_name='N-degron', start=0, end=4, sequence='MRKK')}


