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
    assert P.fraction_aromatic == 0.16393442622950818
    assert P.fraction_aliphatic == 0.06557377049180328
    assert P.fraction_polar == 0.6721311475409836
    assert P.fraction_proline == 0.04918032786885246
    assert np.mean(P.predictor.disorder()) == 0.8636131147540983
    assert P.hydrophobicity == 3.052459016393442
    assert P.compute_residue_fractions(['P','E','K','R','D']) == 0.09836065573770492
    assert P.is_IDP == True

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


def test_kappa():

    das = [
        'EKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEK',
        'EEEKKKEEEKKKEEEKKKEEEKKKEEEKKKEEEKKKEEEKKKEEEKKKEK',
        'KEKKKEKKEEKKEEKEKEKEKEEKKKEEKEKEKEKKKEEKEKEEKKEEEE',
        'KEKEEKEKKKEEEEKEKKKKEEKEKEKEKEEKKEEKKKKEEKEEKEKEKE',
        'KEKEKKEEKEKKEEEKKEKEKEKKKEEKKKEEKEEKKEEKKKEEKEEEKE',
        'EEEKKEKKEEKEEKKEKKEKEEEKKKEKEEKKEEEKKKEKEEEEKKKKEK',
        'EEEEKKKKEEEEKKKKEEEEKKKKEEEEKKKKEEEEKKKKEEEEKKKKEK',
        'KKKKEEEEKKKKEEEEKKKKEEEEKKKKEEEEKKKKEEEEKKKKEEEEKE',
        'EEKKEEEKEKEKEEEEEKKEKKEKKEKKKEEKEKEKKKEKKKKEKEEEKE',
        'EKKKKKKEEKKKEEEEEKKKEEEKKKEKKEEKEKEEKEKKEKKEEKEEEE',
        'EKEKKKKKEEEKKEKEEEEKEEEEKKKKKEKEEEKEEKKEEKEKKKEEKK',
        'EKKEEEEEEKEKKEEEEKEKEKKEKEEKEKKEKKKEKKEEEKEKKKKEKK',
        'KEKKKEKEKKEKKKEEEKKKEEEKEKKKEEKKEKKEKKEEEEEEEKEEKE',
        'EKKEKEEKEEEEKKKKKEEKEKKEKKKKEKKKKKEEEEEEKEEKEKEKEE',
        'KKEKKEKKKEKKEKKEEEKEKEKKEKKKKEKEKKEEEEEEEEKEEKKEEE',
        'EKEKEEKKKEEKKKKEKKEKEEKKEKEKEKKEEEEEEEEEKEKKEKKKKE',
        'EKEKKKKKKEKEKKKKEKEKKEKKEKEEEKEEKEKEKKEEKKEEEEEEEE',
        'KEEKKEEEEEEEKEEKKKKKEKKKEKKEEEKKKEEKKKEEEEEEKKKKEK',
        'EEEEEKKKKKEEEEEKKKKKEEEEEKKKKKEEEEEKKKKKEEEEEKKKKK',
        'EEKEEEEEEKEEEKEEKKEEEKEKKEKKEKEEKKEKKKKKKKKKKKKEEE',
        'EEEEEEEEEKEKKKKKEKEEKKKKKKEKKEKKKKEKKEEEEEEKEEEKKK',
        'KEEEEKEEKEEKKKKEKEEKEKKKKKKKKKKKKEKKEEEEEEEEKEKEEE',
        'EEEEEKEEEEEEEEEEEKEEKEKKKKKKEKKKKKKKEKEKKKKEKKEEKK',
        'EEEEKEEEEEKEEEEEEEEEEEEKKKEEKKKKKEKKKKKKKEKKKKKKKK',
        'EEEEEEEEEEEKEEEEKEEKEEKEKKKKKKKKKKKKKKKKKKEEKKEEKE',
        'KEEEEEEEKEEKEEEEEEEEEKEEEEKEEKKKKKKKKKKKKKKKKKKKKE',
        'KKEKKKEKKEEEEEEEEEEEEEEEEEEEEKEEKKKKKKKKKKKKKKKEKK',
        'EKKKKKKKKKKKKKKKKKKKKKEEEEEEEEEEEEEEEEEEKKEEEEEKEK',
        'KEEEEKEEEEEEEEEEEEEEEEEEEEEKKKKKKKKKKKKKKKKKKKKKKK',
        'EEEEEEEEEEEEEEEEEEEEEEEEEKKKKKKKKKKKKKKKKKKKKKKKKK']

    das_kappa_vals = [0.000963782329781065,
                      0.006849987601594839,
                      0.02510380091732725,
                      0.023779919834168346,
                      0.014793830994527891,
                      0.030699929748093432,
                      0.055155094748869704,
                      0.055155094748869704,
                      0.06207283537900597,
                      0.09244645817707578,
                      0.08182457866549872,
                      0.08535584477384989,
                      0.09376754013641903,
                      0.12779464725771064,
                      0.13589023055307498,
                      0.14253932524913954,
                      0.17465693111603184,
                      0.16361063576296123,
                      0.2184643791753562,
                      0.2683678441326591,
                      0.2836833506008589,
                      0.3168464032629612,
                      0.35941633427624997,
                      0.45755189798526164,
                      0.5278595348152701,
                      0.5935761144891406,
                      0.6553235220661426,
                      0.7440558474562516,
                      0.8658988417475169,
                      1.0]

    for p in range(len(das)):
        assert das_kappa_vals[p] == Protein(das[p]).kappa
        
