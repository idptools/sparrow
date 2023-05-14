# Import package, test suite, and other packages as needed
import sparrow
import pytest
import sys
import numpy as np
from sparrow.protein import Protein
from sparrow.data.amino_acids import VALID_AMINO_ACIDS
import random



USE_LOCALCIDER = True


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
        assert np.isclose(das_kappa_vals[p], Protein(das[p]).kappa, atol=0.03)
        
    if USE_LOCALCIDER:
        from localcider.sequenceParameters import SequenceParameters
        nseqs = 100
        max_count = 100
        n_diff_res = 10

        res_set = VALID_AMINO_ACIDS.copy()

        for i in range(nseqs):
            random.shuffle(res_set)
            local_res = res_set[:n_diff_res]
            seq = ''
            for aa in local_res:
                seq = seq + aa*random.randint(1,max_count)

            seq = list(seq)
            random.shuffle(seq)
            seq = "".join(seq)

            P = Protein(seq)

            # skip sequences 
            if P.fraction_negative == 0 or P.fraction_positive == 0:
                continue
            
            SO = SequenceParameters(seq)
            assert np.isclose(P.NCPR, SO.get_NCPR())
            assert np.isclose(P.FCR, SO.get_FCR())

            # note, this will stochastically fial from time to time..
            assert np.isclose(P.kappa, SO.get_kappa(), atol=0.03)


def test_kappa_range():

    for i in range(100):

        Es = 'E'*random.randint(1,60)
        Ks = 'K'*random.randint(1,60)
        Gs = 'G'*random.randint(1,100)

        tmp = Es+Ks+Gs
        if len(tmp) < 7:
            continue

        tmp_list = list(tmp)
        random.shuffle(tmp_list)
        tmp = "".join(tmp_list)
    
        p = Protein(tmp)
        k = p.kappa

        assert k > 0
        assert k < 1
            
