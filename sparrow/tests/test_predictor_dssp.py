# Import package, test suite, and other packages as needed
import sparrow
import protfasta
import pytest
import sys
import numpy as np
from sparrow.protein import Protein
import pickle

# test both predictor versions and ensure v2 is default
from sparrow.predictors.dssp.dssp_predictor import DSSPPredictor


## ABOUT:
## Updated in July 2023 based on Stephen's enhanced helicity predictor. The test data used
## here was generate by the notebook in generate_test_data/generate_dssp_data.ipynb
##



def test_dssp_helicity_simple():
    """
    Initial sanity check just to make sure things work at a basic level...
    """
    
    
    s = 'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD'


    X1 = DSSPPredictor(version=1)
    X2 = DSSPPredictor(version=2)

    valid_v2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])


    valid_v1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
                         0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                         1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # test defaults

    # check v2 works correctly at the predictor level
    assert np.array_equal(X2.predict_helicity_smart(seq=s), valid_v2)

    # check v1 works correctly at the predictor level
    assert np.array_equal(X1.predict_helicity_smart(seq=s), valid_v1)

    # check default classifier from protein is working
    assert np.array_equal(Protein(s).predictor.dssp_helicity(), valid_v2)



##
## The six tests below systematically test the predictors against a set of 100 natural
## sequences using the defaults. Important to ensure defaults don't silently change!!!
##
    
def test_dssp_helicity_class_test100():

    # read seqs
    seqs = protfasta.read_fasta('test_data/test_seqs_100.fasta')

    # read precomputed data
    with open('test_data/helicity_class_v2_default_test_seqs_100.pickle', 'rb') as f:
        test_data = pickle.load(f)


    for k in seqs:
        P = Protein(seqs[k])
        assert np.array_equal(P.predictor.dssp_helicity(), test_data[k])


        
    
def test_dssp_helicity_prob_test100():

    # read seqs
    seqs = protfasta.read_fasta('test_data/test_seqs_100.fasta')

    # read precomputed data
    with open('test_data/helicity_prob_v2_default_test_seqs_100.pickle', 'rb') as f:
        test_data = pickle.load(f)


    for k in seqs:
        P = Protein(seqs[k])
        assert np.array_equal(P.predictor.dssp_helicity(mode='probability'), test_data[k])


def test_dssp_extended_class_test100():

    # read seqs
    seqs = protfasta.read_fasta('test_data/test_seqs_100.fasta')

    # read precomputed data
    with open('test_data/extended_class_v2_default_test_seqs_100.pickle', 'rb') as f:
        test_data = pickle.load(f)


    for k in seqs:
        P = Protein(seqs[k])
        assert np.array_equal(P.predictor.dssp_extended(), test_data[k])

    
def test_dssp_extended_prob_test100():

    # read seqs
    seqs = protfasta.read_fasta('test_data/test_seqs_100.fasta')

    # read precomputed data
    with open('test_data/extended_prob_v2_default_test_seqs_100.pickle', 'rb') as f:
        test_data = pickle.load(f)


    for k in seqs:
        P = Protein(seqs[k])
        assert np.array_equal(P.predictor.dssp_extended(mode='probability'), test_data[k])


        
def test_dssp_coil_class_test100():

    # read seqs
    seqs = protfasta.read_fasta('test_data/test_seqs_100.fasta')

    # read precomputed data
    with open('test_data/coil_class_v2_default_test_seqs_100.pickle', 'rb') as f:
        test_data = pickle.load(f)


    for k in seqs:
        P = Protein(seqs[k])
        assert np.array_equal(P.predictor.dssp_coil(), test_data[k])

    
def test_dssp_coil_prob_test100():

    # read seqs
    seqs = protfasta.read_fasta('test_data/test_seqs_100.fasta')

    # read precomputed data
    with open('test_data/coil_prob_v2_default_test_seqs_100.pickle', 'rb') as f:
        test_data = pickle.load(f)


    for k in seqs:
        P = Protein(seqs[k])
        assert np.array_equal(P.predictor.dssp_coil(mode='probability'), test_data[k])


##
## The three tests below systematically test the classifier with a variety of thresholds and 
## minimum element sizes to ensure this all works as expected.
##


        
def test_dssp_helicity_class_test100_non_default():

    # read seqs
    seqs = protfasta.read_fasta('test_data/test_seqs_100.fasta')

    # read precomputed data
    with open('test_data/helicity_class_v2_non_default_test_seqs_100.pickle', 'rb') as f:
        test_data = pickle.load(f)


    for k in seqs:
        P = Protein(seqs[k])

        thresh  = test_data[k][0]
        minlen  = test_data[k][1]
        vals    = test_data[k][2]
        
        assert np.array_equal(P.predictor.dssp_helicity(threshold=thresh, minimum_helical_length=minlen), vals)


def test_dssp_extended_class_test100_non_default():

    # read seqs
    seqs = protfasta.read_fasta('test_data/test_seqs_100.fasta')

    # read precomputed data
    with open('test_data/extended_class_v2_non_default_test_seqs_100.pickle', 'rb') as f:
        test_data = pickle.load(f)


    for k in seqs:
        P = Protein(seqs[k])

        thresh  = test_data[k][0]
        minlen  = test_data[k][1]
        vals    = test_data[k][2]
        
        assert np.array_equal(P.predictor.dssp_extended(threshold=thresh, minimum_extended_length=minlen), vals)
        

def test_dssp_coil_class_test100_non_default():

    # read seqs
    seqs = protfasta.read_fasta('test_data/test_seqs_100.fasta')

    # read precomputed data
    with open('test_data/coil_class_v2_non_default_test_seqs_100.pickle', 'rb') as f:
        test_data = pickle.load(f)


    for k in seqs:
        P = Protein(seqs[k])

        thresh  = test_data[k][0]
        minlen  = test_data[k][1]
        vals    = test_data[k][2]
        
        assert np.array_equal(P.predictor.dssp_coil(threshold=thresh, minimum_coil_length=minlen), vals)
        
