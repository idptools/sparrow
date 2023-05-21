from sparrow import Protein
import pytest
import protfasta
import os
import numpy as np
from sparrow.predictors.batch_predict import batch_predict
from sparrow import read_fasta as read_fasta_sparrow


current_filepath = os.getcwd()
onehundred_seqs = "{}/test_data/test_seqs_100.fasta".format(current_filepath)

seqs = protfasta.read_fasta(onehundred_seqs)
seqs_sparrow = read_fasta_sparrow(onehundred_seqs)


##
## ALBATROSS tests for albatross V1 (May 5th 2023)
##
## If/when the networks are rebuilt, please recalculate these values using
## the notebook in test_data.
##
## Tests for V2 added May 21st 2023)
##


from . import build_seq




def test_rg_prediction_v1():

    ## Batch prediction works the same way for all networks, so we test the rg prediction extensively in terms
    ## of input output but do not do the same level of testing for the other networks because the coding logic
    ## is literally the same

    
    uid_2_rg_v1 = np.load('test_data/test_100_rg.npy', allow_pickle=True).item()
    

    ## return_seq2prediction is True
    # input dict of sparrow.protein.Protein objects
    v1 = batch_predict({k:Protein(seqs[k]) for k in seqs}, version=1, batch_size=1, network='rg', return_seq2prediction=True)
    for k in seqs:
        s = seqs[k]        
        assert  v1[s] ==  uid_2_rg_v1[k]

    # input dict of strings
    v1 = batch_predict(seqs, version=1, batch_size=1, network='rg', return_seq2prediction=True)
    for k in seqs:
        s = seqs[k]        
        assert  v1[s] ==  uid_2_rg_v1[k]

    # input list of sparrow.protein.Protein objects
    v1 = batch_predict([Protein(seqs[k]) for k in seqs], version=1, batch_size=1, network='rg', return_seq2prediction=True)
    keys = list(seqs.keys())
    for idx in range(len(keys)):
        k = keys[idx]
        s = seqs[k]
        assert  v1[s] ==  uid_2_rg_v1[k]

    # input list of sparrow.protein.Protein objects
    v1 = batch_predict(list(seqs.values()), version=1, batch_size=1, network='rg', return_seq2prediction=True)
    keys = list(seqs.keys())
    for idx in range(len(keys)):
        k = keys[idx]
        s = seqs[k]
        assert  v1[s] ==  uid_2_rg_v1[k]


        
    ## return_seq2prediction is False
    # input dict of sparrow.protein.Protein objects
    v1 = batch_predict({k:Protein(seqs[k]) for k in seqs}, version=1, batch_size=1, network='rg')
    for k in seqs:
        s = seqs[k]        
        assert  v1[k][1] ==  uid_2_rg_v1[k]
        assert  v1[k][0] ==  s

    # v1 test (using str input)
    v1 = batch_predict(seqs, version=1, batch_size=1, network='rg')
    for k in seqs:
        s = seqs[k]        
        assert  v1[k][1] ==  uid_2_rg_v1[k]
        assert  v1[k][0] ==  s

    # input list of sparrow.protein.Protein objects
    v1 = batch_predict([Protein(seqs[k]) for k in seqs], version=1, batch_size=1, network='rg')
    keys = list(seqs.keys())
    for idx in range(len(keys)):
        k = keys[idx]
        assert  v1[idx][1] ==  uid_2_rg_v1[k]


    # input list of sparrow.protein.Protein objects
    v1 = batch_predict(list(seqs.values()), version=1, batch_size=1, network='rg')
    keys = list(seqs.keys())
    for idx in range(len(keys)):
        k = keys[idx]
        assert  v1[idx][1] ==  uid_2_rg_v1[k]



def test_rg_prediction_v2():

    
    uid_2_rg_v2 = np.load('test_data/test_100_rg_v2.npy', allow_pickle=True).item()

    ## return_seq2prediction is True
    # input dict of sparrow.protein.Protein objects
    v2 = batch_predict({k:Protein(seqs[k]) for k in seqs}, version=2, batch_size=1, network='rg', return_seq2prediction=True)
    for k in seqs:
        s = seqs[k]        
        assert  v2[s] ==  uid_2_rg_v2[k]

    # input dict of strings
    v2 = batch_predict(seqs, version=2, batch_size=1, network='rg', return_seq2prediction=True)
    for k in seqs:
        s = seqs[k]        
        assert  v2[s] ==  uid_2_rg_v2[k]

    # input list of sparrow.protein.Protein objects
    v2 = batch_predict([Protein(seqs[k]) for k in seqs], version=2, batch_size=1, network='rg', return_seq2prediction=True)
    keys = list(seqs.keys())
    for idx in range(len(keys)):
        k = keys[idx]
        s = seqs[k]
        assert  v2[s] ==  uid_2_rg_v2[k]

    # input list of sparrow.protein.Protein objects
    v2 = batch_predict(list(seqs.values()), version=2, batch_size=1, network='rg', return_seq2prediction=True)
    keys = list(seqs.keys())
    for idx in range(len(keys)):
        k = keys[idx]
        s = seqs[k]
        assert  v2[s] ==  uid_2_rg_v2[k]

        
    ## return_seq2prediction is False
    # input dict of sparrow.protein.Protein objects
    v2 = batch_predict({k:Protein(seqs[k]) for k in seqs}, version=2, batch_size=1, network='rg')
    for k in seqs:
        s = seqs[k]        
        assert  v2[k][1] ==  uid_2_rg_v2[k]
        assert  v2[k][0] ==  s

    # v2 test (using str input)
    v2 = batch_predict(seqs, version=2, batch_size=1, network='rg')
    for k in seqs:
        s = seqs[k]        
        assert  v2[k][1] ==  uid_2_rg_v2[k]
        assert  v2[k][0] ==  s

    # input list of sparrow.protein.Protein objects
    v2 = batch_predict([Protein(seqs[k]) for k in seqs], version=2, batch_size=1, network='rg')
    keys = list(seqs.keys())
    for idx in range(len(keys)):
        k = keys[idx]
        assert  v2[idx][1] ==  uid_2_rg_v2[k]


    # input list of sparrow.protein.Protein objects
    v2 = batch_predict(list(seqs.values()), version=2, batch_size=1, network='rg')
    keys = list(seqs.keys())
    for idx in range(len(keys)):
        k = keys[idx]
        assert  v2[idx][1] ==  uid_2_rg_v2[k]
        
        

def test_rg_scaled_prediction_v1():
    uid_2_rg_scaled_v1 = np.load('test_data/test_100_rg_scaled.npy', allow_pickle=True).item()

    ## return_seq2prediction is True
    # input dict of sparrow.protein.Protein objects
    v1 = batch_predict({k:Protein(seqs[k]) for k in seqs}, version=1, batch_size=1, network='scaled_rg', return_seq2prediction=True)
    for k in seqs:
        s = seqs[k]        
        assert  v1[s] ==  uid_2_rg_scaled_v1[k]


def test_rg_scaled_prediction_v2():
    uid_2_rg_scaled_v2 = np.load('test_data/test_100_rg_scaled_v2.npy', allow_pickle=True).item()

    ## return_seq2prediction is True
    # input dict of sparrow.protein.Protein objects
    v2 = batch_predict({k:Protein(seqs[k]) for k in seqs}, version=2, batch_size=1, network='scaled_rg', return_seq2prediction=True)
    for k in seqs:
        s = seqs[k]        
        assert  v2[s] ==  uid_2_rg_scaled_v2[k]


def test_re_prediction_v1():
    uid_2_rg_scaled_v1 = np.load('test_data/test_100_re.npy', allow_pickle=True).item()

    ## return_seq2prediction is True
    # input dict of sparrow.protein.Protein objects
    v1 = batch_predict({k:Protein(seqs[k]) for k in seqs}, version=1, batch_size=1, network='re', return_seq2prediction=True)
    for k in seqs:
        s = seqs[k]        
        assert  v1[s] ==  uid_2_rg_scaled_v1[k]


def test_re_prediction_v2():
    uid_2_rg_scaled_v2 = np.load('test_data/test_100_re_v2.npy', allow_pickle=True).item()

    ## return_seq2prediction is True
    # input dict of sparrow.protein.Protein objects
    v2 = batch_predict({k:Protein(seqs[k]) for k in seqs}, version=2, batch_size=1, network='re', return_seq2prediction=True)
    for k in seqs:
        s = seqs[k]        
        assert  v2[s] ==  uid_2_rg_scaled_v2[k]
        


def test_re_scaled_prediction_v1():
    uid_2_rg_scaled_v1 = np.load('test_data/test_100_re_scaled.npy', allow_pickle=True).item()

    ## return_seq2prediction is True
    # input dict of sparrow.protein.Protein objects
    v1 = batch_predict({k:Protein(seqs[k]) for k in seqs}, version=1, batch_size=1, network='scaled_re', return_seq2prediction=True)
    for k in seqs:
        s = seqs[k]        
        assert  v1[s] ==  uid_2_rg_scaled_v1[k]


def test_re_scaled_prediction_v2():
    uid_2_rg_scaled_v2 = np.load('test_data/test_100_re_scaled_v2.npy', allow_pickle=True).item()

    ## return_seq2prediction is True
    # input dict of sparrow.protein.Protein objects
    v2 = batch_predict({k:Protein(seqs[k]) for k in seqs}, version=2, batch_size=1, network='scaled_re', return_seq2prediction=True)
    for k in seqs:
        s = seqs[k]        
        assert  v2[s] ==  uid_2_rg_scaled_v2[k]
        

def test_asph_prediction_v1():
    uid_2_data= np.load('test_data/test_100_asph.npy', allow_pickle=True).item()

    ## return_seq2prediction is True
    # input dict of sparrow.protein.Protein objects
    v1 = batch_predict({k:Protein(seqs[k]) for k in seqs}, version=1, batch_size=1, network='asphericity', return_seq2prediction=True)
    for k in seqs:
        s = seqs[k]        
        assert  v1[s] ==  uid_2_data[k]

def test_asph_prediction_v2():
    uid_2_data= np.load('test_data/test_100_asph_v2.npy', allow_pickle=True).item()

    ## return_seq2prediction is True
    # input dict of sparrow.protein.Protein objects
    v2 = batch_predict({k:Protein(seqs[k]) for k in seqs}, version=2, batch_size=1, network='asphericity', return_seq2prediction=True)
    for k in seqs:
        s = seqs[k]        
        assert  v2[s] ==  uid_2_data[k]

def test_scaling_exponent_prediction_v1():
    uid_2_data = np.load('test_data/test_100_exponent.npy', allow_pickle=True).item()

    ## return_seq2prediction is True
    # input dict of sparrow.protein.Protein objects
    v1 = batch_predict({k:Protein(seqs[k]) for k in seqs}, version=1, batch_size=1, network='scaling_exponent', return_seq2prediction=True)
    for k in seqs:
        s = seqs[k]        
        assert  v1[s] ==  uid_2_data[k]

def test_scaling_exponent_prediction_v2():        
    uid_2_data = np.load('test_data/test_100_exponent_v2.npy', allow_pickle=True).item()

    ## return_seq2prediction is True
    # input dict of sparrow.protein.Protein objects
    v2 = batch_predict({k:Protein(seqs[k]) for k in seqs}, version=2, batch_size=1, network='scaling_exponent', return_seq2prediction=True)
    for k in seqs:
        s = seqs[k]        
        assert  v2[s] ==  uid_2_data[k]
        


def test_scaling_prefactor_prediction_v1():
    uid_2_data = np.load('test_data/test_100_prefactor.npy', allow_pickle=True).item()

    ## return_seq2prediction is True
    # input dict of sparrow.protein.Protein objects
    v1 = batch_predict({k:Protein(seqs[k]) for k in seqs}, version=1, batch_size=1, network='prefactor', return_seq2prediction=True)
    for k in seqs:
        s = seqs[k]        
        assert  v1[s] ==  uid_2_data[k]

def test_scaling_prefactor_prediction_v2():
    uid_2_data = np.load('test_data/test_100_prefactor_v2.npy', allow_pickle=True).item()

    ## return_seq2prediction is True
    # input dict of sparrow.protein.Protein objects
    v2 = batch_predict({k:Protein(seqs[k]) for k in seqs}, version=2, batch_size=1, network='prefactor', return_seq2prediction=True)
    for k in seqs:
        s = seqs[k]        
        assert  v2[s] ==  uid_2_data[k]
        
        
        
def test_batch_vs_single_rg():
    # assert same value comes from batch vs. single prediction
    # for a set of randomly generated sequences
    n_seqs = 100
    seqs = []
    for i in range(n_seqs):
        seqs.append(build_seq())
    
    
    batch_pred = batch_predict(seqs, batch_size=1, network='rg')
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.radius_of_gyration(), batch_pred[idx][1], atol=1e-05)        

        
def test_batch_vs_single_re():
    # assert same value comes from batch vs. single prediction
    # for a set of randomly generated sequences
    n_seqs = 100
    seqs = []
    for i in range(n_seqs):
        seqs.append(build_seq())
    
    
    batch_pred = batch_predict(seqs, batch_size=1, network='re')
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.end_to_end_distance(), batch_pred[idx][1], atol=1e-05)        





def test_batch_vs_single_rg_scaled():
    # assert same value comes from batch vs. single prediction
    # for a set of randomly generated sequences
    n_seqs = 100
    seqs = []
    for i in range(n_seqs):
        seqs.append(build_seq())
    
    
    batch_pred = batch_predict(seqs, batch_size=1, network='scaled_rg')
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.radius_of_gyration(use_scaled=True), batch_pred[idx][1], atol=1e-05)        

        
def test_batch_vs_single_re_scaled():
    # assert same value comes from batch vs. single prediction
    # for a set of randomly generated sequences
    n_seqs = 100
    seqs = []
    for i in range(n_seqs):
        seqs.append(build_seq())
    
    
    batch_pred = batch_predict(seqs, batch_size=1, network='scaled_re')
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.end_to_end_distance(use_scaled=True), batch_pred[idx][1], atol=1e-05)        


def test_batch_vs_single_asph():
    # assert same value comes from batch vs. single prediction
    # for a set of randomly generated sequences
    n_seqs = 100
    seqs = []
    for i in range(n_seqs):
        seqs.append(build_seq())
    
    
    batch_pred = batch_predict(seqs, batch_size=1, network='asphericity')
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.asphericity(), batch_pred[idx][1], atol=1e-05)        
        



def test_batch_vs_single_scaling_exponent():
    # assert same value comes from batch vs. single prediction
    # for a set of randomly generated sequences
    n_seqs = 100
    seqs = []
    for i in range(n_seqs):
        seqs.append(build_seq())
    
    
    batch_pred = batch_predict(seqs, batch_size=1, network='scaling_exponent')
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.scaling_exponent(), batch_pred[idx][1], atol=1e-05)        


def test_batch_vs_single_prefactor():
    # assert same value comes from batch vs. single prediction
    # for a set of randomly generated sequences
    n_seqs = 100
    seqs = []
    for i in range(n_seqs):
        seqs.append(build_seq())
    
    
    batch_pred = batch_predict(seqs, batch_size=1, network='prefactor')
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.prefactor(), batch_pred[idx][1], atol=1e-05)        
        

        
def test_batch_vs_single_rg_short_seqs_only():
    """
    Test explicitly looking at seqs that are between 3 and 30 residues in
    length to ensure batch and non- batch give same answers both by default
    and when safe=False
    
    """

    
    n_seqs = 100
    seqs = []
    for i in range(n_seqs):
        seqs.append(build_seq(5,20)[0:np.random.randint(3,30)])

    # first check safe defaults
    batch_pred = batch_predict(seqs, batch_size=1, network='rg')
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.radius_of_gyration(), batch_pred[idx][1], atol=1e-05)        

    batch_pred = batch_predict(seqs, batch_size=1, network='scaled_rg')
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.radius_of_gyration(use_scaled=True), batch_pred[idx][1], atol=1e-05)        

    # next check unsafe defaults
    batch_pred = batch_predict(seqs, batch_size=1, network='rg', safe=False)
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.radius_of_gyration(safe=False), batch_pred[idx][1], atol=1e-05)        

    batch_pred = batch_predict(seqs, batch_size=1, network='scaled_rg', safe=False)
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.radius_of_gyration(use_scaled=True, safe=False), batch_pred[idx][1], atol=1e-05)        



def test_batch_vs_single_rg_short_and_long():
    """
    Test explicitly looking at seqs that are between 3 and 30 residues in
    length to ensure batch and non- batch give same answers both by default
    and when safe=False
    
    """

    
    n_seqs = 50
    seqs = []
    for i in range(n_seqs):
        seqs.append(build_seq(5,20)[0:np.random.randint(3,30)])

    for i in range(n_seqs):
        seqs.append(build_seq(5,20))
        
    # first check safe defaults
    batch_pred = batch_predict(seqs, batch_size=1, network='rg')
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.radius_of_gyration(), batch_pred[idx][1], atol=1e-05)        

    batch_pred = batch_predict(seqs, batch_size=1, network='scaled_rg')
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.radius_of_gyration(use_scaled=True), batch_pred[idx][1], atol=1e-05)        

    # next check unsafe defaults
    batch_pred = batch_predict(seqs, batch_size=1, network='rg', safe=False)
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.radius_of_gyration(safe=False), batch_pred[idx][1], atol=1e-05)        

    batch_pred = batch_predict(seqs, batch_size=1, network='scaled_rg', safe=False)
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.radius_of_gyration(use_scaled=True, safe=False), batch_pred[idx][1], atol=1e-05)        
        

def test_batch_vs_single_re_short_and_long():
    """
    Test explicitly looking at seqs that are between 3 and 30 residues in
    length to ensure batch and non- batch give same answers both by default
    and when safe=False
    
    """

    
    n_seqs = 50
    seqs = []
    for i in range(n_seqs):
        seqs.append(build_seq(5,20)[0:np.random.randint(3,30)])

    for i in range(n_seqs):
        seqs.append(build_seq(5,20))
        

    # first check safe defaults
    batch_pred = batch_predict(seqs, batch_size=1, network='re')
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.end_to_end_distance(), batch_pred[idx][1], atol=1e-05)        

    batch_pred = batch_predict(seqs, batch_size=1, network='scaled_re')
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.end_to_end_distance(use_scaled=True), batch_pred[idx][1], atol=1e-05)        

    # next check unsafe defaults
    batch_pred = batch_predict(seqs, batch_size=1, network='re', safe=False)
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.end_to_end_distance(safe=False), batch_pred[idx][1], atol=1e-05)        

    batch_pred = batch_predict(seqs, batch_size=1, network='scaled_re', safe=False)
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.end_to_end_distance(use_scaled=True, safe=False), batch_pred[idx][1], atol=1e-05)        


def test_batch_vs_single_re_short_seqs_only():
    """
    Test explicitly looking at seqs that are between 3 and 30 residues in
    length to ensure batch and non- batch give same answers both by default
    and when safe=False
    
    """

    
    n_seqs = 100
    seqs = []
    for i in range(n_seqs):
        seqs.append(build_seq(5,20)[0:np.random.randint(3,30)])

    # first check safe defaults
    batch_pred = batch_predict(seqs, batch_size=1, network='re')
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.end_to_end_distance(), batch_pred[idx][1], atol=1e-05)        

    batch_pred = batch_predict(seqs, batch_size=1, network='scaled_re')
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.end_to_end_distance(use_scaled=True), batch_pred[idx][1], atol=1e-05)        

    # next check unsafe defaults
    batch_pred = batch_predict(seqs, batch_size=1, network='re', safe=False)
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.end_to_end_distance(safe=False), batch_pred[idx][1], atol=1e-05)        

    batch_pred = batch_predict(seqs, batch_size=1, network='scaled_re', safe=False)
    for idx in batch_pred:
        assert np.isclose(Protein(seqs[idx]).predictor.end_to_end_distance(use_scaled=True, safe=False), batch_pred[idx][1], atol=1e-05)        



def test_batch_vs_single_batchsize():
    # assert same value comes from batch vs. single prediction
    # for a set of randomly generated sequences

    for n_seqs in [1, 10, 50, 200]:
        seqs = []
        for i in range(n_seqs):
            seqs.append(build_seq())

        for bs in [1,10,32,120,2000]:
    
            batch_pred = batch_predict(seqs, batch_size=bs, network='rg')
            for idx in batch_pred:
                assert np.isclose(Protein(seqs[idx]).predictor.radius_of_gyration(), batch_pred[idx][1], atol=1e-05)        
        


def test_batch_size_impact_all_networks():
    # assert same value comes from batch vs. single prediction
    # for a set of randomly generated sequences

    for nw in ['rg', 're', 'asphericity', 'scaled_rg', 'scaled_re', 'scaling_exponent', 'prefactor']:

        for n_seqs in [1, 10, 50, 20]:
            seqs = []
            for i in range(n_seqs):
                seqs.append(build_seq())

            for bs in [1,10,32,120,2000]:
    
                batch_pred_big = batch_predict(seqs, batch_size=bs, network=nw)
                batch_pred_1 = batch_predict(seqs, batch_size=1, network=nw)
                
                for idx in batch_pred_big:
                    assert np.isclose(batch_pred_big[idx][1], batch_pred_1[idx][1] , atol=1e-05)
                
