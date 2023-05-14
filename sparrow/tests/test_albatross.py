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
## the notebook in test_data (
##


def test_rg_prediction():
    uid_2_rg = np.load('test_data/test_100_rg.npy', allow_pickle=True).item()

    for k in seqs:        
        assert Protein(seqs[k]).predictor.radius_of_gyration() ==  uid_2_rg[k]


def test_rg_scaled_prediction():
    uid_2_rg = np.load('test_data/test_100_rg_scaled.npy', allow_pickle=True).item()

    for k in seqs:        
        assert Protein(seqs[k]).predictor.radius_of_gyration(use_scaled=True) ==  uid_2_rg[k]
        

def test_re_prediction():
    uid_2_re = np.load('test_data/test_100_re.npy', allow_pickle=True).item()

    for k in seqs:        
        assert Protein(seqs[k]).predictor.end_to_end_distance() ==  uid_2_re[k]


def test_re_scaled_prediction():
    uid_2_re = np.load('test_data/test_100_re_scaled.npy', allow_pickle=True).item()

    for k in seqs:        
        assert Protein(seqs[k]).predictor.end_to_end_distance(use_scaled=True) ==  uid_2_re[k]
        

def test_asph_prediction():
    uid_2_asph = np.load('test_data/test_100_asph.npy', allow_pickle=True).item()

    for k in seqs:        
        assert Protein(seqs[k]).predictor.asphericity() ==  uid_2_asph[k]


def test_scaling_exponent_prediction():
    uid_2_sclexp = np.load('test_data/test_100_exponent.npy', allow_pickle=True).item()

    for k in seqs:        
        assert Protein(seqs[k]).predictor.scaling_exponent() ==  uid_2_sclexp[k]


def test_scaling_prefactor_prediction():
    uid_2_pref = np.load('test_data/test_100_prefactor.npy', allow_pickle=True).item()

    for k in seqs:        
        assert Protein(seqs[k]).predictor.prefactor() ==  uid_2_pref[k]

        
def test_batch_vs_single_rg():
    # assert same value comes from batch vs. single prediction
    batch_pred = batch_predict(seqs_sparrow, batch_size=1, network='rg')
    for s in batch_pred:
        assert np.isclose(Protein(s).predictor.radius_of_gyration(), batch_pred[s], atol=1e-05)
        

def test_batch_vs_single_rg_scaled():
    # assert same value comes from batch vs. single prediction
    batch_pred = batch_predict(seqs_sparrow, batch_size=1, network='scaled_rg')
    for s in batch_pred:
        assert np.isclose(Protein(s).predictor.radius_of_gyration(use_scaled=True), batch_pred[s], atol=1e-05)
        
def test_batch_vs_single_re():
    # assert same value comes from batch vs. single prediction
    batch_pred = batch_predict(seqs_sparrow, batch_size=1, network='re')
    for s in batch_pred:
        assert np.isclose(Protein(s).predictor.end_to_end_distance(), batch_pred[s], atol=1e-05)
        

def test_batch_vs_single_re_scaled():
    # assert same value comes from batch vs. single prediction
    batch_pred = batch_predict(seqs_sparrow, batch_size=1, network='scaled_re')
    for s in batch_pred:
        assert np.isclose(Protein(s).predictor.end_to_end_distance(use_scaled=True), batch_pred[s], atol=1e-05)
        
