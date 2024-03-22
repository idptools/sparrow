import os

import numpy as np
import protfasta

from sparrow.polymer import scaling_parameters

current_filepath = os.getcwd()
onehundred_seqs = f"{current_filepath}/test_data/test_seqs_100.fasta"

seqs = protfasta.read_fasta(onehundred_seqs)


def test_empirical_nu():
    k2val = np.load('test_data/test_100_empirical_nu.npy', allow_pickle=True).item()        
    for k in seqs:                
        assert np.isclose(scaling_parameters.compute_nu_zheng2020(seqs[k]), k2val[k])


def test_empirical_radius_of_gyration():
    k2val = np.load('test_data/test_100_empirical_rg.npy', allow_pickle=True).item()        
    for k in seqs:                
        assert np.isclose(scaling_parameters.compute_rg_zheng2020(seqs[k]), k2val[k])




