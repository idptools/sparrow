from sparrow.patterning import scd
import os
import protfasta
import numpy as np
from sparrow import Protein
from IPython import embed

current_filepath = os.getcwd()
onehundred_seqs = "{}/test_data/test_seqs_100.fasta".format(current_filepath)

seqs = protfasta.read_fasta(onehundred_seqs)

def test_scd():

    k2val = np.load('test_data/test_100_scd.npy', allow_pickle=True).item()
    for k in seqs:
        s = seqs[k]
        cython_SCD = getattr(Protein(s),"SCD")
        no_cython_SCD = k2val[k]
        assert np.isclose(cython_SCD, no_cython_SCD)

def test_shd():
    k2val = np.load('test_data/test_100_shd.npy', allow_pickle=True).item()
    for k in seqs:
        s = seqs[k]
        assert np.isclose(getattr(Protein(s),"SHD"), k2val[k])

