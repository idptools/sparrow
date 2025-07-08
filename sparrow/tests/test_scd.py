import os

import numpy as np
import protfasta
from IPython import embed

from sparrow import Protein
from sparrow.patterning import scd

current_filepath = os.getcwd()
onehundred_seqs = "{}/test_data/test_seqs_100.fasta".format(current_filepath)

seqs = protfasta.read_fasta(onehundred_seqs)
shd_paper_toy_sequences = np.load(
    "test_data/shd_paper_toy_sequences.npy", allow_pickle=True
).item()


def test_scd():
    k2val = np.load("test_data/test_100_scd.npy", allow_pickle=True).item()
    for k in seqs:
        s = seqs[k]
        cython_SCD = getattr(Protein(s), "SCD")
        no_cython_SCD = k2val[k]
        assert np.isclose(cython_SCD, no_cython_SCD)


def test_shd():
    k2val = np.load("test_data/test_100_shd.npy", allow_pickle=True).item()
    for k in seqs:
        s = seqs[k]
        assert np.isclose(getattr(Protein(s), "SHD"), k2val[k])


def test_shd_custom_binary():
    """
    SHD computed on binary '0'/'1' strings using hydro_dict={'0':0., '1':1.}
    Compares to the values initially reported in the SHD supplemental materials [1]

    References:

    [1] Zheng, W.; Dignon, G.; Brown, M.; Kim, Y. C.; Mittal, J. Hydropathy Patterning Complements Charge Patterning to Describe Conformational Preferences of Disordered Proteins. J. Phys. Chem. Lett. 2020, 11 (9), 3408-3415.

    """
    hydro_dict = {"0": 0.0, "1": 1.0}
    for seq, true_val in shd_paper_toy_sequences.items():
        calc = scd.compute_shd(seq, hydro_dict=hydro_dict)
        assert np.isclose(calc, true_val, atol=1e-3), f"{seq}: {calc} != {true_val}"
