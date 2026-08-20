import protfasta
import numpy as np
from sparrow import Protein

from . import reference_data_path

onehundred_seqs = reference_data_path("test_seqs_100.fasta")

seqs = protfasta.read_fasta(onehundred_seqs)


def test_average_bivariate_inverse_distance_charge():

    k2val = np.load(
        reference_data_path("test_average_bivariate_inverse_distance_charge.npy"), allow_pickle=True
    ).item()
    for k in seqs:
        assert np.isclose(Protein(seqs[k]).compute_bivariate_iwd_charged_weighted(), k2val[k])


def test_average_inverse_distance_charge_neg():

    k2val = np.load(reference_data_path("test_average_inverse_distance_charge_neg.npy"), allow_pickle=True).item()
    for k in seqs:
        assert np.isclose(Protein(seqs[k]).compute_iwd_charged_weighted("-"), k2val[k])


def test_average_inverse_distance_charge_pos():

    k2val = np.load(reference_data_path("test_average_inverse_distance_charge_pos.npy"), allow_pickle=True).item()
    for k in seqs:
        assert np.isclose(Protein(seqs[k]).compute_iwd_charged_weighted("+"), k2val[k])


def test_average_inverse_distance_ali():

    k2val = np.load(reference_data_path("test_average_inverse_distance_ali.npy"), allow_pickle=True).item()
    for k in seqs:
        assert np.isclose(Protein(seqs[k]).compute_iwd("ILVAM"), k2val[k])
