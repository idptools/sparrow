from sparrow.protein_batch import ProteinBatch
from sparrow.batch_calculations.batch_tools import seqs_to_matrix
from sparrow.batch_calculations import batch_properties
from sparrow.protein import Protein
import protfasta
import time

# read in seqs
test_seqs=protfasta.read_fasta('./test_data/1000_human_seqs.fasta', invalid_sequence_action='convert')
test_seqs=list(test_seqs.values())
test_seqs=sorted(test_seqs, key=len)

# sort by length. 
seq_batch=ProteinBatch(test_seqs)

# make list of Protein objects
list_seq_objs=[]
for s in test_seqs:
    list_seq_objs.append(Protein(s))


def test_molecular_weight():
    """
    Test the molecular weight calculation for a batch of sequences.
    """
    batch_vals = seq_batch.molecular_weight
    for num, batch_val in enumerate(batch_vals):
        assert abs(batch_val - list_seq_objs[num].molecular_weight) < 0.0001


def test_amino_acid_fractions():
    """
    Test the amino acid fractions calculation for a batch of sequences.
    """
    batch_vals = seq_batch.amino_acid_fractions
    for num, batch_val in enumerate(batch_vals):
        assert batch_val == list_seq_objs[num].amino_acid_fractions

def test_hydrophobicity():
    """
    Test the hydrophobicity calculation for a batch of sequences.
    """
    batch_vals = seq_batch.hydrophobicity
    for num, batch_val in enumerate(batch_vals):
        assert abs(batch_val - list_seq_objs[num].hydrophobicity) < 0.0001

def test_FCR():
    """
    Test the FCR calculation for a batch of sequences.
    """
    batch_vals = seq_batch.FCR
    for num, batch_val in enumerate(batch_vals):
        assert abs(batch_val - list_seq_objs[num].FCR) < 0.0001

def test_NCPR():
    """
    Test the NCPR calculation for a batch of sequences.
    """
    batch_vals = seq_batch.NCPR
    for num, batch_val in enumerate(batch_vals):
        assert abs(batch_val - list_seq_objs[num].NCPR) < 0.0001

def test_fraction_aromatic():
    """
    Test the fraction aromatic calculation for a batch of sequences.
    """
    batch_vals = seq_batch.fraction_aromatic
    for num, batch_val in enumerate(batch_vals):
        assert abs(batch_val - list_seq_objs[num].fraction_aromatic) < 0.0001

def test_fraction_polar():
    """
    Test the fraction polar calculation for a batch of sequences.
    """
    batch_vals = seq_batch.fraction_polar
    for num, batch_val in enumerate(batch_vals):
        assert abs(batch_val - list_seq_objs[num].fraction_polar) < 0.0001

def test_fraction_aliphatic():
    """
    Test the fraction aliphatic calculation for a batch of sequences.
    """
    batch_vals = seq_batch.fraction_aliphatic
    for num, batch_val in enumerate(batch_vals):
        assert abs(batch_val - list_seq_objs[num].fraction_aliphatic) < 0.0001

def test_fraction_positive():
    """
    Test the fraction positive calculation for a batch of sequences.
    """
    batch_vals = seq_batch.fraction_positive
    for num, batch_val in enumerate(batch_vals):
        assert abs(batch_val - list_seq_objs[num].fraction_positive) < 0.0001

def test_fraction_negative():
    """
    Test the fraction negative calculation for a batch of sequences.
    """
    batch_vals = seq_batch.fraction_negative
    for num, batch_val in enumerate(batch_vals):
        assert abs(batch_val - list_seq_objs[num].fraction_negative) < 0.0001

def test_fraction_proline():
    """
    Test the fraction proline calculation for a batch of sequences.
    """
    batch_vals = seq_batch.fraction_proline
    for num, batch_val in enumerate(batch_vals):
        assert abs(batch_val - list_seq_objs[num].fraction_proline) < 0.0001

def test_complexity():
    """
    Test the complexity calculation for a batch of sequences.
    """
    batch_vals = seq_batch.complexity
    for num, batch_val in enumerate(batch_vals):
        assert abs(batch_val - list_seq_objs[num].complexity) < 0.0001

def test_kappa():
    """
    Test the kappa calculation for a batch of sequences.
    """
    batch_vals = seq_batch.kappa
    for num, batch_val in enumerate(batch_vals):
        assert abs(batch_val - list_seq_objs[num].kappa) < 0.0001

def test_IWD():
    """
    Test the IWD calculation for a batch of sequences.
    """
    batch_vals = seq_batch.compute_iwd(target_residues='D')
    for num, batch_val in enumerate(batch_vals):
        assert abs(batch_val - list_seq_objs[num].compute_iwd(target_residues='D')) < 0.0001

def test_SCD():
    """
    Test the SCD calculation for a batch of sequences.
    """
    batch_vals = seq_batch.SCD
    for num, batch_val in enumerate(batch_vals):
        assert abs(batch_val - list_seq_objs[num].SCD) < 0.0001

def test_SHD():
    """
    Test the SHD calculation for a batch of sequences.
    """
    batch_vals = seq_batch.SHD
    for num, batch_val in enumerate(batch_vals):
        assert abs(batch_val - list_seq_objs[num].SHD) < 0.0001