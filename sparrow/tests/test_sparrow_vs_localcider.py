from localcider.sequenceParameters import SequenceParameters
from sparrow import Protein

from . import build_seq

import numpy as np

NSEQS=100

def test_FCR():

    for i in range(NSEQS):
        s = build_seq()
        assert np.isclose(SequenceParameters(s).get_FCR(), Protein(s).FCR, atol=1e-8)
    

def test_NCPR():

    for i in range(NSEQS):
        s = build_seq()
        assert np.isclose(SequenceParameters(s).get_NCPR(), Protein(s).NCPR, atol=1e-8)
    
        
def test_fraction_neg_fraction_pos():

    for i in range(NSEQS):
        s = build_seq()
        assert np.isclose(SequenceParameters(s).get_countNeg()/len(s), Protein(s).fraction_negative, atol=1e-8)
        assert np.isclose(SequenceParameters(s).get_countPos()/len(s), Protein(s).fraction_positive, atol=1e-8)
    
def test_hydrophobiciyty():

    for i in range(NSEQS):
        s = build_seq()
        assert np.isclose(SequenceParameters(s).get_uversky_hydropathy(), Protein(s).hydrophobicity/9, atol=1e-8)

