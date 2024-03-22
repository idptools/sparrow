"""
Unit and regression test for the sparrow package.
"""

# Import package, test suite, and other packages as needed
import os
import sys

import numpy as np

from sparrow.protein import Protein

current_filepath = os.getcwd()
onehundred_seqs = f"{current_filepath}/test_data/test_seqs_100.fasta"


def test_sparrow_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "sparrow" in sys.modules


def test_protein_constructor():
    s = "MKASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQ"
    # constructor
    P = Protein(s)
    assert len(P) == 61

    P = Protein(s, validate=True)
    assert len(P) == 61

    s_broken = "MkASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSXYGQSSYSSYXQ"
    P = Protein(s_broken, validate=True)
    assert len(P) == 61

    assert s == P.sequence


def test_seq_fractions():
    s = "MKASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQ"
    # constructor
    P = Protein(s)
    assert len(P.amino_acid_fractions) == 20

    aa_fraction_dict = {
        "A": 0.04918032786885246,
        "C": 0.0,
        "D": 0.03278688524590164,
        "E": 0.0,
        "F": 0.0,
        "G": 0.13114754098360656,
        "H": 0.0,
        "I": 0.0,
        "K": 0.01639344262295082,
        "L": 0.0,
        "M": 0.01639344262295082,
        "N": 0.01639344262295082,
        "P": 0.04918032786885246,
        "Q": 0.21311475409836064,
        "R": 0.0,
        "S": 0.22950819672131148,
        "T": 0.08196721311475409,
        "V": 0.0,
        "W": 0.0,
        "Y": 0.16393442622950818,
    }

    for aa in P.amino_acid_fractions:
        assert np.isclose(P.amino_acid_fractions[aa], aa_fraction_dict[aa])

    assert np.isclose(
        P.compute_residue_fractions(["P", "E", "K", "R", "D"]), 0.09836065573770492
    )


def test_standard_sequence_properties():
    s = "MKASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQ"

    P = Protein(s)
    assert np.isclose(P.molecular_weight, 6606.600000000002)
    assert np.isclose(P.FCR, 0.04918032786885246)
    assert np.isclose(P.fraction_positive, 0.01639344262295082)
    assert np.isclose(P.fraction_negative, 0.03278688524590164)
    assert np.isclose(P.NCPR, -0.01639344262295082)
    assert np.isclose(P.fraction_aromatic, 0.16393442622950818)
    assert np.isclose(P.fraction_aliphatic, 0.06557377049180328)
    assert np.isclose(P.fraction_polar, 0.6721311475409836)
    assert np.isclose(P.fraction_proline, 0.04918032786885246)
    assert np.isclose(P.hydrophobicity, 3.052459016393442)


def test_complicated_sequence_properties():
    s = "MKASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQ"

    P = Protein(s)

    # complicated sequence properties
    # IWD tested elsewhere
    assert np.isclose(P.complexity, 0.6828600567737508)
    assert np.isclose(P.kappa, 0.5236256867647171)
    assert np.isclose(
        P.compute_kappa_x(
            "K",
            "Q",
        ),
        0.1885949820280075,
    )

    assert np.isclose(P.SCD, -0.037788191722401984)
    assert np.isclose(
        P.compute_SCD_x(
            [
                "D",
                "E",
            ],
            ["K", "R"],
        ),
        -0.037788191722401984,
    )

    # TODO: add tests for SHD and SHD_custom hydropathy dictionaries.
    assert np.isclose(P.SHD, 3.5606496905252505)
    assert np.isclose(P.compute_SHD_custom(hydro_dict=False), 3.5606496905252505)


# TODO: write test for phosphoseqome
def test_phosphoseqome():
    pass


# TODO: test for arbitrary predictors not included in albatross
def test_predictors():
    pass


def test_disorder_prediction():
    s = "MKASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQ"
    P = Protein(s)

    disorder_scores = np.load(
        f"{current_filepath}/test_data/simple_sequence_disorder_scores.npy"
    )
    assert np.allclose(P.predictor.disorder(), disorder_scores)

    assert np.isclose(np.mean(P.predictor.disorder()), 0.8636131147540983)


def test_linear_sequence_profile():
    s = "MKASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQ"

    P = Protein(s)
    assert np.isclose(np.mean(P.linear_sequence_profile("FCR")), 0.04918032786885246)
    assert np.isclose(np.mean(P.linear_sequence_profile("NCPR")), -0.02459016393442623)
    
    assert np.isclose(
        np.mean(P.linear_sequence_profile("aromatic")), 0.1680327868852459
    )

    assert np.isclose(
        np.mean(P.linear_sequence_profile("aliphatic")), 0.05737704918032787
    )
    assert np.isclose(np.mean(P.linear_sequence_profile("polar")), 0.6762295081967213)
    
    assert np.isclose(
        np.mean(P.linear_sequence_profile("proline")), 0.04918032786885246
    )

    assert np.isclose(
        np.mean(P.linear_sequence_profile("positive")), 0.012295081967213115
    )

    assert np.isclose(
        np.mean(P.linear_sequence_profile("negative")), 0.036885245901639344
    )

    assert np.isclose(
        np.mean(P.linear_sequence_profile("hydrophobicity")), 3.0450819672131146
    )

    assert np.isclose(
        np.mean(P.linear_composition_profile(["E", "K"])), 0.012295081967213115
    )


# TODO: test low_complexity_domains
def test_low_complexity_domains():
    pass


def test_elm_locations():
    P = Protein("KRRARKRRARKRRARKRRAR")
    elms = P.elms
    func_sites = []
    elm_sequences = []
    start, end = [], []
    for elm in elms:
        start.append(elm.start)
        end.append(elm.end)
        elm_sequences.append(elm.sequence)
        func_sites.append(elm.functional_site_name)
    func_sites = list(set(func_sites))
    for func_site in func_sites:
        assert func_site in [
            "di Arginine retention/retrieving signal",
            "CendR Motif Binding to Neuropilin Receptors",
            "NLS classical Nuclear Localization Signals",
            "N-degron",
            "NRD cleavage site",
            "PCSK cleavage site",
        ]
    assert start == [
        1,
        6,
        11,
        16,
        4,
        9,
        14,
        0,
        5,
        10,
        15,
        0,
        5,
        10,
        15,
        1,
        11,
        0,
        16,
        1,
        6,
        11,
        16,
        0,
        3,
        13,
        4,
        14,
        1,
        9,
    ]
    assert end == [
        4,
        9,
        14,
        19,
        9,
        14,
        19,
        3,
        8,
        13,
        18,
        3,
        8,
        13,
        18,
        8,
        18,
        3,
        20,
        5,
        10,
        15,
        20,
        20,
        9,
        19,
        10,
        20,
        9,
        15,
    ]
    assert elm_sequences == [
        "RRA",
        "RRA",
        "RRA",
        "RRA",
        "RKRRA",
        "RKRRA",
        "RKRRA",
        "KRR",
        "KRR",
        "KRR",
        "KRR",
        "KRR",
        "KRR",
        "KRR",
        "KRR",
        "RRARKRR",
        "RRARKRR",
        "KRR",
        "RRAR",
        "RRAR",
        "RRAR",
        "RRAR",
        "RRAR",
        "KRRARKRRARKRRARKRRAR",
        "ARKRRA",
        "ARKRRA",
        "RKRRAR",
        "RKRRAR",
        "RRARKRRA",
        "RKRRAR",
    ]
