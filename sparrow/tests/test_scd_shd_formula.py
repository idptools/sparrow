"""Canonical-formula regression tests for SCD and SHD (sparrow.patterning.scd)."""

import math

import pytest

from sparrow import Protein
from sparrow.patterning import scd


def _reference_scd(seq, group1=("E", "D"), group2=("R", "K")):
    """Independent Sawle-Ghosh SCD over all pairs m > n (including neighbors)."""
    charge = {}
    for r in group1:
        charge[r] = -1
    for r in group2:
        charge[r] = 1
    n = len(seq)
    total = 0.0
    for m in range(1, n):
        for k in range(0, m):
            total += charge.get(seq[m], 0) * charge.get(seq[k], 0) * math.sqrt(m - k)
    return total / n


def _reference_shd(seq, hydro):
    n = len(seq)
    total = 0.0
    for m in range(1, n):
        for k in range(m):
            total += (hydro[seq[m]] + hydro[seq[k]]) / abs(m - k)
    return total / n


SEQUENCES = [
    "ED",                       # 2 residues: only a nearest-neighbor pair
    "EDED",
    "EKEKEKEKEKEK",
    "EEEEKKKK",
    "RDRDEKEKEDRK",
    "ACDEFGHIKLMNPQRSTVWY",
]


@pytest.mark.parametrize("seq", SEQUENCES)
def test_compute_scd_matches_reference(seq):
    # regression: a previous bug dropped nearest-neighbor (|m-n|=1) pairs
    assert scd.compute_scd_x(seq) == pytest.approx(_reference_scd(seq), abs=1e-9)


def test_scd_nearest_neighbor_pair_is_counted():
    # 'ED': single adjacent E/D pair -> (-1)(-1)*sqrt(1) / 2 = 0.5
    assert scd.compute_scd_x("ED") == pytest.approx(0.5)


def test_scd_custom_groups_swap_sign():
    seq = "EDEDRKRK"
    forward = scd.compute_scd_x(seq, group1=["E", "D"], group2=["R", "K"])
    swapped = scd.compute_scd_x(seq, group1=["R", "K"], group2=["E", "D"])
    assert forward == pytest.approx(swapped)


def test_compute_shd_matches_reference():
    seq = "ACDEFGHIKLMNPQRSTVWY"
    hydro = {r: float(i + 1) for i, r in enumerate(seq)}
    assert scd.compute_shd(seq, hydro_dict=hydro) == pytest.approx(
        _reference_shd(seq, hydro), abs=1e-9
    )


def test_protein_scd_matches_functional():
    seq = "EDEDRKRKEDRKACDEFG"
    p = Protein(seq)
    assert p.SCD == pytest.approx(scd.compute_scd_x(seq))
    assert p.compute_SCD_x(["E", "D"], ["R", "K"]) == pytest.approx(p.SCD)


def test_protein_shd_matches_functional():
    seq = "EDEDRKRKEDRKACDEFG"
    p = Protein(seq)
    assert p.SHD == pytest.approx(scd.compute_shd(seq))


def test_compute_shd_custom_requires_all_residues():
    p = Protein("ACDEK")
    with pytest.raises(Exception):
        p.compute_SHD_custom({"A": 1.0})  # missing residues
