"""A self-avoiding walk (SAW), computed by the afrc package.

The SAW is the des Cloizeaux scaling form as implemented by O'Brien et al.
(2009), with the scaling exponent fixed at afrc's good-solvent value. It is a
composition-independent reference: the sequence sets only the number of
residues, and the prefactor sets the overall size. To vary the scaling exponent
as well, use :mod:`sparrow.polymer.nudep_saw`.

All distances are in Angstroms.

References
----------
O'Brien, E. P., Morrison, G., Brooks, B. R., & Thirumalai, D. (2009). How
accurate are polymer models in the analysis of Forster resonance energy transfer
experiments on proteins? *J. Chem. Phys.*, 130(12), 124903.
"""

from __future__ import annotations

import numpy as np
from afrc.polymer_models.saw import SAW
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_PREFACTOR_A",
    "DEFAULT_P_OF_R_RESOLUTION_A",
    "end_to_end_distribution",
    "mean_end_to_end_distance",
    "mean_radius_of_gyration",
]

#: The prefactor that sets the SAW's size, in Angstroms, which is afrc's
#: default. About 5 Angstroms is the right scale, but it should be tuned to
#: match excluded-volume simulations.
DEFAULT_PREFACTOR_A = 5.5

#: Bin width of the distance distributions, in Angstroms, which is afrc's
#: default and the fixed grid of :mod:`sparrow.polymer.analytical_frc`.
DEFAULT_P_OF_R_RESOLUTION_A = 0.05


def end_to_end_distribution(
    sequence: str,
    prefactor: float = DEFAULT_PREFACTOR_A,
    p_of_r_resolution: float = DEFAULT_P_OF_R_RESOLUTION_A,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The SAW end-to-end distance distribution for a sequence's length.

    Parameters
    ----------
    sequence : str
        Amino acid sequence; only its length is used.
    prefactor : float, optional
        The prefactor that sets the SAW's size, in Angstroms. Default 5.5.
    p_of_r_resolution : float, optional
        Bin width of the distribution, in Angstroms. Default 0.05.

    Returns
    -------
    tuple of np.ndarray
        ``(distances, probabilities)``: the distances in Angstroms and the
        probability of each.
    """
    return SAW(sequence, p_of_r_resolution).get_end_to_end_distribution(prefactor)


def mean_end_to_end_distance(
    sequence: str,
    prefactor: float = DEFAULT_PREFACTOR_A,
    p_of_r_resolution: float = DEFAULT_P_OF_R_RESOLUTION_A,
) -> float:
    """The mean SAW end-to-end distance for a sequence's length.

    The mean is taken over :func:`end_to_end_distribution`, so it depends very
    slightly on the bin width.

    Parameters
    ----------
    sequence : str
        Amino acid sequence; only its length is used.
    prefactor : float, optional
        The prefactor that sets the SAW's size, in Angstroms. Default 5.5.
    p_of_r_resolution : float, optional
        Bin width of the distribution, in Angstroms. Default 0.05.

    Returns
    -------
    float
        The mean end-to-end distance, in Angstroms.
    """
    return SAW(sequence, p_of_r_resolution).get_mean_end_to_end_distance(prefactor)


def mean_radius_of_gyration(
    sequence: str,
    prefactor: float = DEFAULT_PREFACTOR_A,
    p_of_r_resolution: float = DEFAULT_P_OF_R_RESOLUTION_A,
) -> float:
    """The SAW radius of gyration for a sequence's length.

    afrc calculates it from the root-mean-square end-to-end distance of
    :func:`end_to_end_distribution` and the SAW's universal ratio of the two
    mean-squared radii.

    Parameters
    ----------
    sequence : str
        Amino acid sequence; only its length is used.
    prefactor : float, optional
        The prefactor that sets the SAW's size, in Angstroms. Default 5.5.
    p_of_r_resolution : float, optional
        Bin width of the end-to-end distance distribution, in Angstroms. Default
        0.05.

    Returns
    -------
    float
        The root-mean-square radius of gyration, in Angstroms.
    """
    return SAW(sequence, p_of_r_resolution).get_mean_radius_of_gyration(prefactor)
