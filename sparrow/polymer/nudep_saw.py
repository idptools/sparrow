"""A self-avoiding walk with a free scaling exponent (nu-dependent SAW), computed
by the afrc package.

The same scaling form as :mod:`sparrow.polymer.saw`, but with the Flory scaling
exponent :math:`\\nu` as a parameter, so one model spans a collapsed globule
(:math:`\\nu \\approx 1/3`), the ideal chain (:math:`\\nu = 0.5`) and a
good-solvent coil (:math:`\\nu \\approx 0.588`). Like the SAW, it is
composition-independent: the sequence sets only the number of residues.

All distances are in Angstroms.

References
----------
Zheng, W., Zerze, G. H., Borgia, A., Mittal, J., Schuler, B., & Best, R. B.
(2018). Inferring properties of disordered chains from FRET transfer
efficiencies. *J. Chem. Phys.*, 148(12), 123329.

Soranno, A. (2020). Physical basis of the disorder-order transition. *Arch.
Biochem. Biophys.*, 685, 108305.
"""

from __future__ import annotations

import numpy as np
from afrc.polymer_models.nudep_saw import NuDepSAW
from numpy.typing import NDArray

from sparrow.polymer.saw import DEFAULT_P_OF_R_RESOLUTION_A, DEFAULT_PREFACTOR_A

__all__ = [
    "DEFAULT_NU",
    "end_to_end_distribution",
    "mean_end_to_end_distance",
    "mean_radius_of_gyration",
]

#: The scaling exponent used unless another is given, that of an ideal chain.
DEFAULT_NU = 0.5


def end_to_end_distribution(
    sequence: str,
    nu: float = DEFAULT_NU,
    prefactor: float = DEFAULT_PREFACTOR_A,
    p_of_r_resolution: float = DEFAULT_P_OF_R_RESOLUTION_A,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The nu-dependent SAW end-to-end distance distribution for a sequence's length.

    Parameters
    ----------
    sequence : str
        Amino acid sequence; only its length is used.
    nu : float, optional
        Flory scaling exponent, which should fall between 0.33 and 0.6. Default
        0.5.
    prefactor : float, optional
        The prefactor that sets the chain's size, in Angstroms. Default 5.5.
    p_of_r_resolution : float, optional
        Bin width of the distribution, in Angstroms. Default 0.05.

    Returns
    -------
    tuple of np.ndarray
        ``(distances, probabilities)``: the distances in Angstroms and the
        probability of each.
    """
    return NuDepSAW(sequence, p_of_r_resolution).get_end_to_end_distribution(nu=nu, prefactor=prefactor)


def mean_end_to_end_distance(
    sequence: str,
    nu: float = DEFAULT_NU,
    prefactor: float = DEFAULT_PREFACTOR_A,
    p_of_r_resolution: float = DEFAULT_P_OF_R_RESOLUTION_A,
) -> float:
    """The mean nu-dependent SAW end-to-end distance for a sequence's length.

    The mean is taken over :func:`end_to_end_distribution`, so it depends very
    slightly on the bin width.

    Parameters
    ----------
    sequence : str
        Amino acid sequence; only its length is used.
    nu : float, optional
        Flory scaling exponent, which should fall between 0.33 and 0.6. Default
        0.5.
    prefactor : float, optional
        The prefactor that sets the chain's size, in Angstroms. Default 5.5.
    p_of_r_resolution : float, optional
        Bin width of the distribution, in Angstroms. Default 0.05.

    Returns
    -------
    float
        The mean end-to-end distance, in Angstroms.
    """
    return NuDepSAW(sequence, p_of_r_resolution).get_mean_end_to_end_distance(nu=nu, prefactor=prefactor)


def mean_radius_of_gyration(
    sequence: str,
    nu: float = DEFAULT_NU,
    prefactor: float = DEFAULT_PREFACTOR_A,
    p_of_r_resolution: float = DEFAULT_P_OF_R_RESOLUTION_A,
) -> float:
    """The nu-dependent SAW radius of gyration for a sequence's length.

    afrc calculates it from the root-mean-square end-to-end distance of
    :func:`end_to_end_distribution` and the universal ratio of the two
    mean-squared radii at this ``nu``.

    Parameters
    ----------
    sequence : str
        Amino acid sequence; only its length is used.
    nu : float, optional
        Flory scaling exponent, which should fall between 0.33 and 0.6. Default
        0.5.
    prefactor : float, optional
        The prefactor that sets the chain's size, in Angstroms. Default 5.5.
    p_of_r_resolution : float, optional
        Bin width of the end-to-end distance distribution, in Angstroms. Default
        0.05.

    Returns
    -------
    float
        The root-mean-square radius of gyration, in Angstroms.
    """
    return NuDepSAW(sequence, p_of_r_resolution).get_mean_radius_of_gyration(nu=nu, prefactor=prefactor)
