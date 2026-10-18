"""Empirical estimates of a disordered sequence's dimensions from Zheng et al. (2020).

Zheng et al. fitted the Flory scaling exponent of simulated disordered sequences
to two patterning parameters: the sequence hydropathy decoration (SHD) and the
sequence charge decoration (SCD). The radius of gyration
follows from that exponent through the nu-dependent self-avoiding walk.

References
----------
Zheng, W., Dignon, G. L., Brown, M., Kim, Y. C., & Mittal, J. (2020).
Hydropathy patterning complements charge patterning to describe conformational
preferences of disordered proteins. *J. Phys. Chem. Lett.*, 11(9), 3408-3415.
doi:10.1021/acs.jpclett.0c00288

Le Guillou, J. C., & Zinn-Justin, J. (1977). Critical exponents for the n-vector
model in three dimensions from field theory. *Phys. Rev. Lett.*, 39(2), 95-98.
"""

from __future__ import annotations

import numpy as np

from sparrow.properties import patterning

__all__ = [
    "NU_INTERCEPT",
    "NU_SCD_COEFFICIENT",
    "NU_SHD_COEFFICIENT",
    "RADIUS_OF_GYRATION_PREFACTOR_A",
    "SAW_GAMMA",
    "radius_of_gyration",
    "scaling_exponent",
]

#: The coefficient of the SHD in Zheng et al.'s fit of the scaling exponent,
#: :math:`\nu = -0.0423\,\mathrm{SHD} + 0.0074\,\mathrm{SCD} + 0.701`.
NU_SHD_COEFFICIENT = -0.0423

#: The coefficient of the SCD in Zheng et al.'s fit of the scaling exponent.
NU_SCD_COEFFICIENT = 0.0074

#: The intercept of Zheng et al.'s fit of the scaling exponent.
NU_INTERCEPT = 0.701

#: The self-avoiding walk's critical exponent :math:`\gamma` (Le Guillou &
#: Zinn-Justin 1977), which sets the ratio of the radius of gyration to the
#: end-to-end distance at a given scaling exponent.
SAW_GAMMA = 1.1615

#: Zheng et al.'s prefactor :math:`b`, 0.55 nm, in Angstroms.
RADIUS_OF_GYRATION_PREFACTOR_A = 5.5


def scaling_exponent(sequence: str) -> float:
    """The Flory scaling exponent of a sequence, from its SHD and SCD.

    .. math::

       \\nu = -0.0423\\,\\mathrm{SHD} + 0.0074\\,\\mathrm{SCD} + 0.701

    Parameters
    ----------
    sequence : str
        Amino acid sequence, standard residues only.

    Returns
    -------
    float
        The scaling exponent, which is dimensionless and should, in principle,
        fall between 0.33 and 0.6.
    """
    sequence_hydropathy_decoration = float(patterning.shd(sequence)[0])
    sequence_charge_decoration = float(patterning.scd(sequence)[0])

    return (
        (NU_SHD_COEFFICIENT * sequence_hydropathy_decoration)
        + (NU_SCD_COEFFICIENT * sequence_charge_decoration)
        + NU_INTERCEPT
    )


def radius_of_gyration(sequence: str) -> float:
    """The radius of gyration of a sequence, from its scaling exponent.

    The scaling exponent :math:`\\nu` comes from :func:`scaling_exponent`, and the
    radius of gyration from the nu-dependent self-avoiding walk:

    .. math::

       R_g = \\sqrt{\\frac{\\gamma(\\gamma + 1)}{2(\\gamma + 2\\nu)(\\gamma + 2\\nu + 1)}}
             \\; b N^{\\nu}

    with :math:`\\gamma` = 1.1615, :math:`b` = 5.5 Angstroms and :math:`N` the
    number of residues.

    Parameters
    ----------
    sequence : str
        Amino acid sequence, standard residues only.

    Returns
    -------
    float
        The radius of gyration, in Angstroms.
    """
    nu = scaling_exponent(sequence)
    number_of_residues = len(sequence)

    numerator = SAW_GAMMA * (SAW_GAMMA + 1)
    denominator = 2 * (SAW_GAMMA + 2 * nu) * (SAW_GAMMA + 2 * nu + 1)

    return np.sqrt(numerator / denominator) * RADIUS_OF_GYRATION_PREFACTOR_A * np.power(number_of_residues, nu)
