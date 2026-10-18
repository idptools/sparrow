"""The Analytical Flory Random Coil (AFRC), computed by the afrc package.

The AFRC gives, analytically, the dimensions a sequence would have as a Flory
random coil: a chain with only local conformational preferences and no long-range
interactions, whose dimensions scale as :math:`N^{0.5}`. It is a
sequence-specific reference against which to judge whether a disordered region
is more compact or more expanded than a random coil. See
https://github.com/idptools/afrc.

The distributions are on afrc's fixed 0.05 Angstrom grid; ``AnalyticalFRC`` does
not take a bin width. All distances are in Angstroms.

References
----------
Alston, J. J., Ginell, G. M., Soranno, A., & Holehouse, A. S. (2023). The
Analytical Flory Random Coil is a simple-to-use reference model for unfolded and
disordered proteins. *J. Phys. Chem. B*, 127(21), 4746-4760.
"""

from __future__ import annotations

import numpy as np
from afrc import AnalyticalFRC
from numpy.typing import NDArray

__all__ = [
    "DEFAULT_END_TO_END_MODE",
    "DEFAULT_INTERNAL_SCALING_MODE",
    "DEFAULT_RADIUS_OF_GYRATION_MODE",
    "end_to_end_distribution",
    "internal_scaling",
    "mean_end_to_end_distance",
    "mean_radius_of_gyration",
    "radius_of_gyration_distribution",
]

#: How :func:`mean_end_to_end_distance` calculates the mean unless told
#: otherwise, as afrc does: from the scaling law.
DEFAULT_END_TO_END_MODE = "scaling law"

#: How :func:`mean_radius_of_gyration` calculates the mean unless told otherwise,
#: as afrc does: from the full distribution.
DEFAULT_RADIUS_OF_GYRATION_MODE = "distribution"

#: How :func:`internal_scaling` calculates its distances unless told otherwise, as
#: afrc does: from the scaling law.
DEFAULT_INTERNAL_SCALING_MODE = "scaling law"


def _model(sequence: str) -> AnalyticalFRC:
    """Build the AFRC of a sequence, on afrc's fixed 0.05 Angstrom grid.

    Parameters
    ----------
    sequence : str
        Amino acid sequence, standard residues only.

    Returns
    -------
    afrc.AnalyticalFRC
        The model, which afrc builds in well under a millisecond, so it is built
        afresh for every calculation rather than cached.
    """
    # AnalyticalFRC's second argument is adaptable_P_res, not a bin width: a bin
    # width passed there switches on afrc's length-dependent grid (3.7N/500
    # Angstroms), so it is given by name
    return AnalyticalFRC(sequence, adaptable_P_res=False)


def end_to_end_distribution(sequence: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The AFRC end-to-end distance distribution of a sequence.

    Parameters
    ----------
    sequence : str
        Amino acid sequence, standard residues only.

    Returns
    -------
    tuple of np.ndarray
        ``(distances, probabilities)``: the distances in Angstroms, on a 0.05
        Angstrom grid, and the probability of each.
    """
    return _model(sequence).get_end_to_end_distribution()


def radius_of_gyration_distribution(sequence: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The AFRC radius of gyration distribution of a sequence.

    The AFRC has no closed form for this distribution; afrc uses Equation 3 of
    Lhuillier, D. (1988). A simple model for polymeric fractals in a good solvent
    and an improved version of the Flory approximation. *J. Phys. France*, 49,
    705-710.

    Parameters
    ----------
    sequence : str
        Amino acid sequence, standard residues only.

    Returns
    -------
    tuple of np.ndarray
        ``(radii, probabilities)``: the radii of gyration in Angstroms, on a 0.05
        Angstrom grid, and the probability of each.
    """
    return _model(sequence).get_radius_of_gyration_distribution()


def mean_end_to_end_distance(sequence: str, mode: str = DEFAULT_END_TO_END_MODE) -> float:
    """The mean AFRC end-to-end distance of a sequence.

    Parameters
    ----------
    sequence : str
        Amino acid sequence, standard residues only.
    mode : str, optional
        ``'scaling law'`` (the default), which uses
        :math:`\\langle R_e \\rangle = R_0 N^{0.5}`, or ``'distribution'``, the
        mean of the full distribution.

    Returns
    -------
    float
        The mean end-to-end distance, in Angstroms.
    """
    return _model(sequence).get_mean_end_to_end_distance(calculation_mode=mode)


def mean_radius_of_gyration(sequence: str, mode: str = DEFAULT_RADIUS_OF_GYRATION_MODE) -> float:
    """The mean AFRC radius of gyration of a sequence.

    Parameters
    ----------
    sequence : str
        Amino acid sequence, standard residues only.
    mode : str, optional
        ``'distribution'`` (the default), the mean of the full distribution, or
        ``'scaling law'``, which uses :math:`\\langle R_g \\rangle = R_0^g N^{0.5}`.

    Returns
    -------
    float
        The mean radius of gyration, in Angstroms.
    """
    return _model(sequence).get_mean_radius_of_gyration(calculation_mode=mode)


def internal_scaling(sequence: str, mode: str = DEFAULT_INTERNAL_SCALING_MODE) -> NDArray[np.float64]:
    """The AFRC internal scaling profile of a sequence.

    The profile is the mean distance between residues ``|i - j|`` apart in
    sequence, averaged over every such pair, for every separation. A log-log fit
    of it gives a slope of 0.5, the AFRC's scaling exponent.

    Parameters
    ----------
    sequence : str
        Amino acid sequence, standard residues only.
    mode : str, optional
        ``'scaling law'`` (the default) or ``'distribution'``, which computes
        every separation's mean from its full distribution and is far slower
        (about 30 s for 1,000 residues).

    Returns
    -------
    np.ndarray
        An ``(N - 1, 2)`` array for a sequence of N residues: each row is a
        separation ``|i - j|`` and the mean distance between residues that far
        apart, in Angstroms.
    """
    return _model(sequence).get_internal_scaling(calculation_mode=mode)
