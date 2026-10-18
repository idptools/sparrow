"""The polymer-model accessor reached as ``Protein.polymeric``.

:class:`Polymeric` gathers, for one sequence, the ALBATROSS predictions of its
dimensions and the polymer models in the rest of this package, and caches every
result. The models themselves are plain functions of a sequence, one module per
model:

* :mod:`sparrow.polymer.analytical_frc` - the Analytical Flory Random Coil (AFRC)
* :mod:`sparrow.polymer.saw` - a self-avoiding walk, whose size depends only on
  the length
* :mod:`sparrow.polymer.nudep_saw` - a self-avoiding walk with a free scaling
  exponent
* :mod:`sparrow.polymer.zheng2020` - the empirical estimates of Zheng et al.
  (2020), from sequence hydropathy and charge decoration
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar

import numpy as np
from numpy.typing import NDArray

from sparrow.polymer import analytical_frc, nudep_saw, saw, zheng2020
from sparrow.tools.utilities import validate_keyword_option, warn_renamed

if TYPE_CHECKING:
    from sparrow.protein import Protein

__all__ = ["Polymeric"]

# The empirical parameterizations empirical_nu and empirical_radius_of_gyration
# know
EMPIRICAL_MODES = ["zheng2020"]

# The methods that lost their get_ prefix: old name -> new name. The old names
# still work, with a DeprecationWarning, for one release.
_RENAMED_METHODS = {
    "get_predicted_nu_dep_end_to_end_distribution": "predicted_nu_dep_end_to_end_distribution",
    "get_afrc_end_to_end_distribution": "afrc_end_to_end_distribution",
    "get_afrc_radius_of_gyration_distribution": "afrc_radius_of_gyration_distribution",
    "get_mean_afrc_end_to_end_distance": "mean_afrc_end_to_end_distance",
    "get_mean_afrc_radius_of_gyration": "mean_afrc_radius_of_gyration",
    "get_afrc_internal_scaling": "afrc_internal_scaling",
    "get_saw_end_to_end_distribution": "saw_end_to_end_distribution",
    "get_mean_saw_end_to_end_distance": "mean_saw_end_to_end_distance",
    "get_mean_saw_radius_of_gyration": "mean_saw_radius_of_gyration",
    "get_nudep_saw_end_to_end_distribution": "nudep_saw_end_to_end_distribution",
    "get_mean_nudep_saw_end_to_end_distance": "mean_nudep_saw_end_to_end_distance",
    "get_mean_nudep_saw_radius_of_gyration": "mean_nudep_saw_radius_of_gyration",
}

Result = TypeVar("Result")
Distribution = tuple[NDArray[np.float64], NDArray[np.float64]]


class Polymeric:
    """Predicted and model polymer properties of a sequence.

    The ALBATROSS predictions of the sequence's dimensions, the Analytical Flory
    Random Coil, the self-avoiding walk with a fixed and a free scaling
    exponent, and the empirical estimates of Zheng et al. (2020). Each is
    calculated on first request and cached.

    Of course, most of these are only meaningful if the sequence behaves as an
    intrinsically disordered or unfolded polypeptide.
    """

    def __init__(
        self,
        protein_obj: Protein,
        p_of_r_resolution: float = saw.DEFAULT_P_OF_R_RESOLUTION_A,
    ) -> None:
        """Init method for the Polymeric class.

        A ``Polymeric`` object is not normally constructed directly - it is
        reached as the cached ``Protein.polymeric`` accessor.

        Parameters
        ----------
        protein_obj : sparrow.protein.Protein
            The protein whose sequence is modelled, and whose ALBATROSS
            predictors give the predicted properties.

        p_of_r_resolution : float
            Bin width for building the SAW and nu-dependent SAW probability
            distributions, defined in Angstroms, by default 0.05. The AFRC
            distributions always use afrc's own fixed 0.05 Angstrom grid,
            because ``AnalyticalFRC`` does not take a bin width.
        """
        self.__protein = protein_obj
        self.__p_of_r_resolution = p_of_r_resolution

        # the ALBATROSS predictions, made on first request
        self.__predicted_scaling_exponent = None
        self.__predicted_radius_of_gyration = None
        self.__predicted_end_to_end_distance = None
        self.__predicted_asphericity = None
        self.__predicted_prefactor = None

        # every polymer-model result, keyed by the function that calculated it
        # and the arguments it was called with
        self.__precomputed: dict[tuple, Any] = {}

    def __getattr__(self, name: str) -> Any:
        """Resolve a method's old ``get_`` name, with a ``DeprecationWarning``.

        Python calls this only for a name the object does not otherwise have.

        Parameters
        ----------
        name : str
            The attribute asked for.

        Returns
        -------
        Any
            The renamed method.

        Raises
        ------
        AttributeError
            If ``name`` is not one of the renamed methods.
        """
        if name not in _RENAMED_METHODS:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        new_name = _RENAMED_METHODS[name]
        warn_renamed(f"Polymeric.{name}", f"Polymeric.{new_name}")
        return getattr(self, new_name)

    def __cached(self, recompute: bool, function: Callable[..., Result], *arguments: Any) -> Result:
        """Call ``function(sequence, *arguments)``, unless it has been called already.

        The cache key is the function and every argument, so a result calculated
        with one mode, nu or prefactor is never returned for a request with
        another.

        Parameters
        ----------
        recompute : bool
            If True, calculate the result again even if it is cached.
        function : callable
            A polymer-model function, which takes the sequence first.
        *arguments
            The function's remaining arguments, in order.

        Returns
        -------
        Any
            What ``function`` returns.
        """
        key = (function, *arguments)
        if recompute or key not in self.__precomputed:
            self.__precomputed[key] = function(self.__protein.sequence, *arguments)
        return self.__precomputed[key]

    ##########################  ALBATROSS PREDICTIONS  ##########################

    @property
    def predicted_nu(self) -> float:
        """The scaling exponent (nu) of the sequence, predicted by ALBATROSS.

        Returns
        -------
        float
            The scaling exponent, from the ALBATROSS network trained on
            simulations of single disordered chains.
        """
        if self.__predicted_scaling_exponent is None:
            self.__predicted_scaling_exponent = self.__protein.predictor.scaling_exponent()
        return self.__predicted_scaling_exponent

    @property
    def predicted_rg(self) -> float:
        """The mean radius of gyration (Rg) of the sequence, predicted by ALBATROSS.

        Returns
        -------
        float
            The radius of gyration in Angstroms, from the ALBATROSS network
            trained on simulations of single disordered chains.
        """
        if self.__predicted_radius_of_gyration is None:
            self.__predicted_radius_of_gyration = self.__protein.predictor.radius_of_gyration()
        return self.__predicted_radius_of_gyration

    @property
    def predicted_re(self) -> float:
        """The mean end-to-end distance (Re) of the sequence, predicted by ALBATROSS.

        Returns
        -------
        float
            The end-to-end distance in Angstroms, from the ALBATROSS network
            trained on simulations of single disordered chains.
        """
        if self.__predicted_end_to_end_distance is None:
            self.__predicted_end_to_end_distance = self.__protein.predictor.end_to_end_distance()
        return self.__predicted_end_to_end_distance

    @property
    def predicted_asphericity(self) -> float:
        """The mean asphericity (δ) of the sequence, predicted by ALBATROSS.

        Returns
        -------
        float
            The asphericity, from the ALBATROSS network trained on simulations
            of single disordered chains.
        """
        if self.__predicted_asphericity is None:
            self.__predicted_asphericity = self.__protein.predictor.asphericity()
        return self.__predicted_asphericity

    @property
    def predicted_prefactor(self) -> float:
        """The scaling-law prefactor (A_0) of the sequence, predicted by ALBATROSS.

        Returns
        -------
        float
            The prefactor in Angstroms, from the ALBATROSS network trained on
            simulations of single disordered chains.
        """
        if self.__predicted_prefactor is None:
            self.__predicted_prefactor = self.__protein.predictor.prefactor()
        return self.__predicted_prefactor

    def predicted_nu_dep_end_to_end_distribution(self) -> Distribution:
        """The end-to-end distance distribution of the nu-dependent SAW with the
        predicted scaling exponent and prefactor.

        Both the scaling exponent (nu) and the prefactor are taken from the
        ALBATROSS predictors (see :attr:`predicted_nu` and
        :attr:`predicted_prefactor`), so this function takes no arguments. Use
        :meth:`nudep_saw_end_to_end_distribution` to set nu and the prefactor
        yourself.

        Returns
        -------
        tuple of np.ndarray
            ``(distances, probabilities)``: the distances in Angstroms and the
            probability of each.
        """
        return self.__cached(
            False,
            nudep_saw.end_to_end_distribution,
            self.predicted_nu,
            self.predicted_prefactor,
            self.__p_of_r_resolution,
        )

    ##########################  ANALYTICAL FLORY RANDOM COIL  ##########################

    def afrc_end_to_end_distribution(self, recompute: bool = False) -> Distribution:
        """The end-to-end distance distribution of the Analytical Flory Random Coil.

        See :func:`sparrow.polymer.analytical_frc.end_to_end_distribution`.

        Parameters
        ----------
        recompute : bool, optional
            If True, recompute the distribution even if it has already been
            calculated. Default is False.

        Returns
        -------
        tuple of np.ndarray
            ``(distances, probabilities)``: the distances in Angstroms, on a 0.05
            Angstrom grid, and the probability of each.
        """
        return self.__cached(recompute, analytical_frc.end_to_end_distribution)

    def afrc_radius_of_gyration_distribution(self, recompute: bool = False) -> Distribution:
        """The radius of gyration distribution of the Analytical Flory Random Coil.

        See :func:`sparrow.polymer.analytical_frc.radius_of_gyration_distribution`.

        Parameters
        ----------
        recompute : bool, optional
            If True, recompute the distribution even if it has already been
            calculated. Default is False.

        Returns
        -------
        tuple of np.ndarray
            ``(radii, probabilities)``: the radii of gyration in Angstroms, on a
            0.05 Angstrom grid, and the probability of each.
        """
        return self.__cached(recompute, analytical_frc.radius_of_gyration_distribution)

    def mean_afrc_end_to_end_distance(
        self,
        mode: str = analytical_frc.DEFAULT_END_TO_END_MODE,
        recompute: bool = False,
    ) -> float:
        """The mean end-to-end distance of the Analytical Flory Random Coil.

        See :func:`sparrow.polymer.analytical_frc.mean_end_to_end_distance`.

        Parameters
        ----------
        mode : str, optional
            ``'scaling law'`` (the default), which uses
            :math:`\\langle R_e \\rangle = R_0 N^{0.5}`, or ``'distribution'``,
            the mean of the full distribution.
        recompute : bool, optional
            If True, recompute the mean end-to-end distance even if a value for
            this ``mode`` has already been calculated. Default is False.

        Returns
        -------
        float
            The mean end-to-end distance, in Angstroms.
        """
        return self.__cached(recompute, analytical_frc.mean_end_to_end_distance, mode)

    def mean_afrc_radius_of_gyration(
        self,
        mode: str = analytical_frc.DEFAULT_RADIUS_OF_GYRATION_MODE,
        recompute: bool = False,
    ) -> float:
        """The mean radius of gyration of the Analytical Flory Random Coil.

        See :func:`sparrow.polymer.analytical_frc.mean_radius_of_gyration`.

        Parameters
        ----------
        mode : str, optional
            ``'distribution'`` (the default), the mean of the full distribution,
            or ``'scaling law'``, which uses
            :math:`\\langle R_g \\rangle = R_0^g N^{0.5}`.
        recompute : bool, optional
            If True, recompute the mean radius of gyration even if a value for
            this ``mode`` has already been calculated. Default is False.

        Returns
        -------
        float
            The mean radius of gyration, in Angstroms.
        """
        return self.__cached(recompute, analytical_frc.mean_radius_of_gyration, mode)

    def afrc_internal_scaling(
        self,
        mode: str = analytical_frc.DEFAULT_INTERNAL_SCALING_MODE,
        recompute: bool = False,
    ) -> NDArray[np.float64]:
        """The internal scaling profile of the Analytical Flory Random Coil.

        The mean distance between residues ``|i - j|`` apart in sequence, for
        every separation. See
        :func:`sparrow.polymer.analytical_frc.internal_scaling`.

        Parameters
        ----------
        mode : str, optional
            ``'scaling law'`` (the default) or ``'distribution'``, which is far
            slower (about 30 s for 1,000 residues).
        recompute : bool, optional
            If True, recompute the internal scaling profile even if a value for
            this ``mode`` has already been calculated. Default is False.

        Returns
        -------
        np.ndarray
            An ``(N - 1, 2)`` array for a sequence of N residues: each row is a
            separation ``|i - j|`` and the mean distance between residues that
            far apart, in Angstroms.
        """
        return self.__cached(recompute, analytical_frc.internal_scaling, mode)

    ##########################  SELF-AVOIDING WALK  ##########################

    def saw_end_to_end_distribution(
        self,
        prefactor: float = saw.DEFAULT_PREFACTOR_A,
        recompute: bool = False,
    ) -> Distribution:
        """The end-to-end distance distribution of the self-avoiding walk.

        This is a composition-independent model, in which the end-to-end
        distance depends only on the number of residues, included as a reference
        model. See :func:`sparrow.polymer.saw.end_to_end_distribution`.

        Parameters
        ----------
        prefactor : float, optional
            The prefactor that sets the SAW's size, in Angstroms. 5 Angstroms is
            the right scale, but it should be tuned to match excluded-volume
            simulations. Default 5.5.
        recompute : bool, optional
            If True, recompute the distribution even if one for this
            ``prefactor`` has already been calculated. Default is False.

        Returns
        -------
        tuple of np.ndarray
            ``(distances, probabilities)``: the distances in Angstroms and the
            probability of each.
        """
        return self.__cached(recompute, saw.end_to_end_distribution, prefactor, self.__p_of_r_resolution)

    def mean_saw_end_to_end_distance(
        self,
        prefactor: float = saw.DEFAULT_PREFACTOR_A,
        recompute: bool = False,
    ) -> float:
        """The mean end-to-end distance of the self-avoiding walk.

        See :func:`sparrow.polymer.saw.mean_end_to_end_distance`.

        Parameters
        ----------
        prefactor : float, optional
            The prefactor that sets the SAW's size, in Angstroms. Default 5.5.
        recompute : bool, optional
            If True, recompute the mean end-to-end distance even if a value for
            this ``prefactor`` has already been calculated. Default is False.

        Returns
        -------
        float
            The mean end-to-end distance, in Angstroms.
        """
        return self.__cached(recompute, saw.mean_end_to_end_distance, prefactor, self.__p_of_r_resolution)

    def mean_saw_radius_of_gyration(
        self,
        prefactor: float = saw.DEFAULT_PREFACTOR_A,
        recompute: bool = False,
    ) -> float:
        """The radius of gyration of the self-avoiding walk.

        See :func:`sparrow.polymer.saw.mean_radius_of_gyration`.

        Parameters
        ----------
        prefactor : float, optional
            The prefactor that sets the SAW's size, in Angstroms. Default 5.5.
        recompute : bool, optional
            If True, recompute the radius of gyration even if a value for this
            ``prefactor`` has already been calculated. Default is False.

        Returns
        -------
        float
            The root-mean-square radius of gyration, in Angstroms.
        """
        return self.__cached(recompute, saw.mean_radius_of_gyration, prefactor, self.__p_of_r_resolution)

    ##########################  NU-DEPENDENT SELF-AVOIDING WALK  ##########################

    def nudep_saw_end_to_end_distribution(
        self,
        nu: float = nudep_saw.DEFAULT_NU,
        prefactor: float = saw.DEFAULT_PREFACTOR_A,
        recompute: bool = False,
    ) -> Distribution:
        """The end-to-end distance distribution of the nu-dependent self-avoiding walk.

        This is a composition-independent model, in which the end-to-end
        distance depends only on the number of residues; both nu and the
        prefactor can be varied. See
        :func:`sparrow.polymer.nudep_saw.end_to_end_distribution`.

        Parameters
        ----------
        nu : float, optional
            Flory scaling exponent, which should fall between 0.33 and 0.6.
            Default 0.5.
        prefactor : float, optional
            The prefactor that sets the chain's size, in Angstroms. Default 5.5.
        recompute : bool, optional
            If True, recompute the distribution even if one for this ``nu`` and
            ``prefactor`` has already been calculated. Default is False.

        Returns
        -------
        tuple of np.ndarray
            ``(distances, probabilities)``: the distances in Angstroms and the
            probability of each.
        """
        return self.__cached(
            recompute,
            nudep_saw.end_to_end_distribution,
            nu,
            prefactor,
            self.__p_of_r_resolution,
        )

    def mean_nudep_saw_end_to_end_distance(
        self,
        nu: float = nudep_saw.DEFAULT_NU,
        prefactor: float = saw.DEFAULT_PREFACTOR_A,
        recompute: bool = False,
    ) -> float:
        """The mean end-to-end distance of the nu-dependent self-avoiding walk.

        See :func:`sparrow.polymer.nudep_saw.mean_end_to_end_distance`.

        Parameters
        ----------
        nu : float, optional
            Flory scaling exponent, which should fall between 0.33 and 0.6.
            Default 0.5.
        prefactor : float, optional
            The prefactor that sets the chain's size, in Angstroms. Default 5.5.
        recompute : bool, optional
            If True, recompute the mean end-to-end distance even if a value for
            this ``nu`` and ``prefactor`` has already been calculated. Default
            is False.

        Returns
        -------
        float
            The mean end-to-end distance, in Angstroms.
        """
        return self.__cached(
            recompute,
            nudep_saw.mean_end_to_end_distance,
            nu,
            prefactor,
            self.__p_of_r_resolution,
        )

    def mean_nudep_saw_radius_of_gyration(
        self,
        nu: float = nudep_saw.DEFAULT_NU,
        prefactor: float = saw.DEFAULT_PREFACTOR_A,
        recompute: bool = False,
    ) -> float:
        """The radius of gyration of the nu-dependent self-avoiding walk.

        See :func:`sparrow.polymer.nudep_saw.mean_radius_of_gyration`.

        Parameters
        ----------
        nu : float, optional
            Flory scaling exponent, which should fall between 0.33 and 0.6.
            Default 0.5.
        prefactor : float, optional
            The prefactor that sets the chain's size, in Angstroms. Default 5.5.
        recompute : bool, optional
            If True, recompute the radius of gyration even if a value for this
            ``nu`` and ``prefactor`` has already been calculated. Default is
            False.

        Returns
        -------
        float
            The root-mean-square radius of gyration, in Angstroms.
        """
        return self.__cached(
            recompute,
            nudep_saw.mean_radius_of_gyration,
            nu,
            prefactor,
            self.__p_of_r_resolution,
        )

    #################  EMPIRICAL FUNCTIONS FROM PAPERS BELOW HERE  #################

    def empirical_nu(self, mode: str = "zheng2020") -> float:
        """The scaling exponent (nu) of the sequence from an empirical relationship.

        The only relationship so far is that of Zheng et al. (2020), from the
        sequence hydropathy decoration (SHD) and sequence charge decoration
        (SCD):

        .. math::

           \\nu = -0.0423\\,\\mathrm{SHD} + 0.0074\\,\\mathrm{SCD} + 0.701

        See :func:`sparrow.polymer.zheng2020.scaling_exponent`.

        Parameters
        ----------
        mode : str, optional
            Selector for the empirical parameterization to use. The only
            currently-implemented option is ``'zheng2020'`` (the default);
            passing anything else raises an exception.

        Returns
        -------
        float
            The scaling exponent, which is dimensionless and should, in
            principle, fall between 0.33 and 0.6.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If ``mode`` is not a recognised parameterization.

        References
        ----------
        Zheng, W., Dignon, G. L., Brown, M., Kim, Y. C. & Mittal, J. (2020).
        Hydropathy patterning complements charge patterning to describe
        conformational preferences of disordered proteins. *J. Phys. Chem.
        Lett.*, 11(9), 3408-3415. doi:10.1021/acs.jpclett.0c00288
        """
        validate_keyword_option(mode, EMPIRICAL_MODES, "mode")
        return zheng2020.scaling_exponent(self.__protein.sequence)

    def empirical_radius_of_gyration(self, mode: str = "zheng2020") -> float:
        """The radius of gyration of the sequence from an empirical relationship.

        The only relationship so far is that of Zheng et al. (2020): the
        nu-dependent radius of gyration at the scaling exponent of
        :meth:`empirical_nu`. See
        :func:`sparrow.polymer.zheng2020.radius_of_gyration`.

        Parameters
        ----------
        mode : str, optional
            Selector for the empirical parameterization to use. The only
            currently-implemented option is ``'zheng2020'`` (the default);
            passing anything else raises an exception.

        Returns
        -------
        float
            The radius of gyration, in Angstroms.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If ``mode`` is not a recognised parameterization.

        References
        ----------
        Zheng, W., Dignon, G. L., Brown, M., Kim, Y. C. & Mittal, J. (2020).
        Hydropathy patterning complements charge patterning to describe
        conformational preferences of disordered proteins. *J. Phys. Chem.
        Lett.*, 11(9), 3408-3415. doi:10.1021/acs.jpclett.0c00288
        """
        validate_keyword_option(mode, EMPIRICAL_MODES, "mode")
        return zheng2020.radius_of_gyration(self.__protein.sequence)
