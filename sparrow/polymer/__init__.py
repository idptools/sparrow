# from afrc.polymer_models.saw import SAW
# from afrc.polymer_models.wlc import WormLikeChain
# from afrc.polymer_models.wlc2 import WormLikeChain2

from sparrow.tools.utilities import validate_keyword_option

from . import scaling_parameters


class Polymeric:
    """Base class providing programmatic access to various empirical equations and
    machine learned predictions for various different polymer properties for the sequence.

    Note that, of course, many of these would only be valid if the sequence
    behaved as an intrinsically disordered or unfolded polypeptide.
    """

    def __init__(self, protein_obj, p_of_r_resolution=0.05):
        """Init method for the Polymeric class.

        Takes a sparrow.Protein object and provides programmatic access to the
        machine-learned predictors, analytical polymer models, and empirical
        scaling relationships described below. A ``Polymeric`` object is not
        normally constructed directly - it is reached as the cached
        ``Protein.polymeric`` accessor.

        Parameters
        ----------
        protein_obj : sparrow.Protein
            Composition-based interface into the functionalities from the sparrow.Protein class.
            Primarily used to give access to PARROT trained predictor objects.

        p_of_r_resolution : float
            Bin width for building the probability distributions, defined in Angstroms, by default 0.05.
        """
        self.__protein = protein_obj
        self.__nudepsaw = None
        self.__afrc = None
        self.__saw = None
        self.__wlc = None
        self.__wlc2 = None

        # bin width parameter
        self.__p_of_r_resolution = p_of_r_resolution

        # by default, predicted polymer properties are defined as unset class variables so that
        # they're only computed upon request.
        self.__predicted_scaling_exp = None
        self.__predicted_average_rg = None
        self.__predicted_average_re = None
        self.__predicted_average_asphercity = None
        self.__predicted_prefactor = None

        # Adapted from code in AFRC package
        # set distribution info to None, lazily computed as needed
        # storing both 1) the distance distribution and
        # 2) the corresponding probability distribution
        self.__p_of_Re_R = None
        self.__p_of_Re_P = None

        # lacking closed form analytical solution here - careful!
        # set distribution info to None, lazily computed as needed
        # storing both 1) the distance distribution and
        # 2) the corresponding probability distribution
        self.__p_of_Rg_R = None
        self.__p_of_Rg_P = None

        # dictionary where computations can be memoized so that the Polymeric class
        # avoids needing to recompute (potentially expensive) predictions
        self.__precomputed = {}

    @property
    def predicted_nu(self):
        """
        This function returns the predicted scaling exponent (nu) for a given sequence
        based off of the PARROT trained sparrow networks for predicting polymeric properties.

        Returns
        -------
        float
            The sequence specific predicted scaling exponent fit from
            single chain IDR LAMMPS simulations.
        """
        if self.__predicted_scaling_exp is None:
            self.__predicted_scaling_exp = self.__protein.predictor.scaling_exponent()
        return self.__predicted_scaling_exp

    @property
    def predicted_rg(self):
        """
        This function returns the mean predicted radius of gyration (rg) for a given sequence
        based off of the PARROT trained sparrow networks for predicting polymeric properties.

        Returns
        -------
        float
            The sequence specific predicted radius of gyration parameterized from
            single chain IDR LAMMPS simulations.
        """
        if self.__predicted_average_rg is None:
            self.__predicted_average_rg = self.__protein.predictor.radius_of_gyration()
        return self.__predicted_average_rg

    @property
    def predicted_re(self):
        """
        This function returns the mean predicted end-to-end distance (re) for a given sequence
        based off of the PARROT trained sparrow networks for predicting polymeric properties.

        Returns
        -------
        float
            The sequence specific predicted radius of gyration parameterized from
            single chain IDR LAMMPS simulations.
        """
        if self.__predicted_average_re is None:
            self.__predicted_average_re = self.__protein.predictor.end_to_end_distance()
        return self.__predicted_average_re

    @property
    def predicted_asphericity(self):
        """
        This function returns the mean predicted asphericity (δ) for a given sequence
        based off of the PARROT trained sparrow networks for predicting polymeric properties.

        Returns
        -------
        float
            The sequence specific predicted asphericity fit parameterized from
            single chain IDR LAMMPS simulations.
        """
        if self.__predicted_average_asphercity is None:
            self.__predicted_average_asphercity = self.__protein.predictor.asphericity()
        return self.__predicted_average_asphercity

    @property
    def predicted_prefactor(self):
        """
        This function returns the predicted prefactor ('A_0') for a given sequence
        based off of the PARROT trained sparrow networks for predicting polymeric properties.

        Returns
        -------
        float
            The sequence specific predicted scaling exponent fit from parameterized from
            single chain IDR LAMMPS simulations.
        """
        if self.__predicted_prefactor is None:
            self.__predicted_prefactor = self.__protein.predictor.prefactor()
        return self.__predicted_prefactor

    def get_predicted_nu_dep_end_to_end_distribution(self):
        """
        Function that returns the predicted end-to-end distance distribution
        based on the predicted scaling exponent and the nu-dependent SAW model.

        Both the scaling exponent (nu) and the prefactor are taken from the
        ALBATROSS predictors (see :attr:`predicted_nu` and
        :attr:`predicted_prefactor`), so this function takes no arguments. Use
        :meth:`get_nudep_saw_end_to_end_distribution` if you want to set nu
        and/or the prefactor by hand.

        Returns
        -------
        tuple of arrays
            A 2-pair tuple of numpy arrays where the first is the distance (in
            Angstroms) and the second is the probability of that distance.
        """
        if self.__nudepsaw is None:
            from afrc.polymer_models.nudep_saw import NuDepSAW

            self.__nudepsaw = NuDepSAW(self.__protein.sequence, self.__p_of_r_resolution)

        prefactor = self.predicted_prefactor
        nu = self.predicted_nu

        # insert some sort of error handling on predicted prefactor / nu values
        # to ensure the predicted values are reasonable!

        if self.__p_of_Re_R is None or self.__p_of_Re_P is None:
            self.__p_of_Re_R, self.__p_of_Re_P = self.__nudepsaw.get_end_to_end_distribution(
                nu=nu, prefactor=prefactor
            )

        return self.__p_of_Re_R, self.__p_of_Re_P

    ##########################  AFRC PACKAGE FUNCTIONS  ##########################

    def get_afrc_end_to_end_distribution(self, recompute=False):
        """
        Function that returns the predicted end-to-end distance distribution
        based on the Analytical Flory Random Coil (AFRC). For more information
        see https://github.com/idptools/afrc

        Parameters
        ----------
        recompute : bool, optional
            If True, recompute the distribution even if it has already been
            calculated. Default is False.

        Returns
        -------
        tuple of arrays
            A 2-pair tuple of numpy arrays where the first is the distance (in
            Angstroms) and the second is the probability of that distance.
        """
        selector = "afrc-re-dist"
        if self.__afrc is None:
            from afrc import AnalyticalFRC

            self.__afrc = AnalyticalFRC(self.__protein.sequence, self.__p_of_r_resolution)

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__afrc.get_end_to_end_distribution()

        return self.__precomputed[selector]

    def get_afrc_radius_of_gyration_distribution(self, recompute=False):
        """
        Function that returns the predicted radius of gyration distribution
        based on the Analytical Flory Random Coil (AFRC). For more information
        see https://github.com/idptools/afrc

        Internally uses:

        Equation 3 from "A simple model for polymeric fractals in a good solvent
        and an improved version of the Flory approximation" by
        Daniel Lhuillier, J. Phys. France 49 (1988) 705-710.

        Parameters
        ----------
        recompute : bool, optional
            If True, recompute the distribution even if it has already been
            calculated. Default is False.

        Returns
        -------
        tuple of arrays
            A 2-pair tuple of numpy arrays where the first is the radius of
            gyration (in Angstroms) and the second is the probability of that
            value.
        """
        selector = "afrc-rg-dist"
        if self.__afrc is None:
            from afrc import AnalyticalFRC

            self.__afrc = AnalyticalFRC(self.__protein.sequence, self.__p_of_r_resolution)

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__afrc.get_radius_of_gyration_distribution()

        return self.__precomputed[selector]

    def get_mean_afrc_end_to_end_distance(self, mode="scaling law", recompute=False):
        """
        Function that returns the predicted mean end-to-end distance
        based on the Analytical Flory Random Coil (AFRC). For more information
        see https://github.com/idptools/afrc

        Parameters
        ----------
        mode : str, optional
            defines the mode in which the average is calculated, and can be
            set to either 'scaling law' (default) or 'distribution'. If 'distribution' is used
            then the complete Re distribution is used to calculate the expected value. If the
            'scaling law' is used then the standard Re = R0 * N^{0.5} is used,
            by default "scaling law"

        recompute : bool, optional
            If True, recompute the mean end-to-end distance even if a value for
            this ``mode`` has already been calculated. Default is False.

        Returns
        -------
        float
            Value equal to the average end-to-end distance (as defined by ``mode``).
        """

        # NB: mode is part of the cache key - results computed with one mode must
        # never be returned for a request made with another
        selector = f"afrc-mean-re-{mode}"
        if self.__afrc is None:
            from afrc import AnalyticalFRC

            self.__afrc = AnalyticalFRC(self.__protein.sequence, self.__p_of_r_resolution)

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__afrc.get_mean_end_to_end_distance(calculation_mode=mode)

        return self.__precomputed[selector]

    def get_mean_afrc_radius_of_gyration(self, mode="distribution", recompute=False):
        """
        Function that returns the predicted mean radius of gyration
        based on the Analytical Flory Random Coil (AFRC). For more
        information see https://github.com/idptools/afrc

        Parameters
        ----------
        mode : str, optional
            Defines the mode in which the average is calculated, and can be set
            to either 'distribution' (default) or 'scaling law'. If
            'distribution' is used then the complete Rg distribution is used to
            calculate the expected value. If 'scaling law' is used then the
            standard Rg = R0 * N^{0.5} is used.

        recompute : bool, optional
            If True, recompute the mean radius of gyration even if a value for
            this ``mode`` has already been calculated. Default is False.

        Returns
        -------
        float
            Value equal to the average radius of gyration (as defined by ``mode``).
        """
        # NB: mode is part of the cache key - see get_mean_afrc_end_to_end_distance
        selector = f"afrc-mean-rg-{mode}"
        if self.__afrc is None:
            from afrc import AnalyticalFRC

            self.__afrc = AnalyticalFRC(self.__protein.sequence, self.__p_of_r_resolution)

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__afrc.get_mean_radius_of_gyration(calculation_mode=mode)

        return self.__precomputed[selector]

    def get_afrc_internal_scaling(self, mode="scaling law", recompute=False):
        """
        Returns the internal scaling profile - a [2 by n] matrix that reports on the average
        distance between all residues that are n positions apart ( where n  is | i - j | ).

        Distances are in angstroms and are measured from the residue center of mass.

        A linear log-log fit of this data gives a gradient of 0.5 (:math:`\\nu^{app} = 0.5`).

        Parameters
        ----------
        mode : str, optional
            Mode used to compute the internal scaling profile, either
            'scaling law' (default) or 'distribution'.
        recompute : bool, optional
            If True, recompute the internal scaling profile even if a value for
            this ``mode`` has already been calculated. Default is False.

        Returns
        -------
        np.ndarray
           An [2 x n] matrix (where n = length of the amino acid sequence). The first column
           is the set of | i-j | distances, and the second defines the average inter-residue
           distances between every pair of residues that are | i-j | residues apart in sequnce
           space.

        """
        # NB: mode is part of the cache key - see get_mean_afrc_end_to_end_distance
        selector = f"afrc-internal-scaling-{mode}"
        if self.__afrc is None:
            from afrc import AnalyticalFRC

            self.__afrc = AnalyticalFRC(self.__protein.sequence, self.__p_of_r_resolution)

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__afrc.get_internal_scaling(calculation_mode=mode)

        return self.__precomputed[selector]

    def get_saw_end_to_end_distribution(self, prefactor=5.5, recompute=False):
        """
        Defines the end-to-end distribution based on the SAW as defined by:

        O’Brien, E. P., Morrison, G., Brooks, B. R., & Thirumalai, D. (2009).
        How accurate are polymer models in the analysis of Forster resonance
        energy transfer experiments on proteins? The Journal of Chemical Physics,
        130(12), 124903.

        This is a composition independent model for which the end-to-end distance depends
        solely on the number of amino acids. It is included here as an additional reference
        model.

        Parameters
        ------------
        prefactor : float
            Prefactor is a number that tunes the SAW dimensions. 5 angstroms is in the right ballpark
            but this number should be tuned to match EV sims, by default 5.5 A.

        recompute : bool, optional
            If True, recompute the distribution even if a value for this
            ``prefactor`` has already been calculated. Default is False.

        Returns
        -------
        tuple of arrays
           A 2-pair tuple of numpy arrays where the first is the distance (in Angstroms) and
           the second array is the probability of that distance.
        """

        # NB: prefactor is part of the cache key - results computed with one
        # prefactor must never be returned for a request made with another
        selector = f"saw-re-dist-{prefactor}"
        if self.__saw is None:
            from afrc.polymer_models.saw import SAW

            self.__saw = SAW(self.__protein.sequence, self.__p_of_r_resolution)

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__saw.get_end_to_end_distribution(prefactor)

        return self.__precomputed[selector]

    # not implemented
    # def get_saw_radius_of_gyration_distribution(self, recompute=False):
    #     selector = "saw-rg-dist"
    #     if self.__saw is None:
    #         from afrc.polymer_models.saw import SAW
    #         self.__saw = SAW(self.__protein.sequence, self.__p_of_r_resolution)

    #     if selector not in self.__precomputed or recompute is True:
    #         self.__precomputed[selector] = self.__afrc.get_radius_of_gyration_distribution()

    #     return self.__precomputed[selector]

    def get_mean_saw_end_to_end_distance(self, prefactor=5.5, recompute=False):
        """
        Returns the mean end-to-end distance (:math:`R_e`). As calculated
        from the SAW model as defined https://aip.scitation.org/doi/10.1063/1.3082151.

        By default this uses a prefactor of 5.5 A (0.55 nanometers).

        Parameters
        ------------
        prefactor : float
            Prefactor is a number that tunes the SAW dimensions. 5 angstroms is in the right ballpark
            but this number should be tuned to match EV sims, by default 5.5 A.

        recompute : bool, optional
            If True, recompute the mean end-to-end distance even if a value for
            this ``prefactor`` has already been calculated. Default is False.

        Returns
        -------
        float
           Returns the value equal to the mean end-to-end distance.

        """

        # NB: prefactor is part of the cache key - see get_saw_end_to_end_distribution
        selector = f"saw-mean-re-{prefactor}"
        if self.__saw is None:
            from afrc.polymer_models.saw import SAW

            self.__saw = SAW(self.__protein.sequence, self.__p_of_r_resolution)

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__saw.get_mean_end_to_end_distance(prefactor)

        return self.__precomputed[selector]

    def get_mean_saw_radius_of_gyration(self, prefactor=5.5, recompute=False):
        """
        Returns the mean radius of gyration (:math:`R_g`). As calculated
        from the SAW model as defined https://aip.scitation.org/doi/10.1063/1.3082151.

        By default this uses a prefactor of 5.5 A (0.55 nanometers).

        Parameters
        ------------
        prefactor : float
            Prefactor is a number that tunes the SAW dimensions. 5 angstroms is in the right ballpark
            but this number should be tuned to match EV sims, by default 5.5 A.

        recompute : bool, optional
            If True, recompute the mean radius of gyration even if a value for
            this ``prefactor`` has already been calculated. Default is False.

        Returns
        -------
        float
           Value equal to the mean radius of gyration.

        """
        # NB: prefactor is part of the cache key - see get_saw_end_to_end_distribution
        selector = f"saw-mean-rg-{prefactor}"
        if self.__saw is None:
            from afrc.polymer_models.saw import SAW

            self.__saw = SAW(self.__protein.sequence, self.__p_of_r_resolution)

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__saw.get_mean_radius_of_gyration(prefactor)

        return self.__precomputed[selector]

    def get_nudep_saw_end_to_end_distribution(self, nu=0.5, prefactor=5.5, recompute=False):
        """
        Defines the end-to-end distribution based on the nu-dependent SAW model.

        This is a composition independent model for which the end-to-end distance depends
        solely on the number of amino acids. Both nu and the prefactor can be varied

        Parameters
        ------------
        nu : float
            Flory scaling exponent. Should fall between 0.33 and 0.6

        prefactor : float
            Prefactor is a number that tunes the SAW dimensions. Default is 5.5 A.

        recompute : bool, optional
            If True, recompute the distribution even if a value for this ``nu``
            and ``prefactor`` combination has already been calculated.
            Default is False.

        Returns
        -------
        tuple of arrays
           A 2-pair tuple of numpy arrays where the first is the distance (in Angstroms) and
           the second array is the probability of that distance.

        """
        # NB: nu and prefactor are part of the cache key - results computed with
        # one set of parameters must never be returned for a different set
        selector = f"nudep-saw-re-dist-{nu}-{prefactor}"
        if self.__nudepsaw is None:
            from afrc.polymer_models.nudep_saw import NuDepSAW

            self.__nudepsaw = NuDepSAW(self.__protein.sequence, self.__p_of_r_resolution)

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__nudepsaw.get_end_to_end_distribution(nu=nu, prefactor=prefactor)

        return self.__precomputed[selector]

    def get_mean_nudep_saw_end_to_end_distance(self, nu=0.5, prefactor=5.5, recompute=False):
        """
        Returns the mean end-to-end distance (:math:`R_e`). As calculated
        from the nu-dependent SAW model.

        Parameters
        ------------
        nu : float
            Flory scaling exponent. Should fall between 0.33 and 0.6

        prefactor : float
            Prefactor is a number that tunes the SAW dimensions, by default 5.5 A.

        recompute : bool, optional
            If True, recompute the mean end-to-end distance even if a value for
            this ``nu`` and ``prefactor`` combination has already been
            calculated. Default is False.

        Returns
        -------
        float
           Returns the value equal to the mean end-to-end distance.
        """
        # NB: nu and prefactor are part of the cache key - see
        # get_nudep_saw_end_to_end_distribution
        selector = f"nudep-saw-mean-re-{nu}-{prefactor}"
        if self.__nudepsaw is None:
            from afrc.polymer_models.nudep_saw import NuDepSAW

            self.__nudepsaw = NuDepSAW(self.__protein.sequence, self.__p_of_r_resolution)

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__nudepsaw.get_mean_end_to_end_distance(nu=nu, prefactor=prefactor)

        return self.__precomputed[selector]

    def get_mean_nudep_saw_radius_of_gyration(self, nu=0.5, prefactor=5.5, recompute=False):
        """
        Returns the mean radius of gyration (:math:`R_g`). As calculated
        from the nu-dependent SAW model.

        Parameters
        ------------
        nu : float
            Flory scaling exponent. Should fall between 0.33 and 0.6

        prefactor : float
            Prefactor is a number that tunes the SAW dimensions, by default 5.5 A.

        recompute : bool, optional
            If True, recompute the mean radius of gyration even if a value for
            this ``nu`` and ``prefactor`` combination has already been
            calculated. Default is False.

        Returns
        -------
        float
           Returns the value equal to the mean radius of gyration.
        """
        # NB: nu and prefactor are part of the cache key - see
        # get_nudep_saw_end_to_end_distribution
        selector = f"nudep-saw-mean-rg-{nu}-{prefactor}"
        if self.__nudepsaw is None:
            from afrc.polymer_models.nudep_saw import NuDepSAW

            self.__nudepsaw = NuDepSAW(self.__protein.sequence, self.__p_of_r_resolution)

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__nudepsaw.get_mean_radius_of_gyration(nu=nu, prefactor=prefactor)

        return self.__precomputed[selector]

    #################  EMPIRICAL FUNCTIONS FROM PAPERS BELOW HERE  #################
    def empirical_nu(self, mode="zheng2020"):
        """Computes the scaling exponent (nu) for the given sequence as parameterized by the
        equation from zheng2020 via the Sequence Hydropathy Decoration (SHD)
        and Sequence Charge Decoration (SCD)

            Nu = -0.0423*SHD + 0.0074*SCD+0.701

        This equation for predicting nu is adopeted from Zheng et al. [1].

        Parameters
        ----------
        mode : str, optional
            Selector for the empirical parameterization to use. The only
            currently-implemented option is ``'zheng2020'`` (the default);
            passing anything else raises an exception.

        Returns
        -------
        float
            Returns the predict scalinge exponent (nu), a dimensionless
            parameter which should fall between 0.33 and 0.6 (in theory).

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If ``mode`` is not a recognised parameterization.

        References
        ---------------
        [1] Zheng, W., Dignon, G. L., Brown, M., Kim, Y. C. & Mittal, J.
        Hydropathy Patterning Complements Charge Patterning to Describe
        Conformational Preferences of Disordered Proteins. J. Phys.
        Chem. Lett. (2020). doi:10.1021/acs.jpclett.0c00288

        """
        validate_keyword_option(mode, ["zheng2020"], "mode")
        return scaling_parameters.compute_nu_zheng2020(self.__protein.sequence)

    def empirical_radius_of_gyration(self, mode="zheng2020"):
        """Function that takes in an amino acid sequence and computes the expected
        radius of gyration using the nu-dependent Rg as developed by Zheng et al.

        Parameters
        ----------
        mode : str, optional
            Selector for the empirical parameterization to use. The only
            currently-implemented option is ``'zheng2020'`` (the default);
            passing anything else raises an exception.

        Returns
        ------------------
        float
            Returns the empirically predicted radius of gyration in Angstorms.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If ``mode`` is not a recognised parameterization.

        References
        ---------------
        [1] Zheng, W., Dignon, G. L., Brown, M., Kim, Y. C. & Mittal, J.
        Hydropathy Patterning Complements Charge Patterning to Describe
        Conformational Preferences of Disordered Proteins. J. Phys.
        Chem. Lett. (2020). doi:10.1021/acs.jpclett.0c00288
        """
        validate_keyword_option(mode, ["zheng2020"], "mode")
        return scaling_parameters.compute_rg_zheng2020(self.__protein.sequence)
