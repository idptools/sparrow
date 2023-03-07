import numpy as np

# from afrc.polymer_models.saw import SAW
# from afrc.polymer_models.wlc import WormLikeChain
# from afrc.polymer_models.wlc2 import WormLikeChain2

from . import scaling_parameters


class Polymeric:
    """Base class providing programmatic access to various empirical equations and 
    machine learned predictions for various different polymer properties for the sequence.

    Note that, of course, many of these would only be valid if the sequence
    behaved as an intrinsically disordered or unfolded polypeptide. 
    """
    def __init__(self, protein_obj, p_of_r_resolution=0.05):
        """Init method for the Polymeric class that takes in a sparrow.Protein object
        to provide programmatic access to various machine learned predictors and 

        Parameters
        ----------
        protein_obj : sparrow.Protein
            Composition-based interface into the functionalities from the sparrow.Protein class.
            Primarily used to give access to PARROT trained predictor objects.
        
        p_of_r_resolution : float
            Bin width for building the probability distributions, defined in Angstroms, by default 0.05.
        """
        self.__protein = protein_obj
        self.__NuDepSAW = None
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

        Returns
        -------
        np.ndarray 
            2D numpy array in which the first column is the distance (in angstroms) 
            and the second column is the probablity.
        """
        if self.__NuDepSAW is None:
            from afrc.polymer_models.nudep_saw import NuDepSAW
            self.__NuDepSAW = NuDepSAW(self.__protein.sequence, self.__p_of_r_resolution)
        
        prefactor = self.predicted_prefactor 
        nu = self.predicted_nu

        # insert some sort of error handling on predicted prefactor / nu values
        # to ensure the predicted values are reasonable!

        if self.__p_of_Re_R is None or self.__p_of_Re_P is None:
            self.__p_of_Re_R, self.__p_of_Re_P = self.__NuDepSAW.get_end_to_end_distribution(nu=nu,prefactor=prefactor)

        return self.__p_of_Re_R, self.__p_of_Re_P

    def get_afrc_end_to_end_distribution(self, recompute=False):
        selector = "afrc-re-dist"
        if self.__afrc is None: 
            from afrc import AnalyticalFRC
            self.__afrc = AnalyticalFRC(self.__protein.sequence, self.__p_of_r_resolution)

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__afrc.end_to_end_distribution()

        return self.__precomputed[selector]
    
    def get_afrc_radius_of_gyration_distribution(self, recompute=False):
        selector = "afrc-rg-dist"
        if self.__afrc is None: 
            from afrc import AnalyticalFRC
            self.__afrc = AnalyticalFRC(self.__protein.sequence, self.__p_of_r_resolution)

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__afrc.get_radius_of_gyration_distribution()

        return self.__precomputed[selector]

    def get_mean_afrc_end_to_end_distance(self, recompute=False):
        selector = "afrc-mean-re"
        if self.__afrc is None: 
            from afrc import AnalyticalFRC
            self.__afrc = AnalyticalFRC(self.__protein.sequence, self.__p_of_r_resolution)

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__afrc.get_mean_end_to_end_distance()

        return self.__precomputed[selector]
    
    def get_mean_afrc_radius_of_gyration(self, recompute=False):
        selector = "afrc-mean-rg"
        if self.__afrc is None: 
            from afrc import AnalyticalFRC
            self.__afrc = AnalyticalFRC(self.__protein.sequence, self.__p_of_r_resolution)

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__afrc.get_mean_radius_of_gyration()

        return self.__precomputed[selector]
    
    def get_afrc_internal_scaling(self, recompute=False):
        selector = "afrc-internal-scaling"
        if self.__afrc is None: 
            from afrc import AnalyticalFRC
            self.__afrc = AnalyticalFRC(self.__protein.sequence, self.__p_of_r_resolution)

        if selector not in self.__precomputed or recompute is True:
            self.__precomputed[selector] = self.__afrc.get_internal_scaling()

        return self.__precomputed[selector]

    #################  EMPIRICAL FUNCTIONS FROM PAPERS BELOW HERE  #################
    def empirical_nu(self, mode='zheng2020'):
        """Computes the scaling exponent (nu) for the given sequence as parameterized by the
        equation from zheng2020 via the Sequence Hydropathy Decoration (SHD)
        and Sequence Charge Decoration (SCD)
    
            Nu = -0.0423*SHD + 0.0074*SCD+0.701
    
        This equation for predicting nu is adopeted from Zheng et al. [1].

        Returns
        -------
        float
            Returns the predict scalinge exponent (nu), a dimensionless 
            parameter which should fall between 0.33 and 0.6 (in theory).

        References
        ---------------
        [1] Zheng, W., Dignon, G. L., Brown, M., Kim, Y. C. & Mittal, J. 
        Hydropathy Patterning Complements Charge Patterning to Describe 
        Conformational Preferences of Disordered Proteins. J. Phys. 
        Chem. Lett. (2020). doi:10.1021/acs.jpclett.0c00288

        """
        return scaling_parameters.compute_nu_zheng2020(self.__protein.sequence)

    
    def empirical_radius_of_gyration(self, mode='zheng2020'):
        """Function that takes in an amino acid sequence and computes the expected 
        radius of gyration using the nu-dependent Rg as developed by Zheng et al.

        Returns
        ------------------
        float
            Returns the empirically predicted radius of gyration in Angstorms.

        References
        ---------------
        [1] Zheng, W., Dignon, G. L., Brown, M., Kim, Y. C. & Mittal, J. 
        Hydropathy Patterning Complements Charge Patterning to Describe 
        Conformational Preferences of Disordered Proteins. J. Phys. 
        Chem. Lett. (2020). doi:10.1021/acs.jpclett.0c00288
    """
        return scaling_parameters.compute_rg_zheng2020(self.__protein.sequence)



        
