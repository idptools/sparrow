import numpy as np


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
        
        # bin width parameter
        self.__p_of_r_resolution = p_of_r_resolution

        # gamma definition copy-pasta'd from AFRC 
        # set gamma - orginally defined in
        # Le Guillou, J. C., & Zinn-Justin, J. (1977). Critical Exponents for the n-Vector
        # Model in Three Dimensions from Field Theory. Physical Review Letters, 39(2), 95–98.
        # for the case of n=0 (polymer), and raised in this context in the Soranno form
        # of the Zheng et al nu-dependent polymer model (see eq 9b in Soranno, A. (2020).
        # Physical basis of the disorder-order transition. Archives of Biochemistry and
        # Biophysics, 685, 108305.
        self.gamma = 1.1615

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

    def get_predicted_end_to_end_distribution(self):
        """
        Function that returns the predicted end-to-end distance distribution 
        based on the predicted scaling exponent.

        Returns
        -------
        np.ndarray 
            2D numpy array in which the first column is the distance (in angstroms) 
            and the second column is the probablity.
        """

        # in AFRC they don't use memoization because nu and the prefactor
        # can change; however, I don't think (?) this is a problem here because 
        # we're working with a Polymeric model per sequence - check with Alex 
        # -JML (feb 2023)

        prefactor = self.predicted_prefactor 
        nu = self.predicted_nu

        # insert some sort of error handling on predicted prefactor / nu values
        # to ensure the predicted values are reasonable!

        if self.__p_of_Re_R is None:
            self.__compute_end_to_end_distribution(prefactor=prefactor, nu=nu)

        return (self.__p_of_Re_R, self.__p_of_Re_P)

    def __compute_end_to_end_distribution(self, nu, prefactor):
        """_summary_

        Parameters
        ----------
        nu : float
            flory apparent scaling exponent
        prefactor : float
            prefactor from flory.
        """
        gamma = self.gamma
        g = (gamma - 1) / nu
        delta = 1 / (1 - nu)
        A1 = self.__compute_A1(delta, g)
        A2 = self.__compute_A2(delta, g)
        
        # self.__p_of_Re_R = 
        # self.__p_of_Re_P = 

    def __compute_A1(self,delta, g):
        pass

    def __compute_A2(self,delta, g):
        pass
    
    def sample_predicted_end_to_end_distribution(self, dist_size=3000):
        """
        Function to randomly sample from the end-to-end distance distribution
        """
        if len(self.__protein) == 0:
            return np.repeat(0.0, dist_size)
        else:
            if self.__p_of_Re_R is None:
                self.__compute_end_to_end_distribution()
                
            return np.random.choice(self.__p_of_Re_R, dist_size, p=self.__p_of_Re_P)

    #################  EMPIRICAL FUNCTIONS FROM PAPERS BELOW HERE  #################
    def nu(self, mode='zheng2020'):
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

    
    def radius_of_gyration(self, mode='zheng2020'):
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



        
