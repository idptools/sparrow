import numpy as np


from . import scaling_parameters


class Polymeric:
    """Base class providing programmatic access to various empirical equations and 
    machine learned predictions for various different polymer properties for the sequence.

    Note that, of course, many of these would only be valid if the sequence
    behaved as an intrinsically disordered or unfolded polypeptide. 
    """
    def __init__(self, protein_obj):
        """Init method for the Polymeric class that takes in a sparrow.Protein object
        to provide programmatic access to various machine learned predictors and 

        Parameters
        ----------
        protein_obj : sparrow.Protein
            
        """
        self.__protein = protein_obj
        self.__predicted_scaling_exp = None
        self.__predicted_average_rg = None
        self.__predicted_average_re = None
        self.__predicted_average_asphercity = None
        self.__predicted_prefactor = None
        
        # Adapted from code in AFRC package
        # set distribution info to false, lazily computed as needed
        # storing both 1) the distance distribution and 2) the probability distribution
        self.__p_of_Re_R = False
        self.__p_of_Re_P = False

        # lacking closed form analytical solution here - careful!
        # set distribution info to false, lazily computed as needed
        # storing both 1) the distance distribution and 2) the probability distribution
        self.__p_of_Rg_R = False
        self.__p_of_Rg_P = False

    @property
    def predicted_nu(self):
        """This function returns the predicted scaling exponent (nu) for a given sequence
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
        """This function returns the predicted radius of gyration (rg) for a given sequence
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
        """This function returns the predicted end-to-end distance (re) for a given sequence
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
        """This function returns the predicted asphericity (δ) for a given sequence
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
        """This function returns the predicted prefactor ('A_0') for a given sequence
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

        # if we have not yet done it, computed the end-to-end distance distribution
        if self.__p_of_Re_R is False:
            self.__compute_end_to_end_distribution()

        return (self.__p_of_Re_R, self.__p_of_Re_P)

    def __compute_end_to_end_distribution(self):
        pass

    def sample_predicted_end_to_end_distribution(self, dist_size=3000):
        """
        Function to randomly sample from the end-to-end distance distribution
        """
        if len(self.__protein) == 0:
            return np.repeat(0.0, dist_size)
        else:
            if self.__p_of_Re_R is False:
                self.__compute_end_to_end_distribution()
                
            return np.random.choice(self.__p_of_Re_R, dist_size, p=self.__p_of_Re_P)


    def nu(self, mode='zheng2020'):
        """Computes the scaling exponent (nu) for the given sequence as parameterized by the
        equation from zheng2020 via the Sequence Hydropathy Decoration (SHD)
        and Sequence Charge Decoration (SCD)
    
            Nu = -0.0423×SHD + 0.0074×SCD+0.701
    
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



        
