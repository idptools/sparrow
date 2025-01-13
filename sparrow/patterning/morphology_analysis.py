'''
This is the morphology analysis based on:
Kania M, Rix H, Fereniec M, Zavala-Fernandez H, Janusek D, Mroczka T, Stix G, Maniewski R. The effect of precordial lead displacement on ECG morphology. Med Biol Eng Comput. 2014 Feb;52(2):109-19. doi: 10.1007/s11517-013-1115-9. Epub 2013 Oct 19. PMID: 24142562; PMCID: PMC3899452.

The code was written by Nick Razo
'''

#imports for the morphology analysis
import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicSpline
from scipy.stats import shapiro
from matplotlib import pyplot as plt


'''
Classes
'''


class MorphologyAnalysis:
    '''
    This class is designed to perform a morphology analysis based on:
    Kania M, Rix H, Fereniec M, Zavala-Fernandez H, Janusek D, Mroczka T, Stix G, Maniewski R. The effect of precordial lead displacement on ECG morphology. Med Biol Eng Comput. 2014 Feb;52(2):109-19. doi: 10.1007/s11517-013-1115-9. Epub 2013 Oct 19. PMID: 24142562; PMCID: PMC3899452.

    The object mostly stores values from throughout the analysis as there are many values that can be leveraged.
    The following values are stored:

    reference_signal #post zero adjust
    test_signal #post zero adjust
    number_comparison_pts

    self.__frac_reference_signal = None
    self.__frac_test_signal = None
    self.__comparison_pts = None
    self.__reference_affine_pts = None
    self.__test_affine_pts = None
    self.__normalized_reference_affine_pts = None
    self.__normalized_test_affine_pts = None
    self.__fit_slope = None
    self.__fit_offset = None
    self.__fit_residuals = None
    self.__fit_rmse = None
    self.__norm_fit_slope = None
    self.__norm_fit_offset = None
    self.__norm_fit_residuals = None
    self.__norm_fit_rmse = None
    self.__perf_fit_residuals = None
    self.__perf_fit_rmse = None
    self.__linear_fit_function = None
    self.__norm_linear_fit_function = None

    '''
    def __init__(self, test_sig : np.ndarray, ref_sig : np.ndarray, number_comparison_points = 50,
                 zero_adjustment = True, interp_method : str ='Linear',
                 **kwargs):
        '''Compute all values that might be needed from the regression based morphology analysis.
        Upon initialization this object computes all comparizon properties of the system provided.
        
        Keywords include: 'identifier', 'z_adj_frac'', 'identifier_val'
        identifier : This will typically be a string of text to explain the comparison this object is making
        identifier_val : This should be a numeric value to pull so you can create an ordinate in a plot
        z_adj_frac : This is the total error in your integral if you choose to zero adjust with the default function(default is 0.0000001).

        Parameters
        ----------
        sig1 : numpy.ndarray
            This is the signal that will get compared against the reference
        ref_sig : numpy.ndarray
            This is the reference signal
        zero_adjustment : bool
            This boolean deliniates whether or not to perform an adjustment on the signals zeroing.
        interp_method : str
            This is the method you wish to use to interpolate (note some may cause instability)
            Linear, CubicSpline, Akima

        Returns
        -------
        MorphologyAnalysis
            This object allows your quantify the dissimilarity between two signals
        '''
        #check if an identifier was passed to this object
        self.identifier = kwargs.get('identifier', None)
        self.identifier_val = kwargs.get('identifier_val', None)

        #save the signals that are being compared
        if zero_adjustment:
            self.z_adj_frac = kwargs.get('z_adj_frac', 0.0000001)
            self.reference_signal = zero_adj(ref_sig, self.z_adj_frac)
            self.test_signal = zero_adj(test_sig, self.z_adj_frac)
        else:
            self.reference_signal = ref_sig
            self.test_signal = test_sig

        #save the number of comparison points
        self.number_comparison_points = number_comparison_points
        self.interp_method = interp_method

        #define all the compute on demand variables
        self.__frac_reference_signal = None
        self.__frac_test_signal = None
        self.__comparison_pts = None
        self.__reference_affine_pts = None
        self.__test_affine_pts = None
        self.__normalized_reference_affine_pts = None
        self.__normalized_test_affine_pts = None
        self.__fit_slope = None
        self.__fit_offset = None
        self.__fit_residuals = None
        self.__fit_rmse = None
        self.__norm_fit_slope = None
        self.__norm_fit_offset = None
        self.__norm_fit_residuals = None
        self.__norm_fit_rmse = None
        self.__perf_fit_residuals = None
        self.__perf_fit_rmse = None
        self.__linear_fit_function = None
        self.__norm_linear_fit_function = None


    '''
    Define the properties that are useful for the class
    '''
    @property
    def frac_reference_signal(self) -> np.ndarray:
        '''
        Returns the cumulative sum at each point in the reference divided by the total sum.
        This is the fraction of total area under the reference at each point.

        Returns
        -------
        numpy.ndarray
            The cumulative sum at each point in the reference divided by the total sum
        '''
        #check if the frac_reference_signal exists
        if self.__frac_reference_signal is None:
            self.__frac_reference_signal = convert_vec2fractional_vec(self.reference_signal)
        
        #return the value
        return self.__frac_reference_signal
    
    @property
    def frac_test_signal(self) -> np.ndarray:
        '''
        Returns the cumulative sum at each point in the signal divided by the total sum.
        This is the fraction of total area under the signal at each point.

        Returns
        -------
        numpy.ndarray
            The cumulative sum at each point in the signal divided by the total sum
        '''
        #check if the frac_test_signal exists
        if self.__frac_test_signal is None:
            self.__frac_test_signal = convert_vec2fractional_vec(self.test_signal)

        #return the value
        return self.__frac_test_signal
    
    @property
    def comparison_pts(self) -> np.ndarray:
        '''
        Returns the fractional area points that are used for comparison in the morphology analysis.

        Returns
        -------
        numpy.ndarray
            The fractional area points that are used for comparison in the morphology analysis
        '''
        #check if the comparison points already exist
        if self.__comparison_pts is None:
            comparison_pts_p = np.linspace(0,1, self.number_comparison_points+2)
            self.__comparison_pts = comparison_pts_p[1:self.number_comparison_points+1]

        #return the value back
        return self.__comparison_pts
    
    @property
    def reference_affine_pts(self) -> np.ndarray:
        '''
        Returns the reference interpolated index positions that correspond to the comparison_pts.

        Returns
        -------
        numpy.ndarray
            The reference interpolated index positions that correspond to the comparison_pts
        '''
        #check if the value has already been computed
        if self.__reference_affine_pts is None:
            self.__reference_affine_pts = find_interp_idx_given_y(self.frac_reference_signal, self.comparison_pts, method= self.interp_method)
        
        #check for NaNs in the reference affine points
        ref_nan = np.isnan(self.__reference_affine_pts)
        if np.any(ref_nan):
            ref_nan_idxs = [k for k, val in enumerate(ref_nan) if val]
            raise Exception(f"NaN's were computed in the interpolation for the reference signal.\nNaN indexes: {ref_nan_idxs}")

        #return the value
        return self.__reference_affine_pts
    
    @property 
    def test_affine_pts(self) -> np.ndarray:
        '''
        Returns the signal interpolated index positions that correspond to the comparison_pts.

        Returns
        -------
        numpy.ndarray
            The signal interpolated index positions that correspond to the comparison_pts
        '''
        #check if the value has already been computed
        if self.__test_affine_pts is None:
            self.__test_affine_pts = find_interp_idx_given_y(self.frac_test_signal, self.comparison_pts, self.interp_method)

        #check for NaNs in the test affine points
        test_nan = np.isnan(self.__test_affine_pts)
        if np.any(test_nan):
            test_nan_idxs = [k for k, val in enumerate(test_nan) if val]
            raise Exception(f"NaN's were computed in the interpolation for the reference signal.\nNaN indexes: {test_nan_idxs}")
        
        #return the value
        return self.__test_affine_pts
    
    @property 
    def normalized_reference_affine_pts(self) -> np.ndarray:
        '''
        Returns the reference interpolated normalized index positions that correspond to the comparison_pts.

        Returns
        -------
        numpy.ndarray
            The reference interpolated normalized index positions that correspond to the comparison_pts
        '''
        if self.__normalized_reference_affine_pts is None:
            self.__normalized_reference_affine_pts = self.reference_affine_pts / (len(self.reference_signal)-1)

        return self.__normalized_reference_affine_pts
    
    @property 
    def normalized_test_affine_pts(self) -> np.ndarray:
        '''
        Returns the signal interpolated normalized index positions that correspond to the comparison_pts.
        
        Returns
        -------
        numpy.ndarray
            The signal interpolated normalized index positions that correspond to the comparison_pts
        '''
        if self.__normalized_test_affine_pts is None:
            self.__normalized_test_affine_pts = self.test_affine_pts / (len(self.test_signal)-1)

        return self.__normalized_test_affine_pts
    

    @property 
    def linear_fit_function(self):
        '''
        Returns the linear fitting function associated with unnormalized test and reference signals.

        Returns
        -------
        function
            This is the linear function that was used to fit the unnormalized test and reference signals.
            A numpy.ndarray of x value can be passed to get a sense of the fit.
        '''
        if self.__linear_fit_function is None:
            self.__linear_fit_function = lambda x : self.fit_slope*x + self.fit_offset

        return self.__linear_fit_function

    

    @property 
    def norm_linear_fit_function(self):
        '''
        Returns the linear fitting function associated with unnormalized test and reference signals.

        Returns
        -------
        function
            This is the linear function that was used to fit the unnormalized test and reference signals.
            A numpy.ndarray of x value can be passed to get a sense of the fit.
        '''
        if self.__norm_linear_fit_function is None:
            self.__norm_linear_fit_function = lambda x : self.norm_fit_slope*x + self.norm_fit_offset

        return self.__norm_linear_fit_function

    @property 
    def fit_slope(self) -> float:
        '''
        Returns the slope of the line fit to an unnormalized test and reference signals

        Returns
        -------
        float
            The slope of the line fit to an unnormalized test and reference signals
        '''
        if self.__fit_slope is None:
            tup = self._comp_linear_properties(self.reference_affine_pts, self.test_affine_pts)
            self.__fit_slope = tup[0]
            self.__fit_offset = tup[1] #this is done to avoid updates in more than one location
        
        return self.__fit_slope

    @property
    def fit_offset(self) -> float:
        '''
        Returns the offset value for the line that fits the unnormalized test and reference signals

        Returns
        -------
            The offset value for the line that fits the unnormalized test and reference signals
        '''
        if self.__fit_offset is None:
            a = self.fit_slope #this is called to avoid updates in more than one location

        return self.__fit_offset


    @property
    def fit_residuals(self) -> np.ndarray:
        '''
        Returns the residuals from the linear fit of the unnormalized data

        Returns
        -------
        numpy.ndarray
            The residuals from the linear fit fo the unnormalized data
        '''
        if self.__fit_residuals is None:
            self.__fit_residuals = calc_error(self.reference_affine_pts, self.test_affine_pts, self.linear_fit_function)
        
        return self.__fit_residuals


    @property 
    def fit_rmse(self) -> float:
        '''
        Returns the root mean square of the residuals 

        Returns
        -------
        float
            The root mean square of the residuals for the 
        '''
        if self.__fit_rmse is None:
            self.__fit_rmse = calc_rmse(self.fit_residuals)

        return self.__fit_rmse
    
    @property    
    def norm_fit_slope(self) -> float:
        '''
        Returns the slope of the line fit to an normalized test and reference signals

        Returns
        -------
        float
            The slope of the line fit to an normalized test and reference signals
        '''
        if self.__norm_fit_slope is None:
            tup = self._comp_linear_properties(self.normalized_reference_affine_pts, self.normalized_test_affine_pts)
            self.__norm_fit_slope = tup[0]
            self.__norm_fit_offset = tup[1] #this is done to avoid updates in more than one location
        
        return self.__norm_fit_slope

    @property
    def norm_fit_offset(self) -> float:
        '''
        Returns the offset value for the line that fits the normalized test and reference signals

        Returns
        -------
            The offset value for the line that fits the normalized test and reference signals
        '''
        if self.__norm_fit_offset is None:
            a = self.norm_fit_slope #this is called to avoid updates in more than one location

        return self.__norm_fit_offset
    

    @property
    def norm_fit_residuals(self) -> np.ndarray:
        '''
        Returns the residuals from the linear fit of the normalized data

        Returns
        -------
        numpy.ndarray
            The residuals from the linear fit fo the normalized data
        '''
        if self.__norm_fit_residuals is None:
            self.__norm_fit_residuals = calc_error(self.normalized_reference_affine_pts, self.normalized_test_affine_pts, self.norm_linear_fit_function)
        
        return self.__norm_fit_residuals


    @property 
    def norm_fit_rmse(self) -> float:
        '''
        Returns the root mean square of the normalized residuals 

        Returns
        -------
        float
            The root mean square of the normalized residuals
        '''
        if self.__norm_fit_rmse is None:
            self.__norm_fit_rmse = calc_rmse(self.norm_fit_residuals)

        return self.__norm_fit_rmse


    @property
    def perf_fit_residuals(self) -> np.ndarray:
        '''
        Returns the symetric residuals for the normalized data assuming the line of best fit is y=x

        Returns
        -------
        numpy.ndarray
            The residuals for the perfect fit line on the normalized data. These are the symmetric residuals.
        '''
        if self.__perf_fit_residuals is None:
            self.__perf_fit_residuals = (self.normalized_test_affine_pts-self.normalized_reference_affine_pts)/np.sqrt(2)

        return self.__perf_fit_residuals


    @property
    def perf_fit_rmse(self) -> float:
        '''
        The RMSE value for the perfect fit residuals (symmetric)
        
        Returns
        -------
        float
            The RMSE for the perfect fit residuals.
        '''
        if self.__perf_fit_rmse is None:
            self.__perf_fit_rmse = calc_rmse(self.perf_fit_residuals)

        return self.__perf_fit_rmse





    '''
    Functions for this classes use only
    '''
    def _comp_linear_properties(self, ref_pts : np.ndarray, test_pts : np.ndarray) -> tuple:
        '''This function fits a line to the (reference_values, test_values) pairs
        
        Parameters
        ----------
        ref_pts : numpy.ndarray
            These are the reference points to correlate to the same fractinal area swept as in the test_pts
        test_pts : numpy.ndarray
            These are the the t4est points that correlat4e to the same factional area swept in the ref_pts
        Returns
        -------
        tuple
            This returns the slope and the offset of the linear fit (slope, offset)
        '''
        linear_fit_coef = np.polyfit(ref_pts, test_pts, 1)
        return linear_fit_coef[0], linear_fit_coef[1]
    


    
    '''
    Useful getters
    '''

    def get_affine_pairs(self, mode : str = 'regular') -> tuple:
        '''Returns the values that are paired in the affine projection ready for scatter plotting.
        
        Parameters
        ----------
        mode : str
            The mode you wish to get the pairs for: 'regular', 'normalized', 'perfect alignment'
        
        Returns
        -------
        tuple
            reference points in the affine space, test points in the affine space
        '''
        if mode == 'regular':
            ref_af_pts = self.reference_affine_pts
            test_af_pts = self.test_affine_pts
        elif mode == 'normalized' or mode == 'perfect alignment':
            ref_af_pts = self.normalized_reference_affine_pts
            test_af_pts = self.normalized_test_affine_pts
        else:
            raise Exception(f"{mode} is not a valid mode.")
        return ref_af_pts, test_af_pts
    
    def get_affine_fit_pairs(self, mode : str = 'regular') -> tuple:
        '''Returns a series of points that correspond to the affine projection linear fit
        
        Parameters
        ----------
        mode : str
            The mode you wish to get the fitted pairs for: 'regular', 'normalized', 'perfect alignment'
        
        Returns
        -------
        tuple
            reference points in the fitted affine space, test points in the fitted affine space
        '''
        if mode == 'regular':
            ref_af_pts = self.reference_affine_pts
            lin_pts = self.linear_fit_function(ref_af_pts)
        elif mode == 'normalized':
            ref_af_pts = self.normalized_reference_affine_pts
            lin_pts = self.norm_linear_fit_function(ref_af_pts)
        elif mode == 'perfect alignment':
            ref_af_pts = self.normalized_reference_affine_pts
            lin_pts = ref_af_pts
        else:
            raise Exception(f"{mode} is not a valid mode.")
        return ref_af_pts, lin_pts
    
    def get_error_projection(self, mode : str = 'regular') -> tuple:
        '''Returns the error projected on the comparison signal. Note: this returns an (x,y) for the error only
        
        Parameters
        ----------
        mode : str
            The mode you wish to get the error projection for: 'regular', 'normalized', 'perfect alignment'
        
        Returns
        -------
        tuple
            test points in affine space, error for those points
        '''
        if mode == 'regular':
            test_af_pts = self.test_affine_pts
            error_pts = self.fit_residuals
        elif mode == 'normalized':
            test_af_pts = self.normalized_test_affine_pts
            error_pts = self.__norm_fit_residuals
        elif mode == 'perfect alignment':
            test_af_pts = self.normalized_test_affine_pts
            error_pts = self.__perf_fit_residuals
        else:
            raise Exception(f"{mode} is not a valid mode.")
        return test_af_pts, error_pts
    
    def get_original_data(self) -> tuple:
        '''Gets the original data that was used for the morphology analysis
        This data is post any processing you allow in the object initialization.
        
        Returns
        -------
        tuple
            test signal, reference signal
        '''
        return self.test_signal, self.reference_signal
    
    def get_expansion_factor(self) -> float:
        '''Returns the factor that the test signal is expanded or compressed by assuming it is morphologically identical to the reference.
        
        Returns
        -------
        float
            This is the expansion factor or the slope of the unnormalized linear fit
        '''
        return self.fit_slope
    
    def get_dissimilarity_factor(self, mode : str = 'regular', trim = None) -> float:
        '''Returns the factor that quantifies the morphologic dissimilar between the test and reference signalusing an RMSE.
        
        Parameters
        ----------
        mode : str
            The mode you wish to get the dissimilarity factor: 'regular', 'normalized', 'perfect alignment'
        trim : float
            This is the percentile of points that you wish to trim from the residuals before performing the summation.
            Ex: 90 means keep everything below the 90th percentile
        Returns
        -------
        float
            This is the RMSE of the trimmed residual
        '''
        #compute the residuals of interest
        res = None
        if mode == 'regular':
            res =  self.fit_residuals
        elif mode == 'normalized':
            res = self.norm_fit_residuals
        elif mode == 'perfect alignment':
            res = self.perf_fit_residuals
        else:
            raise Exception(f"{mode} is not a valid mode.")
        
        #remove any values if needed
        if trim is None:
            return calc_rmse(res)
        else:
            #get the percentile value for trimming
            percentile_val = self.get_percentile_score(trim, mode)
            #get all the values below the percentile value
            isBelowThresh = np.abs(res) < percentile_val
            res_trim = res[isBelowThresh]
            #compute the RMSE
            return calc_rmse(res_trim)
        
    def get_summed_dissimilarity_factor(self, mode : str = 'regular', trim = None) -> float:
        '''Returns the factor that quantifies the morphologic dissimilar between the test and reference signalusing a sum.
        
        This value will have dependance on the number of points you used in the interpolation.

        Parameters
        ----------
        mode : str
            The mode you wish to get the dissimilarity factor: 'regular', 'normalized', 'perfect alignment'
        trim : float
            This is the percentile of points that you wish to trim from the residuals before performing the summation.
            Ex: 90 means keep everything below the 90th percentile
        Returns
        -------
        float
            This is the sum of the absolute value of the trimmed residuals.
        '''
        #compute the residuals of interest
        res = None
        if mode == 'regular':
            res =  self.fit_residuals
        elif mode == 'normalized':
            res = self.norm_fit_residuals
        elif mode == 'perfect alignment':
            res = self.perf_fit_residuals
        else:
            raise Exception(f"{mode} is not a valid mode.")
        
        #remove any values if needed
        if trim is None:
            return np.mean(np.abs(res))/len(res)
        else: #this is the section for if a trim value was passed
            #get the percentile value for trimming
            percentile_val = self.get_percentile_score(trim, mode)
            #get all the values below the percentile value
            isBelowThresh = np.abs(res) < percentile_val
            res_trim = res[isBelowThresh]
            #compute the 
            return np.mean(np.abs(res_trim))

    def get_morphology_significance(self, mode : str = 'regular') -> float:
        '''Returns the p value for the morphological similarity

        Parameters
        ----------
        mode : str
            This is the mode you wish to get the significance for: 'regular', 'normalized', 'perfect alignment'
        
        Returns
        -------
        float
            The p-value for the shapiro wilkes test on the residuals choosen by the 'mode' parameter
        '''
        p_val = None
        if mode == 'regular':
            a,p_val = shapiro(self.fit_residuals)
        elif mode == 'normalized':
            a,p_val = shapiro(self.norm_fit_residuals)
        elif mode == 'perfect alignment':
            a,p_val = shapiro(self.perf_fit_residuals)
        else:
            raise Exception(f"{mode} is not a valid mode.")

        return p_val
    
    def get_percentile_score(self, percent : float = 90, mode : str = 'regular') -> float:
        '''Returns the percentile of the absolute value of the residuals for the selected mode.
        
        Parameters
        ----------
        percent : float
            This is the percent for the distribution that you want returned
        mode : str
            This is the mode you wish to get the significance for: 'regular', 'normalized', 'perfect alignment'
        
        Returns
        -------
        float
            The residual value that is larger that X percent of other residues where X is the 'percent' parameter
            passed to this function
        '''
        #check if the percentile is a meaningful value
        if percent < 0  or percent > 100:
            raise Exception(f"Percentile must be between 0 and 100. The percentile {percent} was passed.")
        if mode == 'regular':
            dist = np.abs(self.fit_residuals)
        elif mode == 'normalized':
            dist = np.abs(self.norm_fit_residuals)
        elif mode == 'perfect alignment':
            dist = np.abs(self.perf_fit_residuals)
        else:
            raise Exception(f"{mode} is not a valid mode.")
        
        return np.percentile(dist, percent)




    '''
    Useful plotting functions
    '''

    def plot_original_data(self, mode = 'regular') -> tuple:
        '''Plots a figure of the original data that was used in the comparison.'''
        # Create the figure and axes
        fig, ax = plt.subplots()
        
        # Generate the plot
        if mode == 'regular':
            x1 = np.arange(len(self.test_signal))
            x2 = np.arange(len(self.reference_signal))
        elif mode == 'normalized' or mode == 'perfect alignment':
            x1 = np.arange(len(self.test_signal))/len(self.test_signal)
            x2 = np.arange(len(self.reference_signal))/len(self.reference_signal)
        ax.plot(x1, self.test_signal, label="Test Signal")
        ax.plot(x2, self.reference_signal, label="Reference Signal")
        ax.set_title("Original Data")
        ax.set_xlabel("Index")
        ax.set_ylabel("Input Data")
        ax.legend()
    
        # Return the figure and axes objects for further modification
        return fig, ax
    
    def plot_fractional_sweep(self, mode = 'regular') -> tuple:
        '''Plots a figure of the original data that was used in the comparison.'''
        # Create the figure and axes
        fig, ax = plt.subplots()
        
        # Generate the plot
        if mode == 'regular':
            x1 = np.arange(len(self.test_signal))
            x2 = np.arange(len(self.reference_signal))
        elif mode == 'normalized' or mode == 'perfect alignment':
            x1 = np.arange(len(self.test_signal))/len(self.test_signal)
            x2 = np.arange(len(self.reference_signal))/len(self.reference_signal)
        ax.plot(x1, self.frac_test_signal, label="Test Signal") #plot the test signals fractional area curve
        ax.plot(x2, self.frac_reference_signal, label="Reference Signal") #plot the 
        ax.set_title("Fractional Sweep Plot")
        ax.set_xlabel("Index")
        ax.set_ylabel("Fraction of Area")
        ax.legend()
    
        # Return the figure and axes objects for further modification
        return fig, ax
    
    def plot_affine_pairs(self, mode = 'regular') -> tuple:
        '''Plots the affine pairs that are the same area after interpolation'''
        #Create the figure and aes
        fig, ax = plt.subplots()

        #generate the plot for the affine corrds
        ax.scatter(*self.get_affine_pairs(mode))
        ax.plot(*self.get_affine_fit_pairs(mode), color='red')
        ax.set_xlabel(f"Reference Index Pair")
        ax.set_ylabel(f"Text Index Pair")
        ax.set_title(f"Point Pairs")

        #test to add about the stats
        if mode == 'regular':
            rmse = self.fit_rmse
            slope = self.fit_slope
            offset = self.fit_offset
        elif mode == 'normalized':
            rmse = self.norm_fit_rmse
            slope = self.norm_fit_slope
            offset = self.norm_fit_offset
        elif mode == 'perfect alignment':
            rmse = self.perf_fit_rmse
            slope = 1
            offset = 0
        txt_str = f"RMSE: {rmse}\nY={slope}x+{offset}"
        ax.text(ax.get_xlim()[0], ax.get_ylim()[1],txt_str, ha="left", va="top")

        #return the figure and axes objects fo rfurther modification
        return fig, ax
    

    def plot_error_and_signal(self, mode = 'regular') -> tuple:
        '''Plots the error and the test signal curves on the same plot'''
        fig, ax = plt.subplots()

        #generate the plot for the error and the test signals
        if mode == 'regular':
            x1 = np.arange(len(self.test_signal))
        elif mode == 'normalized' or mode == 'perfect alignment':
            x1 = np.arange(len(self.test_signal))/len(self.test_signal)
        ax.scatter(*self.get_error_projection(mode), color="red", label="Fit Error")
        ax.plot(x1, self.test_signal, label="Test Signal")
        ax.set_xlabel(f"Signal Index")
        ax.set_ylabel(f"Value")
        ax.set_title(f"Test Signal and Fit Error")
        ax.legend()

        return fig, ax
    



class SignedMorphologyAnalysis:
    '''This is a sort of wrapper class on the MorphologyAnalysis class.
    This is designed to create combined metrics on both the attractive and repulsive MorphologyAnalysis object.
    It also houses a full signal analysis and a sign flipped full signal analysis object as well.
    '''
    def __init__(self, test_sig : np.ndarray, ref_sig : np.ndarray, number_comparison_points = 50,
                 zero_adjustment = True, interp_method : str ='Linear',
                 **kwargs):
        '''This is a wrapper on the MorphologyAnalysis object designed create combined morphology metrics'''
        #Produce the morphology objects for both 
        self.attractive_morphology, self.repulsive_morphology = get_attractive_repulsive_morph(test_sig, ref_sig,
                                                                                               number_comparison_points, zero_adjustment,
                                                                                               **kwargs)
        
        #get teh full signal and sign flipped signal analysis objects
        self.full_sig_morphology = MorphologyAnalysis(test_sig,ref_sig,number_comparison_points,
                                                      zero_adjustment, interp_method, **kwargs)
        self.flipped_sig_morphology = MorphologyAnalysis(-1*test_sig,-1*ref_sig,number_comparison_points,
                                                      zero_adjustment, interp_method, **kwargs)
        
    '''Combined Metrics'''
    def get_combined_percentile_score(self, percentile : float = 90, mode = 'perfect alignment',
                                      aset = 'signed') -> float:
        '''This computes the percentile score that contains both attactive and repulsive information
        The percentile parameter is a combined metric of both the attractive and repulsive portions'''
        #get the percentile metrics
        if aset == 'signed':
            at = self.attractive_morphology.get_percentile_score(percentile, mode)
            rep = self.repulsive_morphology.get_percentile_score(percentile, mode)
        elif aset == 'full':
            at = self.flipped_sig_morphology.get_percentile_score(percentile, mode)
            rep = self.full_sig_morphology.get_percentile_score(percentile, mode)

        #delta calculation
        delta = np.sqrt(np.power(at,2) + np.power(rep,2))

        return delta
    
    def get_combined_dissimilarity_score(self, trim = None, mode = 'perfect alignment',
                                         aset = 'signed') -> float:
        '''This computes the dissimilarity score that contains both attactive and repulsive information
        The dissimilarity parameter is a combined metric of both the attractive and repulsive portions'''
        #get the percentile metrics
        if aset == 'signed':
            at = self.attractive_morphology.get_dissimilarity_factor(mode, trim)
            rep = self.repulsive_morphology.get_dissimilarity_factor(mode, trim)
        elif aset == 'full':
            at = self.flipped_sig_morphology.get_dissimilarity_factor(mode, trim)
            rep = self.full_sig_morphology.get_dissimilarity_factor(mode, trim)

        #delta calculation
        delta = np.sqrt(np.power(at,2) + np.power(rep,2))

        return delta


'''
Misc. Functions
'''
def get_attractive_repulsive_morph(sig : np.ndarray, ref_sig : np.ndarray,
                                    number_comparison_pts = 50, zero_adjustment = True,
                                    **kwargs) -> tuple:
    '''This function takes in a signal, splits it into positive and negative portions, and compares it against the reference signals provided.
    
    Parameters
    ----------
    sig : numpy.ndarray
        This is the signal you wish to test both the positive and negative portions of 
    ref_sig : numpy.ndarray
        This is the signal you want to use as your comparison against
    
    Returns
    -------
    tuple
        This contains the attractive mophology analysis and repulsive morphology analysis 
        Attractive morphology analysis, Repulsive morphology analysis
    '''
    #split the signal into positive and negative portions
    sig_attrac,sig_rep = split_attractive_repulsive(sig)
    #split the refernce into positive and negative portions
    ref_attrac,ref_rep = split_attractive_repulsive(ref_sig)

    #return the Morphology analysis objects
    attrac =  MorphologyAnalysis(sig_attrac, ref_attrac, number_comparison_pts, zero_adjustment, **kwargs)
    rep =  MorphologyAnalysis(sig_rep, ref_rep, number_comparison_pts, zero_adjustment, **kwargs)
    return attrac, rep


def split_attractive_repulsive(sig : np.ndarray) -> tuple:
    '''This function splits the attactive and repulsive components from a epsilon signal
    This function applies a mask over the positive and negative regions of the signal
    to split the data into positive and negative regions. It also multiplies by -1
    for the attractive portion to make attractive be positive.
    
    Parameters
    ----------
    sig : numpy.ndarray
        This is the signal you want to split

    Returns
    -------
    tuple
        numpy.ndarray (attractive portion of signal), numpy.ndarray (repulsive signal)
    '''
    repulsive = (sig>0)*sig
    attractive = -1*sig*(sig < 0)

    return attractive, repulsive

def average_eps_signals(sig_list : list, num_interp_pts : int = 50) -> np.ndarray:
    '''This function averages a list of signals
    Averages a list of signals to create a baseline for comparison

    Parameters
    ----------
    sig_list : list
        This is a list of numpy arrays that you want to create an average signal from
    Returns
    -------
    numpy.ndarray
        This is the normalized signal average
    '''
    #get an akima interpolator object for each signal
    interp_sig_list = []
    for sig in sig_list:
        #create an interpolation object and normalize the residues
        interpolation_function = Akima1DInterpolator(np.linspace(0,1,len(sig)), sig)
        #get the interpolated values of interest
        interp_sig_list.append(interpolation_function(np.linspace(0,1,num_interp_pts)))

    #create a numpy array to work with
    interp_mat = np.array(interp_sig_list)

    #compute the mean value
    avg_sig = np.mean(interp_mat, axis=0)

    return avg_sig
    


def convert_vec2fractional_vec(vec : np.ndarray) -> np.ndarray:
    '''Converts a vector (1D array) into a fraction of the total sum at each point
    The result is computed as the cummulative sum/total sum of the vector
    Parameters
    ----------
    vec : numpy.ndarray
        This is the vector to convert to the fractional vector
    
    Returns
    -------
    numpy.ndarray
        The cummulative sum of vec divided by the total sum of vec
    '''
    return np.cumsum(vec) / np.sum(vec)

def find_interp_idx_given_y(y_values : np.ndarray, y_of_interest : np.ndarray,
                            method = 'Linear') -> np.ndarray:
    '''Takes in the y values for increasing function and y values you wish to interpolate on and returns the fractional indexes where these occur.
    The y values that are passed must be increasing (this is stronger than monotonicity). 

    Ex: numpy.array([1,2,3,4,5]) is increasing
    numpy.array([1,2,2,3,4]) is not increasing but is monotonic

    It utilizes scipy's 1D akima interpolator to perform the interpolation.
    Parameters
    ----------
    y_values : numpy.ndarray
        This is the vector that contains the increasing values.
    y_of_interest : numpy.ndarray
        This is the series of y values you want to interpolate on and return fractional indexes for
    
    Returns
    -------
    numpy.ndarray
        The cummulative sum of vec divided by the total sum of vec
    '''
    #Define x values as indices of y_values
    x_values = np.arange(len(y_values))
    diff_larger = np.diff(y_values) > 0
    if not np.all(diff_larger):
        idx_not_increasing = [k for k,val in enumerate(diff_larger) if val == 0]
        snippet = []
        for ii in idx_not_increasing:
            #get the start and stop of the snippet of interest
            i_sa = ii - 3
            i_sp = ii + 3
            if i_sa < 0:
                i_sa = 0
            if i_sp > len(y_values):
                i_sp = len(y_values)
            snippet.append(y_values[i_sa:i_sp])
            
        raise Exception(f"The y values are not always increasing.\n{snippet}")
    
    #Create an interpolation function for the flipped ordinates
    if method == 'Linear':
        interpolation_function = np.interp(y_of_interest, y_values, x_values)
        return interpolation_function
    else:
        if method == 'CubicSpline':
            interpolation_function = CubicSpline(y_values, x_values)
        elif method == 'Akima':
            interpolation_function = Akima1DInterpolator(y_values, x_values)
        else:
            raise Exception(f"The method specified for interpolation was not a valid one: {method}")
        
        #Return the interpolated x value for the given y_of_interest
        return interpolation_function(y_of_interest)

def calc_error(x_data : np.ndarray, y_data : np.ndarray, fit_func) -> np.ndarray:
    '''Calculates the root mean square error of a data with respect to a fitting funtion.
    Error = difference between the fit and y_data values at a point
    Parameters
    ----------
    x_data : np.ndarray
        This is the x data for the RMSE calculation
    y_data : np.ndarray
        This is the y data for the RMSE
    fit_func : function
        This is a function that allows for the fitting of the data (it does)
    Returns
    -------
    np.ndarray
        This is the error of the data with respect to a reference fitted model.
        y_i - Y_fit(i)
    '''
    #compute out the fitted functions values at the specified x values
    fit_y_vals = fit_func(x_data)

    #compute the difference between the fitted data and the 
    return fit_y_vals - y_data


def zero_adj(dat : np.ndarray, adj_p : float = 0.0000001) -> np.ndarray:
    '''This function adjusts the offset of the input data to ensure all values are > 0.
    This function takes in a signal and changes the offset to ensure that 
    the data is always > 0. The adj_p value defines the total fraction of the current
    area that will be added to the integral by making this adjustment. This is a relative
    error term in that regards and puts a cap on the error of the output signals from the orginal.

    Parameters
    ----------
    dat : numpy.ndarray
        this is the data that you want to offset to ensure it is is always positive.
    adj_p : float
        This is the cap on the error as a fraction of the error that the original function sweeps out.
    Returns
    -------
    np.ndarray
        This is the adjusted input data whos min will be just above 0.
    '''
    #check if the data is all 0's first
    if np.all(np.isclose(dat, 0, atol=0.001)):
        #print an alert to the user
        print(f"Warning: data passed to zero_adj is all zeros within tolerance. Returning all ones.")
        #create the ones vector
        new_dat = np.ones(len(dat))
    else:
        #adjust the zero location
        new_dat = dat - np.min(dat)
        #get the total sum of the data
        tot_sum = np.sum(new_dat)

        #find the number of 0s in order to ensure a fraction of error
        zeros_vec = [k for k,val in enumerate(new_dat) if np.isclose(val,0,atol = 0.001)]
        num_zeros = len(zeros_vec)
        
        #set the values to be just above zero
        set_val = adj_p * tot_sum / num_zeros

        #set the zero values
        for k in zeros_vec:
            new_dat[k] = set_val

    return new_dat

def calc_rmse(error_vec : np.ndarray) -> float:
    '''Computes the RMSE of the fit given the error vector'''
    return np.sqrt(np.mean(np.power(error_vec,2)))

def calc_projection(a : np.ndarray, b : np.ndarray) -> float:
    '''Computes the projection of a onto b'''
    #the dot product of a and b divided by the norm of b
    return np.dot(a,b) / np.sqrt(np.sum(np.power(b,2)))