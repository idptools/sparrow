import numpy as np
from parrot import encode_sequence

from sparrow.predictors.base import BaseNetworkPredictor



"""
Predictor of DSSP scores regions from sequence.

This is an example of how to implement a system-specific predictor 
in sparrow and could/should be used as a template for adding in 
additional predictors.


## Nomenclature

1. The predictor file should be called <relevant_name>_predictor.py
2. This should be inside a module in the /predictor/ directory called <relevant_thing>
3. The single class this module implements should be called <RevevantthingPredictor>

e.g. here we have 

1. dssp/
2. dssp_predictor.py
3. DSSPPredictor


## Class structure

The class should have (at least) two functions:

1. A constructor (__init__()) which PRE LOADS the network from sparrow/data/networks/relevant_name - the get_data() function
   is defined in sparrow/__init__.py and allows absolute-path access to the /data directory. The constructor should
   FULLY load the network along with standard PARROT-style options, as shown here. Trained networks should be versioned and 
   implemented so previous versions can be chosene even if the default version changes

2. Define a function called predict_<THING>(self, seq) where <THING> is a convenient name that obviously means this is
   what the function does.


"""

DEFAULT_VERSION="2"

def softmax(v):
    return (np.e ** v) / np.sum(np.e ** v)



class DSSPPredictor(BaseNetworkPredictor):
    """

    Class that loads a network such that predict_dssp() and the related methods
    can be called to predict per-residue DSSP secondary structure.

    """

    NETWORK_PATH = "networks/dssp/dssp_predictor_network_v{version}.pt"
    DEFAULT_VERSION = DEFAULT_VERSION
    ARCHITECTURE = "MtM"


    # .................................................................................
    #
    #        
    def predict_dssp(self, seq):
        """
        Prediction function. seq should be a valid amino acid sequence. Returns an 
        np.ndarray where each element is the most likely class at that residue position.

        Parameters
        ------------
        seq : str
            Valid amino acid sequence

        Returns
        ----------
        np.ndarray
            Returns a 1D np.ndarray the length of the sequence where each position
            is the transient helicity at that position.

        """

        # convert sequence to uppercase
        seq = seq.upper()


        # Convert to one-hot sequence vector
        seq_vector = encode_sequence.one_hot(seq)
        seq_vector = seq_vector.view(1, len(seq_vector), -1)  # formatting


        # Forward pass  -this is specific for classication
        prediction = self.network(seq_vector.float()).detach().numpy()
        int_vals = []
        for row in prediction[0]:
            int_vals.append(np.argmax(row))

        prediction = np.array(int_vals)
        return prediction


    # .................................................................................
    #
    #
    def predict_dssp_probabilities(self, seq):
        """
        Prediction function. seq should be a valid amino acid sequence. Returns an 
        (seq_len x 3) np.ndarray where each row gives the probability of class 0 (helicity) 1 (beta) or 2 (coil) for each residue.

        Parameters
        ------------
        seq : str
            Valid amino acid sequence

        Returns
        ----------
        np.ndarray
            Returns a 1D np.ndarray the length of the sequence and width of 3 where
            each row reports on the probability of each of the possible 3 classes.

        """

        # convert sequence to uppercase
        seq = seq.upper()

        # Convert to one-hot sequence vector
        seq_vector = encode_sequence.one_hot(seq)
        seq_vector = seq_vector.view(1, len(seq_vector), -1)  # formatting

        prediction = self.network(seq_vector.float()).detach().numpy().flatten()
        prediction = prediction.reshape(-1, self.number_of_classes)
        prediction = np.array(list(map(softmax, prediction)))

        return prediction


    
    # .................................................................................
    #
    #    
    def predict_helicity_smart(self, seq, threshold=0.5, minlen=5, return_probability=False):
        """
        Function which takes an amino acid sequence and returns a binary classification under the assumption
        that true helices must be at least minlen residues long where the helicity is consistently above 
        threshold value.

        Parameters
        -------------
        seq : str
            Valid amino acid sequence

        threshold : float
            Threshold for helicity 

        Returns
        -------------
        np.ndarray
            Returns a numpy array with 1s for helical regions and 0s for non helical regions.
        

        """

        # defensive programming
        if threshold < 0 or threshold > 1:
            raise Exception('Threshold myst be between 0 and 1')

        # defensive programming
        if minlen < 1:
            raise Exception('Minlen must be 1 or greater')

        tmp = self.predict_helical_probability(seq)
        
        if return_probability is True:            
            return (self.__binarize_probabilities(tmp, threshold, minlen), tmp)

        else:
            return self.__binarize_probabilities(tmp, threshold, minlen)


    
    # .................................................................................
    #
    #
    def predict_extended_smart(self, seq, threshold=0.5, minlen=5, return_probability=False):
        """
        Function which takes an amino acid sequence and returns a binary classification under the assumption
        that true beta strands must be at least minlen residues long where the helicity is consistently above 
        threshold value.

        Parameters
        -------------
        seq : str
            Valid amino acid sequence

        threshold : float
            Threshold for beta/extended propensity 

        Returns
        -------------
        np.ndarray
            Returns a numpy array with 1s for extended regions and 0s for non-extended regions.
        

        """

        # defensive programming
        if threshold < 0 or threshold > 1:
            raise Exception('Threshold myst be between 0 and 1')

        # defensive programming
        if minlen < 1:
            raise Exception('Minlen must be 1 or greater')

        tmp = self.predict_extended_probability(seq)
        
        if return_probability is True:            
            return (self.__binarize_probabilities(tmp, threshold, minlen), tmp)

        else:
            return self.__binarize_probabilities(tmp, threshold, minlen)




    # .................................................................................
    #
    #    
    def predict_coil_smart(self, seq, threshold=0.5, minlen=1, return_probability=False):
        """
        Function which takes an amino acid sequence and returns a binary classification under the assumption
        that true coil regions must be at least minlen residues long where the coil is consistently above 
        threshold value.

        Parameters
        -------------
        seq : str
            Valid amino acid sequence

        threshold : float
            Threshold for coil propensity 

        Returns
        -------------
        np.ndarray
            Returns a numpy array with 1s for coil regions and 0s for non coil regions.
        

        """

        # defensive programming
        if threshold < 0 or threshold > 1:
            raise Exception('Threshold myst be between 0 and 1')

        # defensive programming
        if minlen < 1:
            raise Exception('Minlen must be 1 or greater')

        
        tmp = self.predict_coil_probability(seq)
        
        if return_probability is True:            
            return (self.__binarize_probabilities(tmp, threshold, minlen), tmp)

        else:
            return self.__binarize_probabilities(tmp, threshold, minlen)


    # .................................................................................
    #
    #
    def predict_helical_probability(self, seq):
        """
        Predict per-residue helical probabilities.

        Parameters
        -------------
        seq : str
            Valid amino acid sequence

        Returns
        -------------
        np.ndarray
            Returns a numpy array with where each value reports on the probability of
            the residue being in a helix or not.

        """

        probs = self.predict_dssp_probabilities(seq)
        return probs.transpose()[0]


    # .................................................................................
    #
    #
    def predict_extended_probability(self, seq):
        """
        Predict per-residue beta/extended probabilities.

        Parameters
        -------------
        seq : str
            Valid amino acid sequence

        Returns
        -------------
        np.ndarray
            Returns a numpy array with where each value reports on the probability of
            the residue being in an extended state or not.

        """
        probs = self.predict_dssp_probabilities(seq)
        return probs.transpose()[1]
    

    # .................................................................................
    #
    #    
    def predict_coil_probability(self, seq):
        """
        Predict per-residue coil probabilities.

        Parameters
        -------------
        seq : str
            Valid amino acid sequence

        Returns
        -------------
        np.ndarray
            Returns a numpy array with where each value reports on the probability of
            the residue being in an coil state or not.

        """        
        probs = self.predict_dssp_probabilities(seq)
        return probs.transpose()[2]


    # .................................................................................
    #
    #    
    def __binarize_probabilities(self, helical_probability, threshold = 0.5, minlen = 5):
        '''
        Internal function for binarizing probabilities with 
        Takes an array of probabilities over all residues in a sequence. Assigns
        a binary helical label to each residue with a probability above the 
        specificed threshold 'thresh'. Any residues above the threshold which
        are not in a contiguous stretch of residues at or about the length 
        threshold 'minlen' are set to 0.
        '''
        binary_helicity = [1 if x > threshold else 0 for x in helical_probability]


        if binary_helicity[0] == 1:
            inside = True
            counter = 0
        else:
            inside = False
            counter = 0
    
        return_list = []

        # for each binarized residue
        for i in binary_helicity:
        
            # if we're inside a region
            if inside:
            
                # if still inside...
                if i == 1:
                    counter = counter + 1
                    
                # if breaking the region
                else:
                    if counter > minlen:
                        return_list.extend([1]*counter)
                    else:
                        return_list.extend([0]*counter)

                    # also add a '0' for this element
                    return_list.append(0)

                    # reset counters for next region
                    inside = False
                    counter = 0
                
            # if we're outside a region
            else:
                if i == 1:
                    counter = counter + 1
                    inside = True
                else:
                    return_list.append(0)
        if inside:
            if counter > minlen:
                return_list.extend([1]*counter)
            else:
                return_list.extend([0]*counter)
        return np.array(return_list)
        
