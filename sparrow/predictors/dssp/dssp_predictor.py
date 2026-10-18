import numpy as np

from sparrow.predictors.base import BaseNetworkPredictor
from sparrow.sparrow_exceptions import SparrowException


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

1. A constructor (__init__()) which PRE LOADS the network from sparrow/data/networks/relevant_name - the sparrow.data_path() function
   is defined in sparrow/__init__.py and allows absolute-path access to the /data directory. The constructor should
   FULLY load the network along with standard PARROT-style options, as shown here. Trained networks should be versioned and 
   implemented so previous versions can be chosene even if the default version changes

2. Define a function called predict_<THING>(self, seq) where <THING> is a convenient name that obviously means this is
   what the function does.


"""

DEFAULT_VERSION = "2"


def softmax(v):
    return (np.e**v) / np.sum(np.e**v)


def classes_from_probability(probability, threshold=0.5, minlen=5):
    """Binary per-residue classes from the probability of one DSSP class.

    A residue whose probability is above ``threshold`` is in the class, but
    only within a contiguous run of such residues longer than ``minlen``;
    shorter runs are set to 0. The DSSP predictors classify helix, extended and
    coil this way, from probabilities predicted for one sequence or a batch.

    Parameters
    ----------
    probability : sequence of float
        The probability of the class at each residue.
    threshold : float, optional
        The probability a residue must exceed, between 0 and 1. Default 0.5.
    minlen : int, optional
        A run must be longer than this to count, at least 1. Default 5.

    Returns
    -------
    numpy.ndarray
        1 for each residue in the class, else 0.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``threshold`` is not between 0 and 1, or ``minlen`` is below 1.
    """
    if threshold < 0 or threshold > 1:
        raise SparrowException("Threshold must be between 0 and 1")
    if minlen < 1:
        raise SparrowException("Minlen must be 1 or greater")

    binary_helicity = [1 if x > threshold else 0 for x in probability]

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
                    return_list.extend([1] * counter)
                else:
                    return_list.extend([0] * counter)

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
            return_list.extend([1] * counter)
        else:
            return_list.extend([0] * counter)
    return np.array(return_list)


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

        # Forward pass  -this is specific for classication
        prediction = self._raw_predict(seq)
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

        return self.predict(seq)

    def _from_raw(self, seq, raw):
        """The probability of each DSSP class at each residue, from the network's raw output.

        See :meth:`~sparrow.predictors.base.BaseNetworkPredictor._from_raw`.

        Returns
        -------
        numpy.ndarray
            ``(len(seq), 3)``: the probability of helix (column 0), extended
            (1) and coil (2) at each residue.
        """
        prediction = raw.flatten().reshape(-1, self.number_of_classes)
        return np.array(list(map(softmax, prediction)))

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

        tmp = self.predict_helical_probability(seq)
        classes = classes_from_probability(tmp, threshold, minlen)

        if return_probability is True:
            return (classes, tmp)

        return classes

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

        tmp = self.predict_extended_probability(seq)
        classes = classes_from_probability(tmp, threshold, minlen)

        if return_probability is True:
            return (classes, tmp)

        return classes

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

        tmp = self.predict_coil_probability(seq)
        classes = classes_from_probability(tmp, threshold, minlen)

        if return_probability is True:
            return (classes, tmp)

        return classes

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
