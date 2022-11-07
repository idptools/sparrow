from parrot import brnn_architecture
from parrot import encode_sequence

import sparrow

import torch
import numpy as np
import os
from sparrow.sparrow_exceptions import SparrowException
from . import phospho_predictor_utils


"""
Predictor template file. This data file should, in principle, require 
minimal editing to convert into a specific predictor based on a copied
network file found in sparrow/data/networks/<predictor_name>. Some general
guidelines below (also included in the predictor documentation) and inline
comments on things you will want to change. This code WILL NOT RUN as is and
requires you to update missing variables to customize the predictor!!

Missing values will be enclosed in < > to indicate this is where you (the
software developer) must add some content


## Nomenclature

1. The predictor file should be called <relevant_name>_predictor.py
2. This should be inside a module in the /predictor/ directory called <relevant_thing>
3. The single class this module implements should be called <RevevantthingPredictor>


## Class structure

The class should have (at least) two functions:

1. A constructor (__init__()) which PRE LOADS the network from sparrow/data/networks/relevant_name - the get_data() function
   is defined in sparrow/__init__.py and allows absolute-path access to the /data directory. The constructor should
   FULLY load the network along with standard PARROT-style options, as shown here. Trained networks should be versioned and 
   implemented so previous versions can be chosene even if the default version changes

2. Define a function called predict_<THING>(self, seq) where <THING> is a convenient name that obviously means this is
   what the function does.

The idea is that this class should actually be completely stand alone independent of sparrow - i.e. one should be able to run

    >> from sparrow.predictor.<predictor_name> import <RelevantName>Predictor
    >>
    >> P = <RelevantName>Predictor()
    >> P.predict_<SOMETHING_RELEVANT>('myvalidseqnce')

And have it work!
    




"""

# NOTE - this is where you can define the version number that is read by default. If you add a new network MAKE SURE you update
# this default if you want that new network to be used by default
DEFAULT_VERSION="1"

def softmax(v):
    return (np.e ** v) / np.sum(np.e ** v)



## CHANGE class name
class TyrPhosphorylationPredictor():
    """

    Class that loads in a network such that predict_tyr_phosphorylation() can be called to predict
    tyrsosine phosphorylation from a sequence.

    """
    def __init__(self, version=None):
        """
        Constructor for building a predictor object object. The version keyword allows specific
        version(s) of the trained network associated with the predictor to be defined. 

        By default, it's set to None, which leads to the current best/default network being selected
        and is MOSTLY going to be the right option. However, to preserve backwards compatibility we provide
        the ability to pass a string as version. This string is inserted at position <X> in the filename

            <RelevantName>_network_v<X>.pt

        i.e. no need to include the "v" part or the .pt extension

        """

        

        # if no version provided use default, then grab path and check that file actually exists! 
        if version is None:
            version = DEFAULT_VERSION

        # CHANGE THIS!! Make sure oyu change the <DIRECTORY_NAME> and <PREDICTOR_NAME> to the appropriate
        # paths. Keep the network_v{version}.pt because this is how a version-specific string is selected
        saved_weights = sparrow.get_data(f'networks/phosphorylation/tyr_phosphorylation_predictor_network_v{version}.pt')

        if not os.path.isfile(saved_weights):
            raise SparrowException('Error: could not find saved weights file [%s] for %s predictor' %( saved_weights, type(self).__name__))
            
        
        # assuming the file is there, we next read in the parameeter file. Note that we force this to be CPU mainly because
        # we know everyone has a CPU... 
        loaded_model = torch.load(saved_weights, map_location=torch.device('cpu'))

        ## DELETE ME PROBABLY
        # this block of code is relevant ONLY if the trained network has this straneg
        # appended 'module.' text at the start of every keyword. This may happen in older
        # version of PARROT (see DSSP predictor as an example of where its needed) but in
        # 2022 trained networks didn't need this. As such, this can PROBABLY be deleted but
        # in case you're using an older network we've kept this to make things simple
        
        #for i in range(len(loaded_model)):
        #    key, value = loaded_model.popitem(last=False)
        #    new_key = key[7:]
        #    loaded_model[new_key] = value
        ## END OF DELETE ME PROBABLY

      
        # Dynamically calculate the hyperparameters used to train the network. 
        ## NOTE:
        # 
        # The code here works on networks trained using the current version of PARROT (2022), HOWEVER, it's possible
        # that in prevoius versions the keys into the parameter file may be different or may have a prefix. Best example
        # of this is that for the DSSP predictor the world `module.` randomly appears in from of each keyword. If you
        # look at dssp/dssp_predictor.py you can see at this point in the code there's a re-assignment to remove this
        # keyword. 

        # When PARROT runs it's predictions it REQUIRES the keywords in the parameter file to match the expected keywords
        # in PARROT, so it's imperative that these keywords are right. If you run into weird issues here feel free to 
        # reach out to Alex or Dan about this!

        num_layers = 0
        while True:
            s = f'lstm.weight_ih_l{num_layers}'
            try:
                temp = loaded_model[s]
                num_layers += 1
            except KeyError:
                break
                        
        number_of_classes = np.shape(loaded_model['fc.bias'])[0]

        # Hard coded because we always use one-hot encoding, note that if you trained a specific
        # predictor on a different encoding scheme you could, of course, here simply define that
        # encoding scheme 
        input_size = 20 

        hidden_vector_size = int(np.shape(loaded_model['lstm.weight_ih_l0'])[0] / 4)
        
        # set these here so we can sanity check if needed
        self.number_of_classes = number_of_classes
        self.input_size = input_size
        self.number_of_layers = num_layers
        self.hidden_vector_size = hidden_vector_size

        # Instantiate network weights into object
        self.network = brnn_architecture.BRNN_MtM(input_size, hidden_vector_size, num_layers, number_of_classes, 'cpu')
                                                 
        # load parameters into model
        self.network.load_state_dict(loaded_model)



    def predict_tyr_phosphorylation(self, seq, raw_values=False, return_sites_only=False, windowsize=4, threshold=0.6):
        """
        Function to predict the presence of phospho-tyrosine residues. By default returns a 
        binary mask of the sequence as a list, where predicted phosphosites are set to 1
        and other residues are set to 0.

        Parameters
        ------------
        seq : str
            Valid amino acid sequence

        raw_values : bool
            Flag which, if set to true, means the raw per-residue prediction values (between
            0 and 1) are returned. Default = False. 

        return_sites_only : bool
            Flag which, if set to true, means a list of phosphosite positions in numerical
            order are returned

        windowsize : int
            Define the size of the window this algorithm uses to extend the
            influenc of a local phosphosite probability. Note the windowsize
            gets applied +/- a central position
        
        threshold : float
            A threshold value used to deliniate between phosphosites for masking.
            Default is 0.6.


        Returns
        ----------
        np.ndarray
            Returns a 1D np.ndarray the length of the sequence where each position
            is the predicted value <CHANGE THIS>

        """

        # convert sequence to uppercase
        seq = seq.upper()

        # Convert to one-hot sequence vector - note, as mentioned above if you 
        # did't use one-hot in the original training you could just edit this here        
        seq_vector = encode_sequence.one_hot(seq)
        seq_vector = seq_vector.view(1, len(seq_vector), -1)  # formatting


        # run prediction and apply softmax
        prediction = self.network(seq_vector.float()).detach().numpy().flatten()
        prediction = prediction.reshape(-1, self.number_of_classes)
        prediction = np.array(list(map(softmax, prediction)))

        # finally we extract out local probabilities
        phospho_probability = []
        for val in prediction:
            phospho_probability.append(round(val[1],5))


        # if we want the raw model prediction values just return here
        if raw_values:
            return np.array(phospho_probability)
        
        else:
            if return_sites_only:
                return phospho_predictor_utils.return_hits(seq, phospho_probability, 'Y', windowsize=windowsize, threshold=threshold, return_sites_only=True)
            else:
                return np.array(phospho_predictor_utils.return_hits(seq, phospho_probability, 'Y', windowsize=windowsize, threshold=threshold))

                    
                    

                    
                    

                                        
                                        
                
                
            
            
