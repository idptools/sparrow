from parrot import brnn_architecture
from parrot import encode_sequence

import sparrow

import torch
import numpy as np
import os
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

1. A constructor (__init__()) which PRE LOADS the network from sparrow/data/networks/relevant_name - the get_data() function
   is defined in sparrow/__init__.py and allows absolute-path access to the /data directory. The constructor should
   FULLY load the network along with standard PARROT-style options, as shown here. Trained networks should be versioned and 
   implemented so previous versions can be chosene even if the default version changes

2. Define a function called predict_<THING>(self, seq) where <THING> is a convenient name that obviously means this is
   what the function does.


"""

DEFAULT_VERSION="1"


class DSSPPredictor():
    """

    Class that loads in a network such that predict_transmebrane_regions() can be called to predict
    transmembrane regions in a sequence.

    """
    def __init__(self, version=None):
        """
        Constructor for building a TransmembranePredictor object. The version keyword allows specific
        version(s) of the trained network associated with the HelicityPredictor to be defined. 
        By default, it's set to None, which leads to the current best/default network being selected
        and is MOSTLY going to be the right option. However, to preserve backwards compatibility we provide
        the ability to pass a string as version. This string is inserted at position <X> in the filename

            HelicityPredictor_network_v<X>.pt

        i.e. no need to include the "v" part or the .pt extension

        """

        if version is None:
            version = DEFAULT_VERSION

        saved_weights = sparrow.get_data(f'networks/dssp/dssp_predictor_network_v{version}.pt')

        if not os.path.isfile(saved_weights):
            raise SparrowException('Error: could not find saved weights file [%s] for %s predictor' %( saved_weights, type(self).__name__))
            
        
        loaded_model = torch.load(saved_weights, map_location=torch.device('cpu'))

        # this removes this weird appending of 'module.' to the front of the keyes
        # which I honestly don't know where came from, but this fixes the issue.
        # 
        for i in range(len(loaded_model)):
            key, value = loaded_model.popitem(last=False)
            new_key = key[7:]
            loaded_model[new_key] = value
      
        # Dynamically read in correct hyperparameters:
        num_layers = 0
        while True:
            s = f'lstm.weight_ih_l{num_layers}'
            try:
                temp = loaded_model[s]
                num_layers += 1
            except KeyError:
                break
                     

        ##  determine the number of classes; note you may need to change the key names here no leading
        # module. in ther
        number_of_classes = np.shape(loaded_model['fc.bias'])[0]
        input_size = 20 # (hardcoded at 20 for 20 amino acids)

        hidden_vector_size = int(np.shape(loaded_model['lstm.weight_ih_l0'])[0] / 4)

        
        # set these here so we can sanity check if needed
        self.number_of_classes = number_of_classes
        self.input_size = input_size
        self.number_of_layers = num_layers
        self.hidden_vector_size = hidden_vector_size

        # Instantiate network weights into object
        self.network = brnn_architecture.BRNN_MtM(input_size, hidden_vector_size, num_layers, number_of_classes, 'cpu')
                                                  
        self.network.load_state_dict(loaded_model)



    def predict_dssp(self, seq):
        """
        Prediction function. seq should be a valid amino acid sequence.

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

        prediction = int_vals


        # for regression use the line below instead  - included here so this
        # file can be easily copied over for future predictors
        # prediction = self.network(seq_vector.float()).detach().numpy().flatten()
        # prediction = np.clip(prediction, 0.0, 1.0)

        return prediction
