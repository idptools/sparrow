from parrot import brnn_architecture
from parrot import encode_sequence

import sparrow

import torch
import numpy as np
import os
from sparrow.sparrow_exceptions import SparrowException




DEFAULT_VERSION="1"


class MitochondrialTargetingPredictor():
    """

    Class that loads in a network such that predict_mitochondrial_targeting() can be called to predict
    mitochondrial targeting for a sequence.

    """
    def __init__(self, version=None):
        """
        Constructor for building a MitochondrialTargetingPredictor object. The version keyword allows specific
        version(s) of the trained network associated with the underlying predictor to be defined. 
        By default, it's set to None, which leads to the current best/default network being selected
        and is MOSTLY going to be the right option. However, to preserve backwards compatibility we provide
        the ability to pass a string as version. This string is inserted at position <X> in the filename
        
            mitochondrial_targeting_predictor_network_v{version}.pt

        i.e. no need to include the "v" part or the .pt extension

        """

        if version is None:
            version = DEFAULT_VERSION

        saved_weights = sparrow.get_data(f'networks/mitochondrial_targeting/mitochondrial_targeting_predictor_network_v{version}.pt')

        if not os.path.isfile(saved_weights):
            raise SparrowException('Error: could not find saved weights file [%s] for %s predictor' %( saved_weights, type(self).__name__))
            
        
        loaded_model = torch.load(saved_weights, map_location=torch.device('cpu'))

      
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



    def predict_mitochondrial_targeting(self, seq):
        """
        Prediction function. seq should be a valid amino acid sequence.

        NOTE that this assumes mitochondrial targeting sequences (MTSs) are 
        N-terminal, so truncates anything over 168 residues. This threshold
        was empyrically determined based on the set of annottated MTSs.

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

        # truncate all but 168 - if shorter than this just gets everything
        sub_seq = seq[0:168]

        # Convert to one-hot sequence vector
        seq_vector = encode_sequence.one_hot(sub_seq)
        seq_vector = seq_vector.view(1, len(seq_vector), -1)  # formatting

        # Forward pass  -this is specific for classication
        prediction = self.network(seq_vector.float()).detach().numpy()
        int_vals = []
        for row in prediction[0]:
            int_vals.append(np.argmax(row))

        prediction = int_vals

        # append empty 0s for remainder of sequence
        extra = [0]*(len(seq)-len(sub_seq))

        prediction.extend(extra)
        # return prediction + extra zeros
        return prediction
