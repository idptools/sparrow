import torch
import numpy as np
from typing import List
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import os

from parrot import brnn_architecture
from parrot import encode_sequence
import sparrow

from .asphericity.asphericity_predictor import AsphericityPredictor
from .re.end_to_end_distance_predictor import RePredictor
from .rg.radius_of_gyration_predictor import RgPredictor
from .prefactor.prefactor_predictor import PrefactorPredictor
from .scaling_exponent.scaling_exponent_predictor import ScalingExponentPredictor
from .scaled_re.scaled_end_to_end_distance_predictor import ScaledRePredictor
from .scaled_rg.scaled_radius_of_gyration_predictor import ScaledRgPredictor
from ..sparrow_exceptions import SparrowException

def prepare_model(network,version):
    saved_weights = sparrow.get_data(f'networks/{network}/{network}_network_v{version}.pt')

    if not os.path.isfile(saved_weights):
        raise SparrowException(f'Error: could not find saved weights file {saved_weights} for {network} predictor')
    
    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    loaded_model = torch.load(saved_weights, map_location=torch.device(device))

    # count number of network layers
    num_layers = 0
    while True:
        s = f'lstm.weight_ih_l{num_layers}'
        try:
            temp = loaded_model[s]
            num_layers += 1
        except KeyError:
            break
                    
    number_of_classes = np.shape(loaded_model['fc.bias'])[0]
    
    # hardcoded for 20 aa, but could change if different encoding scheme.
    input_size = 20 

    hidden_vector_size = int(np.shape(loaded_model['lstm.weight_ih_l0'])[0] / 4)

    model = brnn_architecture.BRNN_MtO(input_size, hidden_vector_size, num_layers, number_of_classes, device)
    model.load_state_dict(loaded_model)

    return device, model

def batch_predict(protein_objs : List[sparrow.Protein], batch_size : int, network=None,version=1) -> dict:
    if network not in ["rg", "re", "prefactor", "asphericity", "scaling_exponent", "scaled_re", "scaled_rg"]:
        raise SparrowException("Please choose a valid network for batch predictions")
    
    device, model = prepare_model(network,version)
   
    pred_dict = {}
    sequences = [protein_obj.sequence for _, protein_obj in protein_objs.items()]
    seq_loader = DataLoader(sequences, batch_size=batch_size, shuffle=False)

    for batch in tqdm(seq_loader):
        # Pad the sequence vector to have the same length as the longest sequence in the batch
        seqs_padded = pad_sequence([encode_sequence.one_hot(seq).float() for seq in batch], batch_first=True)

        # Move padded sequences to device
        seqs_padded = seqs_padded.to(device)

        # Forward pass
        outputs = model.forward(seqs_padded).detach().cpu().numpy()

        # Save predictions
        for j, seq in enumerate(batch):
            pred_dict[seq] = outputs[j]
    
    return pred_dict