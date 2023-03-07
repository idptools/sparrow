import torch
import numpy as np
from typing import List
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

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

def batch_predict(protein_objs : List[sparrow.Protein], batch_size : int, predictor=None) -> np.array:
        # Check if GPU is available
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        pred_dict = {}
        sequences = [protein_obj.sequence for protein_obj in protein_objs]
        seq_loader = DataLoader(sequences, batch_size=batch_size, shuffle=False)

        for batch in tqdm(seq_loader):
            # Pad the sequence vector to have the same length as the longest sequence in the batch
            seqs_padded = pad_sequence([encode_sequence.one_hot(seq).float() for seq in batch], batch_first=True)

            # Move padded sequences to device
            seqs_padded = seqs_padded.to(device)

            # Forward pass
            outputs = predictor.network(seqs_padded).detach().cpu().numpy()

            # Save predictions
            for j, seq in enumerate(batch):
                pred_dict[seq] = outputs[j]
        
        return pred_dict