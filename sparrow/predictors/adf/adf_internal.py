import torch
import numpy as np
from typing import List, Dict
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import os

from packaging import version as VERSION_PACKAGE

from parrot import brnn_architecture
from parrot import encode_sequence
import sparrow

from sparrow.sparrow_exceptions import SparrowException



# ....................................................................................
#
#
def prepare_model(network, version, gpuid):
    """
    Given a predictor name and version, load the network weights and parameters 
    to the appropriate device.

    Parameters
    ----------
    network : str
        Name of the network you want to predict

    version : int
        The version of the network you want to use for predictions

    Returns
    -------
    tuple
        returns the available device and the appropriately versioned network model.

    Raises
    ------
    SparrowException
        An exception is raised if there is no weights file corresponding to the 
        network and version requested.
    """
    saved_weights = sparrow.get_data(f'networks/adf/{network}_network_v{version}.pt')

    if not os.path.isfile(saved_weights):
        raise SparrowException(f'Error: could not find saved weights file {saved_weights} for {network} predictor')
    
    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpuid}")
    else:
        device = torch.device("cpu")

    # load the model
    loaded_model = torch.load(saved_weights, map_location=device)
    

    # count number of network layers
    num_layers = 0
    while True:
        s = f'lstm.weight_ih_l{num_layers}'
        try:
            temp = loaded_model[s]
            num_layers += 1
        except KeyError:
            break

    # count number of classes
    number_of_classes = np.shape(loaded_model['fc.bias'])[0]
    
    # hardcoded for 20 aa, but could change if different encoding scheme.
    input_size = 20 

    # determine hidden vector size
    hidden_vector_size = int(np.shape(loaded_model['lstm.weight_ih_l0'])[0] / 4)

    # create model
    model = brnn_architecture.BRNN_MtO(input_size, hidden_vector_size, num_layers, number_of_classes, device)

    # load weights
    model.load_state_dict(loaded_model)

    return (device, model)



# ....................................................................................
#
#
def __size_filter(inseqs):
    """
    Helper function that breaks down sequences into groups
    where all sequences are the same size. Used as part of the
    size-collect algorithm.

    Parameters
    ---------------
    inseqs : list
        List of amino acid sequencs

    Returns
    ---------------
    dict
        Returns a dictionary where keys are sequence length and
        values are a list of sequences where all seqs are same length    
    """

    retdict = {}

    for s in inseqs:
        if len(s) not in retdict:
            retdict[len(s)] = []

        retdict[len(s)].append(s)

    return retdict


# ....................................................................................
#
#
def __list2dict(protein_objs):
    """
    Helper function that takes in variable and if it's a list
    converts into a dictionary with keys as indices (1,2,...n)
    while if its a dictionary just returns the same dictionary 
    again.

    Parameters
    --------------
    protein_objs : list or dict
        Input data which will be converted to a dictionary IF
        its a list, otherwise if a dictionary is passed in it
        is returned, else an exception is raised

    Returns
    -------------
    dict
         Guarenteed to return a dictionary

    Raises
    sparrow.exception.SparrowException 


    """
    if type(protein_objs) is list:
        tmp = {}
        for i in range(0,len(protein_objs)):
            tmp[i] = protein_objs[i]

        return tmp
    elif type(protein_objs) is dict:
        return protein_objs
    else:
        raise SparrowException('Input must be one of a list or a dictionary')
    


    
# ....................................................................................
#
#
def adf_prediction(protein_objs,
                   network : str = None,
                   batch_size : int = 32,
                   version : int = 1,
                   gpuid : int = 0,
                   return_seq2prediction : bool = False,
                   show_progress_bar : bool = True,
                   safe : bool = True) -> dict:                  
    """Perform batch predictions for the adf networks

    Input can be either a dictionary or a list.

    If a dictionary is provided, the values in the dictionary must
    either be sparrow.protein.Protein objects, OR must be protein
    sequences represented as strings.

    If a list is provided, the elements in the list must
    either be sparrow.protein.Protein objects, OR must be protein
    sequences represented as strings.

    The function returns a dictionary that enables mapping between
    sequence and prediction, although the details of this mapping
    depend on the parameter return_seq2prediction.

    If return_seq2prediction is False (default), the return 
    dictionary maps unique indices (either those in the input
    dictionary, or indices 1,2,...n) to a list with two elements;

        * [0] = sequence
        * [1] = prediction

    This ensures that it is easy to relate both index information
    and sequence to the new prediction.

    If return_seq2prediction is True, then the return dictionary
    useses sequences as keys and predictions as values. This can
    sometimes be an ideal return type, but makes mapping between
    sequence and index (if an input dictionary was passed) tricky.

    The order of the return dictionaries is guarenteed to match
    the order of the input dictionary and list if possible; i.e.
    if return_seq2prediction=False then the orders are guarenteed,
    while if return_seq2prediction=True then if there are duplicate
    sequences these only preserve the first appearance of the
    sequence.

    NB:
    Note if the requested network is Rg or Re, any sequences
    shorter than 35 amino acids will default to using the scaled
    rg and scaled re networks. These networks give much more accurate
    values for short sequences. That said, if using the non-scaled
    network is preferable, setting safe=False will over-ride this
    option, although we do not recommend this.


    Parameters
    ----------
    protein_objs : dict or list
        Either a dictionary of key-value pairs where values are either sequence strings
        or sparrow.protein.Protein objects, or a list, where elements are either sequence 
        strings or sparrow.protein.Protein objects.

    network : str
        The name of the network you wish to predict, by default None
        Currently implemented options include:

            "rg"
            "re",
            "prefactor",
            "asphericity",
            "scaling_exponent", 
            "scaled_re", 
            "scaled_rg"

    batch_size : int
        The batch size to use for network forward pass. 
        This should be <= the batch size used during training.
        Default = 32.
        
    version : int, optional
        Network version number, by default 2 (released May 21st 2023).
    
    gpuid : int, optional
        GPU ID to use for predictions, by default 0

    batch_algorithm : str
        Selector which defines the approach used to batch processes
        the sequences. Must be one of the following:

        'default' - Selects either of the following two algorithms, 
                    prefering pad-n-pack if verison of torch allows
   
        'size-collect' - Break sequences into groups where all
                         sequences within a group are the same
                         size, and then pass these grouped sequences
                         into the device. This is available in all
                         versions of torch, and gives the best performance
                         for very large sequence sets or sets where there
                         are many sequences of the same length.

        Note that right now the ONLY algorithm available is size-collect,
        so regardless of what selector your choose here, you'll get
        size-collect batch algorithm.

    return_seq2prediction : bool
        Flag which, if set to true, means the return from this function is 
        a dictionary where keys are unique sequences and values are the 
        prediction. This is in contrast to the default return which is a
        dictionary where keys are unique indices matching the input dictionary
        are values are a tuple of sequence and prediction. Default = False.

    show_progress_bar : bool
        Flag which, if set to True, means a progress bar is printed as 
        predictions are made, while if False no progress bar is printed.
        Default  =  True
            
    safe : bool
        Flag which, if set to False, means the requested
        network will be used for rg/re prediction 
        regardless of the sequence length. NOT RECOMMENDED.
        Default = True.

    Returns
    -------
    dict
        sequence, value(s) mapping for the requested predictor.

    Raises
    ------
    SparrowException
        An exception is raised if the requested network is not one of the available options.
    """


    ## ------------------------------------------------------------------------------------
    ##
    ## Sanitize inputs
    ##

    # check a valid network was passed
    #if network not in ["rg", "re", "prefactor", "asphericity", "scaling_exponent", "scaled_re", "scaled_rg"]:
    #    raise SparrowException("For option 'network': Please choose a valid network for batch predictions")



    # check network names are valid
    if network not in ['adf_erijman',
                       'adf_morffy_min',
                       'adf_morffy_max',
                       'adf_morffy_mean',
                       'adf_morffy_class',                       
                       'adf_sanborn_min',
                       'adf_sanborn_max',
                       'adf_sanborn_mean',
                       'adf_sanborn_class']:
        raise SparrowException("For option 'network': Please choose a valid network for batch predictions")

    # check if we're using a probabilitic classfifier or a regression network
    if network in ['adf_erijman', 'adf_morffy_class', 'adf_sanborn_class']:
        mode = 'prob-classification'
    elif network in ['adf_morffy_min', 'adf_morffy_max', 'adf_morffy_mean', 'adf_sanborn_min', 'adf_sanborn_max', 'adf_sanborn_mean']:
        mode = 'regression'
            
    ## ------------------------------------------------------------------------------------
    ##
    ## Homogenize input to ensure protein_objs is a dictionary 
    ##

    ## If we passed a list, convert into a dictionary where indices are integers and 0...n
    ## such that we have unique list->dict mapping with same order preserved.
    protein_objs = __list2dict(protein_objs)

    ## ------------------------------------------------------------------------------------
    ##
    ## Build list of sequences and mapping between sequence and one or more key values in
    ## the input dictioanry
    ##
    seq2id = {}
    if sum([1 for i in protein_objs if type(protein_objs[i]) is sparrow.protein.Protein]) == len(protein_objs):

        ## for input dictionary, if all elements are sparrow.protein.Protein objects
        ## extract out sequences,        
        sequence_list = [protein_obj.sequence for _, protein_obj in protein_objs.items()]

        # and build mapping between sequence and 1 or more keys
        for idx in protein_objs:
            seq = protein_objs[idx].sequence

            if seq not in seq2id:
                seq2id[seq] = []
            seq2id[seq].append(idx)
            
    elif sum([1 for i in protein_objs if type(protein_objs[i]) is str]) == len(protein_objs):

        ## if every element is a string use these as sequences
        sequence_list = list(protein_objs.values())

        # and build mapping between sequence and 1 or more keys
        for idx in protein_objs:
            seq = protein_objs[idx]

            if seq not in seq2id:
                seq2id[seq] = []
            seq2id[seq].append(idx)
        
    else:
        raise SparrowException('Invalid input - must pass either a dictionary of key-values where values are sequences (str)\nor sparrow.protein.Protein objects, or a list of sequences (str) or sparrow.protein.Protein objects')



    ## ------------------------------------------------------------------------------------
    ##
    ## Perform prediction
    ##

    pred_dict = {}
    device, model = prepare_model(network, version, gpuid)
    model.to(device)

    # size-collect means we systematically subdivide the sequences into groups 
    # where they're all the same length in a given megabatch, meaning we don't
    # need to pad. This works well if you have many sequences of the same length,
    # but is not optimal in that the effective batch size ends up being 1 for every
    # uniquely-lengthed sequence.
        
    # build a dictionary where keys are sequence length
    # and values is a list of sequences of that exact length
    size_filtered =  __size_filter(sequence_list)
        
    loop_range = tqdm(size_filtered) if show_progress_bar else size_filtered
    
    for local_size in loop_range:

        local_seqs = size_filtered[local_size]
        seq_loader = DataLoader(local_seqs, batch_size=batch_size, shuffle=False)

        # batch here is a subset of sequences from local_seqs, where each batch is
        # defined by the batch_size parameter
        for batch in seq_loader:

            # Pad the sequence vector to have the same length as the longest sequence in the batch. The batch_first=True
            # parameter means that the first dimension of the tensor is the batch size, and the second dimension is the
            # sequence length. This is the format expected by the network.
            seqs_padded = pad_sequence([encode_sequence.one_hot(seq).float() for seq in batch], batch_first=True)

            # Move padded sequences to device
            seqs_padded = seqs_padded.to(device)

            # this is the actual prediction step
            outputs = model.forward(seqs_padded).detach().cpu().numpy()
            
            for j, seq in enumerate(batch):

                if mode == 'prob-classification':
                    softmax = np.around(np.exp(outputs[j]) / np.sum(np.exp(outputs[j])), decimals=4)
                    pred_dict[seq] = softmax#outputs[j]
                    
                elif mode == 'regression':
                    pred_dict[seq] = outputs[j][0]


    if return_seq2prediction:
        return pred_dict

    else:
        tmp = {}
        for seq in seq2id:
            for idx in seq2id[seq]:
                tmp[idx] = [seq, pred_dict[seq]]

        return_dict = {}
        for idx in protein_objs:
            return_dict[idx] = tmp[idx]
                
        return return_dict
