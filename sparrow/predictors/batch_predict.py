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

from .asphericity.asphericity_predictor import AsphericityPredictor
from .e2e.end_to_end_distance_predictor import RePredictor
from .rg.radius_of_gyration_predictor import RgPredictor
from .prefactor.prefactor_predictor import PrefactorPredictor
from .scaling_exponent.scaling_exponent_predictor import ScalingExponentPredictor
from .scaled_re.scaled_end_to_end_distance_predictor import ScaledRePredictor
from .scaled_rg.scaled_radius_of_gyration_predictor import ScaledRgPredictor
from ..sparrow_exceptions import SparrowException

from sparrow.data.configs import MIN_LENGTH_ALBATROSS_RE_RG

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
    saved_weights = sparrow.get_data(f'networks/{network}/{network}_network_v{version}.pt')

    if not os.path.isfile(saved_weights):
        raise SparrowException(f'Error: could not find saved weights file {saved_weights} for {network} predictor')
    
    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpuid}")
    else:
        device = torch.device("cpu")

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
                    
    number_of_classes = np.shape(loaded_model['fc.bias'])[0]
    
    # hardcoded for 20 aa, but could change if different encoding scheme.
    input_size = 20 

    hidden_vector_size = int(np.shape(loaded_model['lstm.weight_ih_l0'])[0] / 4)

    model = brnn_architecture.BRNN_MtO(input_size, hidden_vector_size, num_layers, number_of_classes, device)
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
def __short_seq_fix(protein_objs,
                    batch_size : int,
                    network : str = None,
                    version : int = 2,
                    gpuid : int = 0,
                    batch_algorithm : str = 'default',
                    return_seq2prediction : bool = False,
                    show_progress_bar : bool = True):
                    

    """
    Helper function that deals with datasets when we want to predict Re and Rg.

    The standard 'rg' and 're' networks perform poorly on short sequences. Here short
    is defined as less than the value set in sparrow.data.config.MIN_LENGTH_ALBATROSS_RE_RG,
    which at the time of writing is 35 residues (i.e. pretty damn short).

    This function scans through all the passed sequences and divides the input dictionary
    into short_seqs and long_seqs.

    Then, any short_seq sequences are predicted using the scaled_ network, while those in
    long_seq are predicted using the non-scaled network.

    Parameters
    ----------
    protein_objs : dict or list
        Either a dictionary of key-value pairs where values are either sequence strings
        or sparrow.protein.Protein objects, or a list, where elements are either sequence 
        strings or sparrow.protein.Protein objects.

    batch_size : int
        The batch size to use for network forward pass. 
        This should be <= the batch size used during training.

    network : str
        Must be either 'rg' or 're'
        
    version : int, optional
        Network version number, by default 2
    
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

    Returns
    -------
    dict
        sequence, value(s) mapping for the requested predictor.
    
    """

    if network not in ['rg', 're']:
        raise SparrowException("__short_seq_fix() requires the network to be 'rg' or 're'")
    

    short_seqs = {}
    long_seqs = {}

    # split the dataset in two based on the threshold. Note, we have two cases here, one if
    # protein_objs is already a dictionary, and one if its a list, whereby we create a new
    # dictionary indexed by unique integers starting at 0 and incrementing with each sequence.
    # NOTE: The new dictionaries, if created, must have unique indices across both sets of
    # sequences because the short and long sequences will be recombined at the end. 
    if type(protein_objs) is dict:
        for s in protein_objs:    
            if len(protein_objs[s]) < MIN_LENGTH_ALBATROSS_RE_RG:
                short_seqs[s] = protein_objs[s]
            else:
                long_seqs[s] = protein_objs[s]
                
    elif type(protein_objs) is list:
        idx = 0
        for s in protein_objs:    
            if len(s) < MIN_LENGTH_ALBATROSS_RE_RG:
                short_seqs[idx] = s
            else:
                long_seqs[idx] = s
            idx = idx + 1
    else:
        raise SparrowException('Invalid types passed to protein_objs; must be a dict or list where values/elements are either strings or sparrow.protein.Protein objects') 

    ##
    ## For the code block below, we build a tmp_return_dict which undergoes a final processing step to ensure
    ## its in the same final order as the input dictonary
    
    # if we have 1 or more short sequences then we know at least a subset of sequences must be run using the scaled_ network
    if len(short_seqs) > 0:

        # run predictions on each set
        return_dict1 = batch_predict(short_seqs, batch_size=batch_size, network=f'scaled_{network}', version=version, gpuid=gpuid, batch_algorithm=batch_algorithm, return_seq2prediction=return_seq2prediction, show_progress_bar=show_progress_bar)

        # if we also had 1 or more long sequences then run this using the non-scaled network
        if len(long_seqs) > 0:

            # note we have need to set safe=False here or we fall into infinite recursive calls
            return_dict2 = batch_predict(long_seqs,  batch_size=batch_size, network=network, version=version, gpuid=gpuid, batch_algorithm=batch_algorithm, return_seq2prediction=return_seq2prediction, show_progress_bar=show_progress_bar, safe=False)

            # merge output and return; recall this would destroy any order of insertion, hence why we use a dictionary instead of a list
            tmp_return_dict = {**return_dict1, **return_dict2}
        else:
            tmp_return_dict = return_dict1
    else:

        # note we have need to set safe=False here or we fall into infinite recursive calls
        tmp_return_dict = batch_predict(protein_objs, batch_size=batch_size, network=network, version=version, gpuid=gpuid, batch_algorithm=batch_algorithm, return_seq2prediction=return_seq2prediction, show_progress_bar=show_progress_bar, safe=False)


    ## this final step ensures we return a dictionary ordered to match the
    ## order of the input

    # convert protein_objs into a dictionary so we can be consistent in how we parse (instead of separately parsing
    # input lists and dicts)
    input_dict = __list2dict(protein_objs)
    return_dict = {}

    # if return mode is seq2predictions...
    if return_seq2prediction:
            
        # If we want to return data as seq:pred dictionary
        # then rebuild the final return dict in the order
        # of the input_dict...            
        for k in input_dict:

            # get 'value' associated with each input element
            s = input_dict[k]

            # if that value is already a protein string 
            if type(s) is str:
                return_dict[s] = tmp_return_dict[s]

            # else if the value is a sparrow.protein.Protein object
            elif type(s) is sparrow.protein.Protein:
                return_dict[s.sequence] = tmp_return_dict[s.sequence]

            # else throw an exception
            else:
                raise SparrowException('Error parsing type of return dictionary. This is a bug, please report https://github.com/idptools/sparrow/issues')

    # if we're returning the standard key:[seq,pred] mapping
    else:
        for k in input_dict:
            return_dict[k] = tmp_return_dict[k]
            
    return return_dict
        
    
# ....................................................................................
#
#
def batch_predict(protein_objs,
                  network : str = None,
                  batch_size : int = 32,
                  version : int = 2,
                  gpuid : int = 0,
                  batch_algorithm = 'default',
                  return_seq2prediction : bool = False,
                  show_progress_bar : bool = True,
                  safe : bool = True) -> dict:                  
    """Perform batch predictions with a PARROT network in sparrow.

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


    ## bonus docstring for when pad-n-pack is working
    """


        'pad-n-pack' -   Use PyTorch'es padding/packing approach to
                         give the illusion of sequences having the 
                         same length. Note this is only available in
                         pytorch version 1.11.0 or higher. Note that
                         for large sequence datasets this may be slower
                         than size-collect.

        Default = 'default' which means in pytorch 1.11.0 or higher 
        pad-n-pack is preferred, although if you're working with large
        sequence datasets and/or all your sequences are the same length
        its probably a good idea to request size-collect.



    """

    ## ------------------------------------------------------------------------------------
    ##
    ## Sanitize inputs
    ##

    # check a valid network was passed
    if network not in ["rg", "re", "prefactor", "asphericity", "scaling_exponent", "scaled_re", "scaled_rg"]:
        raise SparrowException("For option 'network': Please choose a valid network for batch predictions")

    # check a valid batch_algo was passed
    if batch_algorithm not in ['default', 'size-collect', 'pad-n-pack']:
        raise SparrowException("For option 'batch_algorithm': Please choose a valid batch algorithm, one of 'default', 'size-collect', 'pad-n-pack")

    ## OVERRIDE - as of sparrow version 0.2.1 the only batch prediction algorithm working is
    ## size-collect. An initial draft of pad-n-pack is implemented but it DOES NOT WORK, so we hardcode
    ## in size-collect for now
    batch_algorithm = 'size-collect'

    # note this is moot in 0.2.1
    # pad-n-pack only available in PyTorch 1.11.0 or higher
    if batch_algorithm == 'pad-n-pack' and VERSION_PACKAGE.parse(torch.__version__) < VERSION_PACKAGE.parse("1.11.0"):
        raise SparrowException("For option 'batch_algorithm': pad-n-pack is only available in PyTorhc 1.11.0 or higher. Please use 'default' or 'size-collect'")

    # if we've requested to use rg/re predictions, this function ensures short sequences are appropriately
    # dealt with (this is a recurisve function that calls batch_predict)
    if network in ['rg','re'] and safe:
        return  __short_seq_fix(protein_objs,
                                batch_size=batch_size,
                                network=network,
                                version=version,
                                gpuid=gpuid,
                                batch_algorithm=batch_algorithm,
                                return_seq2prediction=return_seq2prediction,
                                show_progress_bar=show_progress_bar)


    ## Note in 0.2.1 this does not get called because of the over-ride, but we're leaving
    # it in in preparation for pad-n-pack working...
    # set the batch algorith if default was selected, otherwise use the requested version
    if batch_algorithm == 'default':
        if VERSION_PACKAGE.parse(torch.__version__) >= VERSION_PACKAGE.parse("1.11.0"):
            batch_algorithm = 'pad-n-pack'
        else:
            batch_algorithm = 'size-collect'
    

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

    if batch_algorithm == 'size-collect':
        # size-collect means we systematically subdivide the sequences into groups 
        # where they're all the same length in a given megabatch, meaning we don't
        # need to pad. This works well in earlier version or torch, but is not optimal
        # in that the effective batch size ends up being 1 for every uniquely-lengthed
        # sequence.
        
        # build a dictionary where keys are sequence length
        # and values is a list of sequences of that exact length
        size_filtered =  __size_filter(sequence_list)
        
        loop_range = tqdm(size_filtered) if show_progress_bar else size_filtered
        
        #for local_size in tqdm(size_filtered):
        for local_size in loop_range:

            local_seqs = size_filtered[local_size]

            seq_loader = DataLoader(local_seqs, batch_size=batch_size, shuffle=False)

            for batch in seq_loader:

                # Pad the sequence vector to have the same length as the longest sequence in the batch
                seqs_padded = pad_sequence([encode_sequence.one_hot(seq).float() for seq in batch], batch_first=True)

                # Move padded sequences to device
                seqs_padded = seqs_padded.to(device)

                outputs = model.forward(seqs_padded).detach().cpu().numpy()

                for j, seq in enumerate(batch):
                    pred_dict[seq] = outputs[j][0]

    elif batch_algorithm == 'pad-n-pack':

        raise SparrowException('pad-n-pack does not work currently - do not use')

        ## 
        ## pad-n-pack is not currently working. Do not use.
        ##

        seq_loader = DataLoader(sequence_list, batch_size=batch_size, shuffle=False)

        loop_range = tqdm(seq_loader) if show_progress_bar else seq_loader
                            
        #for batch in tqdm(seq_loader):
        for batch in loop_range:
            # Pad the sequence vector to have the same length as the longest sequence in the batch
            seqs_padded = pad_sequence([encode_sequence.one_hot(seq).float() for seq in batch], batch_first=True)

            # get lengths for input into pack_padded_sequence
            lengths = torch.tensor([len(seq) for seq in batch])

            # pack up for vacation
            packed_and_padded = pack_padded_sequence(seqs_padded, lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)

            # input packed_and_padded into loaded lstm
            packed_output, (ht, ct) = (model.lstm.forward(packed_and_padded))
            
            # inverse of pack_padded_sequence
            output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
            
            # get the outputs by calling fc
            outputs = model.fc(output).cpu()

            # get the unpacked, finalized values into the dict.
            for cur_ind, score in enumerate(outputs):
                
                # first detach, flatten, etc
                cur_score = score.detach().numpy().flatten()
                
                # get the sequence from batch from seq_loader
                cur_seq = batch[cur_ind]

                pred_dict[cur_seq] = cur_score[0]

    else:
        raise SparrowException('Invalid batch_algorithm passed')

    # finally, if using a scaled network re-multiply by np.sqrt(N)
    if network in ['scaled_re', 'scaled_rg']:        
        for s in pred_dict:
            pred_dict[s] = pred_dict[s]*np.sqrt(len(s))


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
