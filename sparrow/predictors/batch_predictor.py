"""Running sparrow's networks over many sequences at once.

:func:`run_network` is the batch harness: it runs any network saved by PARROT 1
- a many-to-one network, with one value per sequence, or a many-to-many
network, with one value per residue - over a list of sequences in batches, and
returns the network's raw output for each, in the shape the network gives it for
that sequence alone. Each predictor class turns that output into its prediction
with the same transform it uses for one sequence
(:meth:`~sparrow.predictors.base.BaseNetworkPredictor.predict_many`).

:func:`batch_predict` is the public function: it runs one of sparrow's networks
by name (:data:`~sparrow.predictors.NETWORK_PREDICTORS`) over a dict or list of
sequences or ``Protein`` objects. :func:`sparrow.batch` uses it for every
network-backed value it is asked for.
"""

# torch is mocked when the docs are built on Read the Docs, and a mock cannot be part
# of a union such as ``str | torch.device``, so annotations are not evaluated
from __future__ import annotations

from typing import Optional

import torch
from packaging import version as VERSION_PACKAGE
from parrot import encode_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

import sparrow
from sparrow.data.configs import MIN_LENGTH_ALBATROSS_RE_RG
from sparrow.predictors.devices import DEFAULT_DEVICE, full_float32_precision, resolve_device
from sparrow.sparrow_exceptions import SparrowException

#: Sequences per forward pass when none is given. Larger is faster on a GPU, and
#: changes predictions only at the networks' 32-bit precision.
DEFAULT_BATCH_SIZE = 32

#: The batch algorithm used when none is given, which is pad-n-pack where PyTorch has it.
DEFAULT_BATCH_ALGORITHM = "default"

BATCH_ALGORITHMS = ("default", "size-collect", "pad-n-pack")

# pack_padded_sequence with enforce_sorted=False needs PyTorch 1.11 or newer
_MINIMUM_TORCH_FOR_PAD_N_PACK = "1.11.0"

# the two architectures of a network saved by PARROT 1
_SEQUENCE_ARCHITECTURE = "MtO"
_RESIDUE_ARCHITECTURE = "MtM"

# rg and re have a scaled counterpart, which is more accurate for short sequences
_SCALED_COUNTERPARTS = {"rg": "scaled_rg", "re": "scaled_re"}

# The name PARROT 2 gives the one-hot encoding (OneHotEncoder.name), which every
# network saved by PARROT 1 reads
_ONE_HOT_ENCODING = "onehot"


# ....................................................................................
#
#
def resolve_batch_algorithm(batch_algorithm):
    """The batch algorithm to use, with ``'default'`` resolved and availability checked.

    Parameters
    ----------
    batch_algorithm : {'default', 'size-collect', 'pad-n-pack'}
        As for :func:`run_network`.

    Returns
    -------
    str
        ``'size-collect'`` or ``'pad-n-pack'``.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``batch_algorithm`` is not one of the three, or is ``'pad-n-pack'``
        on PyTorch older than 1.11.
    """
    if batch_algorithm not in BATCH_ALGORITHMS:
        raise SparrowException(
            "For option 'batch_algorithm': Please choose a valid batch algorithm, one of 'default', 'size-collect', "
            "'pad-n-pack"
        )

    has_pad_n_pack = VERSION_PACKAGE.parse(torch.__version__) >= VERSION_PACKAGE.parse(_MINIMUM_TORCH_FOR_PAD_N_PACK)
    if batch_algorithm == "pad-n-pack" and not has_pad_n_pack:
        raise SparrowException(
            "For option 'batch_algorithm': pad-n-pack is only available in PyTorch 1.11.0 or higher. Please use "
            "'default' or 'size-collect'"
        )

    if batch_algorithm == "default":
        return "pad-n-pack" if has_pad_n_pack else "size-collect"
    return batch_algorithm


def _one_hot_batch(sequences, device):
    """One-hot encode a batch of sequences, padded with zeros to the longest.

    Parameters
    ----------
    sequences : list of str
        The batch.
    device : torch.device
        Where the network runs.

    Returns
    -------
    torch.Tensor
        ``(len(sequences), longest, 20)``, on ``device``.
    """
    return pad_sequence([encode_sequence.one_hot(seq).float() for seq in sequences], batch_first=True).to(device)


def _run_packed(network, sequences, architecture, device):
    """Run one mixed-length batch through a network, packed so padding has no effect.

    Each batch is padded to its longest member and packed with
    ``pack_padded_sequence``, so the LSTM runs over each sequence's true
    residues only - the reverse direction starts at each sequence's own last
    residue, not in the padding. This is the network's own forward pass - the
    LSTM, then the fully connected layer - on the packed input: a many-to-one
    network decodes the final hidden states of the last forward and reverse
    layers, and a many-to-many network decodes the output at every residue.

    Parameters
    ----------
    network : torch.nn.Module
        The network: a PARROT 1 ``BRNN_MtO`` or ``BRNN_MtM``.
    sequences : list of str
        The batch.
    architecture : {'MtO', 'MtM'}
        The network's architecture.
    device : torch.device
        Where the network runs.

    Returns
    -------
    list of numpy.ndarray
        The output for each sequence, as :func:`run_network` returns it.
    """
    padded = _one_hot_batch(sequences, device)
    # lengths must be a CPU tensor for pack_padded_sequence; enforce_sorted=False
    # lets the batch stay in input order, which the outputs keep
    lengths = torch.tensor([len(seq) for seq in sequences])
    packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)

    with torch.no_grad():
        if architecture == _SEQUENCE_ARCHITECTURE:
            _, (h_n, _) = network.lstm(packed)
            # the final hidden states of the last forward (-2) and reverse (-1)
            # layers, as BRNN_MtO.forward decodes them
            outputs = network.fc(torch.cat((h_n[-2], h_n[-1]), dim=-1)).cpu().numpy()
            return [outputs[index : index + 1] for index in range(len(sequences))]

        packed_output, _ = network.lstm(packed)
        per_residue, _ = pad_packed_sequence(packed_output, batch_first=True)
        outputs = network.fc(per_residue).cpu().numpy()
        return [outputs[index : index + 1, : len(seq)] for index, seq in enumerate(sequences)]


def _run_same_length(network, sequences, device):
    """Run one batch of sequences that are all the same length, which needs no padding.

    Parameters
    ----------
    network : torch.nn.Module
        The network: a PARROT 1 ``BRNN_MtO`` or ``BRNN_MtM``.
    sequences : list of str
        The batch, every sequence the same length.
    device : torch.device
        Where the network runs.

    Returns
    -------
    list of numpy.ndarray
        The output for each sequence, as :func:`run_network` returns it.
    """
    with torch.no_grad():
        outputs = network(_one_hot_batch(sequences, device)).cpu().numpy()
    return [outputs[index : index + 1] for index in range(len(sequences))]


def run_network(
    network,
    sequences,
    *,
    architecture,
    device,
    batch_size=DEFAULT_BATCH_SIZE,
    batch_algorithm=DEFAULT_BATCH_ALGORITHM,
    show_progress_bar=False,
):
    """Run a network saved by PARROT 1 over many sequences, in batches.

    The batch harness behind every network sparrow runs in batches. It returns
    the network's raw output for each sequence, in the shape the network gives
    it for that sequence alone, so each predictor turns it into a prediction
    with the transform it uses for one sequence. The output can differ from the
    output for the sequence alone by about 1e-7 relative, the precision of the
    networks' 32-bit arithmetic, because a batch is summed in a different
    order.

    Parameters
    ----------
    network : torch.nn.Module
        The network, on ``device``: a PARROT 1 ``BRNN_MtO`` or ``BRNN_MtM``.
    sequences : list of str
        Upper-case amino acid sequences, run in the order given. A sequence
        given twice is run twice.
    architecture : {'MtO', 'MtM'}
        ``'MtO'`` for a many-to-one network, with one output per sequence;
        ``'MtM'`` for a many-to-many network, with one per residue.
    device : torch.device
        Where the network runs.
    batch_size : int, optional
        Sequences per forward pass. Default 32.
    batch_algorithm : {'default', 'size-collect', 'pad-n-pack'}, optional
        ``'pad-n-pack'`` batches sequences of any lengths together, each batch
        padded to its longest and packed so that the padding has no effect;
        ``'size-collect'`` batches only sequences of the same length, which
        need no padding, so a length that appears once is a batch of one.
        ``'default'`` (the default) is pad-n-pack, or size-collect on PyTorch
        older than 1.11, which has no pad-n-pack.
    show_progress_bar : bool, optional
        Show a progress bar over the batches (pad-n-pack) or the lengths
        (size-collect). Default False.

    Returns
    -------
    list of numpy.ndarray
        One array per sequence, in order: ``(1, number_of_classes)`` for a
        many-to-one network, and ``(1, len(sequence), number_of_classes)`` for a
        many-to-many network.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the architecture or batch algorithm is not valid.
    """
    if architecture not in (_SEQUENCE_ARCHITECTURE, _RESIDUE_ARCHITECTURE):
        raise SparrowException(f"Unknown network architecture '{architecture}' (expected 'MtO' or 'MtM')")
    batch_algorithm = resolve_batch_algorithm(batch_algorithm)

    # on an NVIDIA GPU, in full float32 precision, as PARROT 2 runs its networks
    with full_float32_precision():
        if batch_algorithm == "size-collect":
            return _run_by_length(network, sequences, device, batch_size, show_progress_bar)
        return _run_in_mixed_batches(network, sequences, architecture, device, batch_size, show_progress_bar)


def _run_by_length(network, sequences, device, batch_size, show_progress_bar):
    """Run a network over sequences batched by length (size-collect), for :func:`run_network`.

    Parameters
    ----------
    network : torch.nn.Module
        The network, on ``device``.
    sequences : list of str
        The sequences.
    device : torch.device
        Where the network runs.
    batch_size : int
        Sequences per forward pass.
    show_progress_bar : bool
        Show a progress bar over the lengths.

    Returns
    -------
    list of numpy.ndarray
        The output for each sequence, in order, as :func:`run_network` returns it.
    """
    outputs = [None] * len(sequences)

    # the positions of the sequences of each length, lengths in order of first appearance
    positions_by_length = {}
    for position, seq in enumerate(sequences):
        positions_by_length.setdefault(len(seq), []).append(position)

    lengths = tqdm(positions_by_length) if show_progress_bar else positions_by_length
    for length in lengths:
        for batch in DataLoader(positions_by_length[length], batch_size=batch_size, shuffle=False):
            batch_positions = batch.tolist()
            batch_outputs = _run_same_length(network, [sequences[position] for position in batch_positions], device)
            for position, output in zip(batch_positions, batch_outputs):
                outputs[position] = output
    return outputs


def _run_in_mixed_batches(network, sequences, architecture, device, batch_size, show_progress_bar):
    """Run a network over sequences in mixed-length batches (pad-n-pack), for :func:`run_network`.

    Parameters
    ----------
    network : torch.nn.Module
        The network, on ``device``.
    sequences : list of str
        The sequences.
    architecture : {'MtO', 'MtM'}
        The network's architecture.
    device : torch.device
        Where the network runs.
    batch_size : int
        Sequences per forward pass.
    show_progress_bar : bool
        Show a progress bar over the batches.

    Returns
    -------
    list of numpy.ndarray
        The output for each sequence, in order, as :func:`run_network` returns it.
    """
    outputs = [None] * len(sequences)

    # torch declares DataLoader(dataset: Dataset[_T_co]) nominally, but a map-style
    # dataset only needs __len__/__getitem__, which a list has (isinstance([],
    # Dataset) is False). cast(Dataset, ...) would also silence this; the directive
    # is preferred because ty reports unused-ignore-comment if the stub is ever
    # fixed, whereas a cast would silently outlive it.
    batches = DataLoader(
        list(range(len(sequences))),  # ty: ignore[invalid-argument-type]
        batch_size=batch_size,
        shuffle=False,
    )
    batches = tqdm(batches) if show_progress_bar else batches
    for batch in batches:
        batch_positions = batch.tolist()
        batch_outputs = _run_packed(
            network, [sequences[position] for position in batch_positions], architecture, device
        )
        for position, output in zip(batch_positions, batch_outputs):
            outputs[position] = output
    return outputs


def predict_many(
    network,
    sequences,
    *,
    version=None,
    batch_size=DEFAULT_BATCH_SIZE,
    gpuid=None,
    batch_algorithm=DEFAULT_BATCH_ALGORITHM,
    show_progress_bar=False,
    device=DEFAULT_DEVICE,
):
    """Predict many sequences with one of sparrow's networks, by name.

    Parameters
    ----------
    network : str
        One of :data:`~sparrow.predictors.NETWORK_PREDICTORS`.
    sequences : list of str
        Valid amino acid sequences.
    version : str, int or None, optional
        The network version. Default None, the predictor's default.
    batch_size : int, optional
        Sequences per forward pass. Default 32.
    gpuid : int or None, optional
        The NVIDIA GPU to run on; see ``device``. Default None.
    batch_algorithm : {'default', 'size-collect', 'pad-n-pack'}, optional
        As for :func:`run_network`. Default ``'default'``.
    show_progress_bar : bool, optional
        Show a progress bar over the batches. Default False.
    device : {'auto', 'cpu', 'cuda', 'mps'} or torch.device, optional
        Where to run the network, as PARROT's ``Predictor`` takes it (see
        :func:`~sparrow.predictors.devices.resolve_device`). Default
        ``'auto'``: CUDA if this machine has it (the GPU ``gpuid`` names, or
        the current one), and the CPU otherwise.

    Returns
    -------
    list
        One prediction per sequence, in order, as the network's predictor
        makes it for one sequence.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``network`` is not one of sparrow's networks, or the device asked
        for is not valid or not on this machine.
    """
    from sparrow.predictors import load_network_predictor

    if version is not None:
        version = str(version)
    predictor = load_network_predictor(network, version=version, device=resolve_device(device, gpuid))
    return predictor.predict_many(
        sequences,
        batch_size=batch_size,
        batch_algorithm=batch_algorithm,
        show_progress_bar=show_progress_bar,
    )


# ....................................................................................
#
#
def _one_hot_module(predictor, network):
    """A loaded network as a torch module run on one-hot encoded sequences, whichever PARROT saved it.

    Parameters
    ----------
    predictor : sparrow.predictors.base.BaseNetworkPredictor
        The loaded predictor.
    network : str
        Its name, for the error message.

    Returns
    -------
    torch.nn.Module
        The network: a PARROT 1 BRNN, or the architecture PARROT 2 saved.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the network was saved by PARROT 2 and trained on an encoding other
        than one-hot, so it cannot be run on one-hot encoded sequences.
    """
    if not predictor.saved_by_parrot2:
        return predictor.network

    # predictor.network is PARROT 2's Predictor, which holds the network and the
    # encoding it was trained on
    parrot_predictor = predictor.network
    if parrot_predictor.encoder.name != _ONE_HOT_ENCODING:
        raise SparrowException(
            f"The {network!r} network was trained on the {parrot_predictor.encoder.name!r} encoding, so it cannot be "
            "run on one-hot encoded sequences as prepare_model's network is. Predict with it by name instead: "
            f"batch_predict(sequences, network={network!r}) or predict_many({network!r}, sequences)"
        )
    return parrot_predictor.network


def prepare_model(network, version=None, gpuid=None, device=DEFAULT_DEVICE):
    """
    Given a network name and version, load the network weights and parameters
    to the appropriate device.

    The network is returned as a ``torch.nn.Module`` in evaluation mode, which
    is called the same way whichever PARROT saved it: on a batch of one-hot
    encoded sequences, ``(batch, length, 20)``, it returns the network's raw
    output, ``(batch, number_of_classes)`` for a network that predicts one value
    per sequence or ``(batch, length, number_of_classes)`` for one per residue.
    sparrow runs its networks on an NVIDIA GPU in full float32 precision; to
    run this one as sparrow does, call it inside
    :func:`~sparrow.predictors.devices.full_float32_precision`.

    Parameters
    ----------
    network : str
        One of :data:`~sparrow.predictors.NETWORK_PREDICTORS`.

    version : int or str or None, optional
        The version of the network you want to use for predictions. Default
        None, the predictor's default.

    gpuid : int or None, optional
        The NVIDIA GPU to load the network onto; see ``device``. Default None.

    device : {'auto', 'cpu', 'cuda', 'mps'} or torch.device, optional
        Where to load the network, as PARROT's ``Predictor`` takes it (see
        :func:`~sparrow.predictors.devices.resolve_device`). Default
        ``'auto'``: CUDA if this machine has it (the GPU ``gpuid`` names, or
        the current one), and the CPU otherwise.

    Returns
    -------
    tuple
        ``(device, network)``: the ``torch.device`` the network is on, and the
        network.

    Raises
    ------
    SparrowException
        An exception is raised if there is no weights file corresponding to the
        network and version requested, if the device asked for is not valid or
        not on this machine, or if the network was trained with PARROT 2 on an
        encoding other than one-hot, so it cannot be run on one-hot encoded
        sequences.
    """
    from sparrow.predictors import load_network_predictor

    device = resolve_device(device, gpuid)
    predictor = load_network_predictor(network, version=None if version is None else str(version), device=device)
    return (device, _one_hot_module(predictor, network))


# ....................................................................................
#
#
def __list2dict(protein_objs):
    """
    Helper function that takes in variable and if it's a list
    converts into a dictionary with keys as 0-based indices
    (0,1,...n-1) while if its a dictionary just returns the same
    dictionary again.

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
        for i in range(0, len(protein_objs)):
            tmp[i] = protein_objs[i]

        return tmp
    elif type(protein_objs) is dict:
        return protein_objs
    else:
        raise SparrowException("Input must be one of a list or a dictionary")


# ....................................................................................
#
#
def __short_seq_fix(
    protein_objs,
    batch_size: int,
    network: Optional[str] = None,
    version=None,
    device: torch.device | None = None,
    batch_algorithm: str = DEFAULT_BATCH_ALGORITHM,
    return_seq2prediction: bool = False,
    show_progress_bar: bool = True,
):
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

    version : int, str or None, optional
        Network version number, by default None, the network's default

    device : torch.device or None, optional
        The device to run on, as :func:`batch_predict` resolved it. Default
        None, which ``batch_predict`` reads as ``'auto'``.

    batch_algorithm : str
        As for :func:`batch_predict`.

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

    if network not in _SCALED_COUNTERPARTS:
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
        raise SparrowException(
            "Invalid types passed to protein_objs; must be a dict or list where values/elements are either strings or sparrow.protein.Protein objects"
        )

    ##
    ## For the code block below, we build a tmp_return_dict which undergoes a final processing step to ensure
    ## its in the same final order as the input dictonary

    # if we have 1 or more short sequences then we know at least a subset of sequences must be run using the scaled_ network
    if len(short_seqs) > 0:
        # run predictions on each set
        return_dict1 = batch_predict(
            short_seqs,
            batch_size=batch_size,
            network=_SCALED_COUNTERPARTS[network],
            version=version,
            device=device,
            batch_algorithm=batch_algorithm,
            return_seq2prediction=return_seq2prediction,
            show_progress_bar=show_progress_bar,
        )

        # if we also had 1 or more long sequences then run this using the non-scaled network
        if len(long_seqs) > 0:
            # note we have need to set safe=False here or we fall into infinite recursive calls
            return_dict2 = batch_predict(
                long_seqs,
                batch_size=batch_size,
                network=network,
                version=version,
                device=device,
                batch_algorithm=batch_algorithm,
                return_seq2prediction=return_seq2prediction,
                show_progress_bar=show_progress_bar,
                safe=False,
            )

            # merge output and return; recall this would destroy any order of insertion, hence why we use a dictionary instead of a list
            tmp_return_dict = {**return_dict1, **return_dict2}
        else:
            tmp_return_dict = return_dict1
    else:
        # note we have need to set safe=False here or we fall into infinite recursive calls
        tmp_return_dict = batch_predict(
            protein_objs,
            batch_size=batch_size,
            network=network,
            version=version,
            device=device,
            batch_algorithm=batch_algorithm,
            return_seq2prediction=return_seq2prediction,
            show_progress_bar=show_progress_bar,
            safe=False,
        )

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
                raise SparrowException(
                    "Error parsing type of return dictionary. This is a bug, please report https://github.com/idptools/sparrow/issues"
                )

    # if we're returning the standard key:[seq,pred] mapping
    else:
        for k in input_dict:
            return_dict[k] = tmp_return_dict[k]

    return return_dict


# ....................................................................................
#
#
def batch_predict(
    protein_objs,
    network: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    version=None,
    gpuid: int | None = None,
    batch_algorithm=DEFAULT_BATCH_ALGORITHM,
    return_seq2prediction: bool = False,
    show_progress_bar: bool = True,
    safe: bool = True,
    device: str | torch.device | None = DEFAULT_DEVICE,
) -> dict:
    """Perform batch predictions with one of sparrow's networks.

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
    dictionary, or 0-based indices 0,1,...n-1 for list input)
    to a list with two elements;

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

    Each prediction is what the network's predictor makes for one sequence -
    for the ALBATROSS networks, the value ``Protein.predictor`` returns - from
    the network's output for that sequence in a batch, which can differ from
    its output for the sequence alone by about 1e-7 relative, the precision of
    the networks' 32-bit arithmetic.

    Parameters
    ----------
    protein_objs : dict or list
        Either a dictionary of key-value pairs where values are either sequence strings
        or sparrow.protein.Protein objects, or a list, where elements are either sequence
        strings or sparrow.protein.Protein objects.

    network : str
        The name of the network to predict with, by default None. One of
        these, which give one value per sequence (the ALBATROSS networks) or
        one per residue:

        * ``"rg"``, ``"scaled_rg"`` - the radius of gyration, in Angstroms
        * ``"re"``, ``"scaled_re"`` - the end-to-end distance, in Angstroms
        * ``"asphericity"``, ``"prefactor"``, ``"scaling_exponent"``
        * ``"dssp"`` - the probability of helix, extended and coil at each
          residue, an array of shape ``(len(sequence), 3)``
        * ``"transmembrane"`` - the predicted class of each residue
        * ``"mitochondrial_targeting"`` - 1 in a predicted targeting
          sequence, else 0; only the first 168 residues are predicted
        * ``"nes"``, ``"nls"``, ``"tad"`` - the probability that each residue
          is in a nuclear export signal, a nuclear import signal or a
          transactivation domain
        * ``"pscore"`` - the phase-separation propensity of each residue, on
          its native 0-6 scale
        * ``"ser_phosphorylation"``, ``"thr_phosphorylation"``,
          ``"tyr_phosphorylation"`` - the probability that each residue is
          phosphorylated

        The ``Protein.predictor`` methods return these, or values made from
        them: ``pscore()`` rescales to 0-1 by default, and the phosphorylation
        and DSSP methods apply their thresholds.

    batch_size : int
        The batch size to use for network forward pass.
        This should be <= the batch size used during training.
        Default = 32.

    version : int, str or None, optional
        Network version, by default None, the network's default (2 for the
        ALBATROSS networks, released May 21st 2023).

    gpuid : int or None, optional
        The NVIDIA GPU to run on; see ``device``. Default None.

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

        'pad-n-pack' - Pad each mixed-length batch to its longest member and
                       use pack_padded_sequence so the padding does not affect
                       the result. Only available in PyTorch 1.11.0+. Provides
                       true mixed-length batching and is typically much faster
                       when sequence lengths vary; it is the default on modern
                       PyTorch.

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

    device : {'auto', 'cpu', 'cuda', 'mps'} or torch.device, optional
        Where to run the network, as PARROT's ``Predictor`` takes it (see
        :func:`~sparrow.predictors.devices.resolve_device`). Default
        ``'auto'``: CUDA if this machine has it (the GPU ``gpuid`` names, or
        the current one), and the CPU otherwise. ``'auto'`` never chooses an
        Apple-silicon GPU, on which sparrow's networks run several times
        slower than on the CPU, but ``'mps'`` runs on one. None is read as
        ``'auto'``.

    Returns
    -------
    dict
        sequence, value(s) mapping for the requested predictor.

    Raises
    ------
    SparrowException
        An exception is raised if the requested network is not one of the
        available options, or the device asked for is not valid or not on
        this machine.
    """
    from sparrow.predictors import NETWORK_PREDICTORS

    ## ------------------------------------------------------------------------------------
    ##
    ## Sanitize inputs
    ##

    # check a valid network was passed
    if network not in NETWORK_PREDICTORS:
        raise SparrowException(
            f"For option 'network': Please choose a valid network for batch predictions, one of {list(NETWORK_PREDICTORS)}"
        )

    # check a valid batch algorithm and device were passed, before anything is run
    resolve_batch_algorithm(batch_algorithm)
    device = resolve_device(DEFAULT_DEVICE if device is None else device, gpuid)

    # if we've requested to use rg/re predictions, this function ensures short sequences are appropriately
    # dealt with (this is a recurisve function that calls batch_predict)
    if network in _SCALED_COUNTERPARTS and safe:
        return __short_seq_fix(
            protein_objs,
            batch_size=batch_size,
            network=network,
            version=version,
            device=device,
            batch_algorithm=batch_algorithm,
            return_seq2prediction=return_seq2prediction,
            show_progress_bar=show_progress_bar,
        )

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
        raise SparrowException(
            "Invalid input - must pass either a dictionary of key-values where values are sequences (str)\nor sparrow.protein.Protein objects, or a list of sequences (str) or sparrow.protein.Protein objects"
        )

    ## ------------------------------------------------------------------------------------
    ##
    ## Perform prediction
    ##

    # every sequence is run, in input order, as it always has been; a sequence that
    # appears twice gets the prediction from its last appearance
    predictions = predict_many(
        network,
        sequence_list,
        version=version,
        batch_size=batch_size,
        batch_algorithm=batch_algorithm,
        show_progress_bar=show_progress_bar,
        device=device,
    )
    pred_dict = dict(zip(sequence_list, predictions))

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
