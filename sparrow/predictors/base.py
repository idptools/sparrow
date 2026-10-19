"""Base class shared by all sparrow network predictors.

A concrete predictor only needs to declare *what differs* from the common
pattern: the path to its weights (``NETWORK_PATH``) and the default version
(``DEFAULT_VERSION``), plus, for a network saved by PARROT 1, its architecture
(``ARCHITECTURE``). Loading the weights and running the forward pass are handled
here, so no predictor subclass needs to touch ``torch`` directly.

Each predictor says how the network's output for a sequence becomes its
prediction, in ``_from_raw``. That one transform serves both
:meth:`BaseNetworkPredictor.predict`, for one sequence, and
:meth:`BaseNetworkPredictor.predict_many`, for many, which runs the network in
batches. Subclasses also expose a ``predict_<thing>(seq, ...)`` method by name.
Whether a network was trained with PARROT 1 or PARROT 2 changes only what
``_from_raw`` is given, never how a predictor is called:

* For a network saved by PARROT 1 (every network sparrow ships), ``_from_raw``
  is given the network's raw output, typically reduced with a helper from
  :mod:`sparrow.predictors.outputs`, and ``predict_many`` runs the network
  with sparrow's batch harness,
  :func:`~sparrow.predictors.batch_predictor.run_network`.
* For a network saved by PARROT 2, ``_from_raw`` is given PARROT's own
  prediction (see ``sparrow/predictors/predictor_template.py``), and
  ``predict_many`` runs the network with PARROT's ``Predictor.predict_batch``.
"""

from typing import ClassVar

from parrot import encode_sequence

from sparrow.predictors.batch_predictor import (
    DEFAULT_BATCH_ALGORITHM,
    DEFAULT_BATCH_SIZE,
    resolve_batch_algorithm,
    run_network,
)
from sparrow.predictors.devices import full_float32_precision, resolve_device
from sparrow.predictors.network_loader import load_parrot_network
from sparrow.sparrow_exceptions import SparrowException

__all__ = ["BaseNetworkPredictor"]


class BaseNetworkPredictor:
    """Common loading and forward-pass machinery for network predictors.

    Class attributes
    ----------------
    NETWORK_PATH : str
        ``str.format``-style template for the weights file relative to the
        sparrow ``data`` directory, containing a single ``{version}`` field,
        e.g. ``"networks/rg/rg_network_v{version}.pt"``.
    DEFAULT_VERSION : str
        Version string used when none is supplied to the constructor.
    ARCHITECTURE : {"MtO", "MtM"}
        Architecture of a network saved by PARROT 1, passed to
        :func:`~sparrow.predictors.network_loader.load_parrot_network`. A
        network saved by PARROT 2 records its own, so this is ignored for it.
    """

    # Declared without a value: every subclass must supply them, and an
    # annotation-only class attribute makes a missing one an AttributeError naming
    # the attribute, rather than a None that fails later inside .format().
    NETWORK_PATH: ClassVar[str]
    DEFAULT_VERSION: ClassVar[str]
    ARCHITECTURE: ClassVar[str] = "MtO"

    def __init__(self, version=None, device=None):
        """Load the versioned network for this predictor.

        Parameters
        ----------
        version : str or None, optional
            Specific network version to load. When ``None`` (default), uses
            :attr:`DEFAULT_VERSION`. The value is substituted into
            :attr:`NETWORK_PATH` (no leading ``"v"`` or ``.pt`` extension).
        device : {'auto', 'cpu', 'cuda', 'mps'}, torch.device, str or None, optional
            The device to load the network onto and predict on: one of the
            choices :func:`~sparrow.predictors.devices.resolve_device` takes,
            as the batch functions take them, or a device as torch names it,
            such as ``'cuda:1'``. Default None, the CPU.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If the device is not valid or not on this machine, or the network
            cannot be loaded (see
            :func:`~sparrow.predictors.network_loader.load_parrot_network`).
        """
        if version is None:
            version = self.DEFAULT_VERSION

        device = resolve_device("cpu" if device is None else device)
        relative_path = self.NETWORK_PATH.format(version=version)
        self.network, meta = load_parrot_network(relative_path, self.ARCHITECTURE, device=device)

        # which of the two prediction methods below applies to this network
        self.saved_by_parrot2 = meta.saved_by_parrot2
        self.device = meta.device

        # The loader copies the weights into a network built on the CPU, so the
        # network is moved to the device here (a no-op on the CPU). A network is
        # only ever used to predict, so it is also put in evaluation mode, in
        # which dropout is off. Every network sparrow ships has a dropout
        # probability of 0, so this changes no prediction; a network with dropout
        # would otherwise predict differently on every call.
        if not self.saved_by_parrot2:
            self.network.to(self.device)
            self.network.eval()

        # surface inferred hyper-parameters as instance attributes (these are
        # useful for sanity checks)
        self.number_of_classes = meta.number_of_classes
        self.input_size = meta.input_size
        self.number_of_layers = meta.number_of_layers
        self.hidden_vector_size = meta.hidden_vector_size

    def _network_input(self, seq):
        """The sequence as the network reads it.

        Upper-cased. A predictor whose network reads only part of a sequence
        overrides this.

        Parameters
        ----------
        seq : str
            Amino acid sequence.

        Returns
        -------
        str
            What the network is run on.
        """
        return seq.upper()

    def _from_raw(self, seq, raw):
        """This predictor's prediction for a sequence, from the network's output for it.

        Every predictor implements this, and :meth:`predict` and
        :meth:`predict_many` both call it, so a sequence's prediction is made
        the same way alone or in a batch.

        Parameters
        ----------
        seq : str
            The sequence, as given (before ``_network_input``).
        raw : numpy.ndarray
            For a network saved by PARROT 1, its raw output for the sequence,
            with a leading batch dimension of 1: ``(1, number_of_classes)`` for a
            many-to-one network and ``(1, len(input), number_of_classes)`` for a
            many-to-many one. For a network saved by PARROT 2, PARROT's
            prediction, as its ``Predictor.predict`` and ``predict_batch``
            return it.

        Returns
        -------
        Any
            The prediction.

        Raises
        ------
        NotImplementedError
            If the predictor does not implement it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not say how its network's output becomes a prediction")

    def predict(self, seq):
        """The prediction for one sequence.

        Parameters
        ----------
        seq : str
            Valid amino acid sequence.

        Returns
        -------
        Any
            The prediction, as ``_from_raw`` makes it.
        """
        if self.saved_by_parrot2:
            return self._from_raw(seq, self._predict(self._network_input(seq)))
        return self._from_raw(seq, self._raw_predict(self._network_input(seq)))

    def predict_many(
        self,
        sequences,
        batch_size=DEFAULT_BATCH_SIZE,
        batch_algorithm=DEFAULT_BATCH_ALGORITHM,
        show_progress_bar=False,
    ):
        """The prediction for each of many sequences, running the network in batches.

        Each prediction is made by ``_from_raw``, as :meth:`predict` makes
        it, from the network's output for that sequence in a batch. That output
        can differ from the output for the sequence alone by about 1e-7
        relative, the precision of the networks' 32-bit arithmetic, because a
        batch is summed in a different order. A network saved by PARROT 1 is
        run by sparrow's batch harness and one saved by PARROT 2 by PARROT's
        ``Predictor.predict_batch``, which batches sequences of similar length
        together and masks the padding.

        Parameters
        ----------
        sequences : list of str
            Valid amino acid sequences.
        batch_size : int, optional
            Sequences per forward pass. Default 32.
        batch_algorithm : {'default', 'pad-n-pack', 'size-collect'}, optional
            How the sequences of a network saved by PARROT 1 are batched; see
            :func:`~sparrow.predictors.batch_predictor.run_network`. Default
            ``'default'``. PARROT batches those of a network saved by PARROT 2
            its own way, so there it is only checked.
        show_progress_bar : bool, optional
            Show progress over the batches. Default False.

        Returns
        -------
        list
            One prediction per sequence, in order.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If the batch algorithm is not valid.
        """
        network_inputs = [self._network_input(seq) for seq in sequences]

        if self.saved_by_parrot2:
            # checked so a bad value fails the same way whichever PARROT saved the network
            resolve_batch_algorithm(batch_algorithm)
            outputs = self.network.predict_batch(
                network_inputs, batch_size=batch_size, show_progress=show_progress_bar
            )
        else:
            outputs = run_network(
                self.network,
                network_inputs,
                architecture=self.ARCHITECTURE,
                device=self.device,
                batch_size=batch_size,
                batch_algorithm=batch_algorithm,
                show_progress_bar=show_progress_bar,
            )
        return [self._from_raw(seq, output) for seq, output in zip(sequences, outputs)]

    def _raw_predict(self, seq):
        """Run a PARROT 1 network forward pass and return the raw numpy output.

        Parameters
        ----------
        seq : str
            Amino acid sequence (upper-cased internally).

        Returns
        -------
        numpy.ndarray
            The detached network output array, including the leading batch
            dimension (shape depends on the architecture).

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If the network was saved by PARROT 2, which may not take one-hot
            input; use ``_predict`` for it.
        """
        if self.saved_by_parrot2:
            raise SparrowException(
                f"{type(self).__name__} loads a network saved by PARROT 2: use _predict(seq), which encodes the "
                "sequence as the network was trained to expect"
            )
        seq = seq.upper()
        seq_vector = encode_sequence.one_hot(seq)
        seq_vector = seq_vector.view(1, len(seq_vector), -1)
        # to the network's device and back, both no-ops on the CPU, and on an
        # NVIDIA GPU in full float32 precision, as run_network runs it
        with full_float32_precision():
            return self.network(seq_vector.float().to(self.device)).detach().cpu().numpy()

    def _predict(self, seq):
        """Predict with a PARROT 2 network, as PARROT's own ``Predictor`` does.

        Parameters
        ----------
        seq : str
            Amino acid sequence (upper-cased by PARROT).

        Returns
        -------
        numpy.ndarray
            For a regression network, one value per residue (a ``residues``
            network) or a one-element array (a ``sequence`` network). For a
            classification network, the probability of each class: an array of
            shape ``(len(seq), num_classes)`` or ``(num_classes,)``.

        Raises
        ------
        sparrow.sparrow_exceptions.SparrowException
            If the network was saved by PARROT 1; use ``_raw_predict`` for it.
        """
        if not self.saved_by_parrot2:
            raise SparrowException(
                f"{type(self).__name__} loads a network saved by PARROT 1: use _raw_predict(seq), which returns "
                "its raw output"
            )
        return self.network.predict(seq)
