"""Base class shared by all sparrow network predictors.

A concrete predictor only needs to declare *what differs* from the common
pattern: the path to its weights (``NETWORK_PATH``) and the default version
(``DEFAULT_VERSION``), plus, for a network saved by PARROT 1, its architecture
(``ARCHITECTURE``). Loading the weights and running the forward pass are handled
here, so no predictor subclass needs to touch ``torch`` directly.

Subclasses then expose a single ``predict_<thing>(seq, ...)`` method that turns
the network's output into the desired result:

* for a network saved by PARROT 1 (every network sparrow ships), the raw output
  of ``BaseNetworkPredictor._raw_predict``, typically through a helper from
  :mod:`sparrow.predictors.outputs`;
* for a network saved by PARROT 2, PARROT's own prediction, from
  ``BaseNetworkPredictor._predict`` (see ``sparrow/predictors/predictor_template.py``).
"""

from typing import ClassVar

from parrot import encode_sequence

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

    def __init__(self, version=None):
        """Load the versioned network for this predictor.

        Parameters
        ----------
        version : str or None, optional
            Specific network version to load. When ``None`` (default), uses
            :attr:`DEFAULT_VERSION`. The value is substituted into
            :attr:`NETWORK_PATH` (no leading ``"v"`` or ``.pt`` extension).
        """
        if version is None:
            version = self.DEFAULT_VERSION

        relative_path = self.NETWORK_PATH.format(version=version)
        self.network, meta = load_parrot_network(relative_path, self.ARCHITECTURE)

        # which of the two prediction methods below applies to this network
        self.saved_by_parrot2 = meta.saved_by_parrot2

        # surface inferred hyper-parameters as instance attributes (these are
        # read by batch_predict and useful for sanity checks)
        self.number_of_classes = meta.number_of_classes
        self.input_size = meta.input_size
        self.number_of_layers = meta.number_of_layers
        self.hidden_vector_size = meta.hidden_vector_size

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
            input; use :meth:`_predict` for it.
        """
        if self.saved_by_parrot2:
            raise SparrowException(
                f"{type(self).__name__} loads a network saved by PARROT 2: use _predict(seq), which encodes the "
                "sequence as the network was trained to expect"
            )
        seq = seq.upper()
        seq_vector = encode_sequence.one_hot(seq)
        seq_vector = seq_vector.view(1, len(seq_vector), -1)
        return self.network(seq_vector.float()).detach().numpy()

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
            If the network was saved by PARROT 1; use :meth:`_raw_predict` for it.
        """
        if not self.saved_by_parrot2:
            raise SparrowException(
                f"{type(self).__name__} loads a network saved by PARROT 1: use _raw_predict(seq), which returns "
                "its raw output"
            )
        return self.network.predict(seq)
