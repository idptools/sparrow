"""Base class shared by all sparrow network predictors.

A concrete predictor only needs to declare *what differs* from the common
pattern: the path to its weights (:attr:`NETWORK_PATH`), the default version
(:attr:`DEFAULT_VERSION`), and the network architecture (:attr:`ARCHITECTURE`).
Loading the weights and running the forward pass are handled here, so no
predictor subclass needs to touch ``torch`` directly.

Subclasses then expose a single ``predict_<thing>(seq, ...)`` method that turns
the raw network output (from :meth:`BaseNetworkPredictor._raw_predict`) into the
desired result, typically using a helper from :mod:`sparrow.predictors.outputs`.
"""

from parrot import encode_sequence

from sparrow.predictors.network_loader import load_parrot_network

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
        Network architecture passed to :func:`load_parrot_network`.
    """

    NETWORK_PATH: str = None
    DEFAULT_VERSION: str = None
    ARCHITECTURE: str = "MtO"

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

        # surface inferred hyper-parameters as instance attributes (these are
        # read by batch_predict and useful for sanity checks)
        self.number_of_classes = meta.number_of_classes
        self.input_size = meta.input_size
        self.number_of_layers = meta.number_of_layers
        self.hidden_vector_size = meta.hidden_vector_size

    def _raw_predict(self, seq):
        """Run the network forward pass and return the raw numpy output.

        Parameters
        ----------
        seq : str
            Amino acid sequence (upper-cased internally).

        Returns
        -------
        numpy.ndarray
            The detached network output array, including the leading batch
            dimension (shape depends on the architecture).
        """
        seq = seq.upper()
        seq_vector = encode_sequence.one_hot(seq)
        seq_vector = seq_vector.view(1, len(seq_vector), -1)
        return self.network(seq_vector.float()).detach().numpy()
