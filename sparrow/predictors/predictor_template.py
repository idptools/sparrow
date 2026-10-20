"""Template for a sparrow predictor built on a network trained with PARROT 2.

To add a predictor called, say, ``my_property``:

1. Train the network with PARROT 2, for example::

       parrot-train brnn --input-file data.tsv --output-network my_property_network_v1.pt \\
           --split kmer -d residues --regression

2. Copy ``my_property_network_v1.pt`` to
   ``sparrow/data/networks/my_property/my_property_network_v1.pt``.
3. Copy this file to ``sparrow/predictors/my_property/my_property_predictor.py``,
   with an empty ``__init__.py`` beside it, and rename ``MyPropertyPredictor``,
   ``predict_my_property`` and the weights path.

That is everything the predictor needs: a PARROT 2 network file records its
architecture, task and sequence encoding, so none of them is declared here.
``docs/adding_a_predictor.rst`` explains how to make the predictor available as
``Protein(seq).predictor.my_property()``.

Loading a network saved by PARROT 2 needs PARROT 2 installed; until it is
released, install it with
:data:`sparrow.predictors.network_loader.PARROT2_INSTALL_COMMAND`.
"""

from sparrow.predictors.base import BaseNetworkPredictor

# Version of the weights loaded when none is asked for. When you train a better
# network, save it as the next version and update this, so the older ones stay
# selectable with MyPropertyPredictor(version="1").
DEFAULT_VERSION = "1"


class MyPropertyPredictor(BaseNetworkPredictor):
    """Loads a network so :meth:`predict_my_property` can predict my property."""

    # Weights file, relative to sparrow/data; {version} is filled in from the
    # constructor's version argument
    NETWORK_PATH = "networks/my_property/my_property_network_v{version}.pt"
    DEFAULT_VERSION = DEFAULT_VERSION

    def predict_my_property(self, seq):
        """Predict my property for a sequence.

        Parameters
        ----------
        seq : str
            Valid amino acid sequence.

        Returns
        -------
        numpy.ndarray
            The prediction, as :meth:`_from_raw` makes it.
        """
        return self.predict(seq)

    def _from_raw(self, seq, raw):
        """The prediction for a sequence, from PARROT's prediction for it.

        ``predict`` and ``predict_many`` both call this, so a sequence's
        prediction is made the same way alone or with many others.

        Parameters
        ----------
        seq : str
            The sequence.
        raw : numpy.ndarray
            PARROT's prediction. For a regression network that is one value per
            residue (a ``residues`` network) or a one-element array (a
            ``sequence`` network); for a classification network, the
            probability of each class, with shape ``(len(seq), num_classes)`` or
            ``(num_classes,)``.

        Returns
        -------
        numpy.ndarray
            PARROT's prediction. Reduce it here to what the predictor should
            return - for example ``float(prediction[0])`` for one value per
            sequence, or ``prediction[:, 1]`` for the probability of class 1 at
            each residue - and say so in this docstring, with units.
        """
        return raw
