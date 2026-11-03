``sparrow.null``
================

.. automodule:: sparrow.null
   :no-members:
   :no-inherited-members:

The four null models can also be imported from :mod:`sparrow.null` itself.
:doc:`../null/index` explains them and shows them in use.

The parent class
----------------

.. automodule:: sparrow.null.base
   :no-members:
   :no-inherited-members:

.. autoclass:: sparrow.null.base.NullModel
   :members: needs_background, fit, generate, is_fitted

The block shuffle
-----------------

.. automodule:: sparrow.null.block_shuffle
   :no-members:
   :no-inherited-members:

.. autoclass:: sparrow.null.block_shuffle.BlockShuffle
   :no-members:

The k-let shuffle
-----------------

.. automodule:: sparrow.null.klet_shuffle
   :no-members:
   :no-inherited-members:

.. autoclass:: sparrow.null.klet_shuffle.KletShuffle
   :no-members:

Background k-mer reconstruction
-------------------------------

.. automodule:: sparrow.null.kmer_background
   :no-members:
   :no-inherited-members:

.. autoclass:: sparrow.null.kmer_background.KmerBackground
   :no-members:

The background Markov chain
---------------------------

.. automodule:: sparrow.null.markov_background
   :no-members:
   :no-inherited-members:

.. autoclass:: sparrow.null.markov_background.MarkovBackground
   :no-members:

.. autodata:: sparrow.null.markov_background.MAX_ORDER

What the models share
---------------------

Every null model has :meth:`~sparrow.null.base.NullModel.fit` and
:meth:`~sparrow.null.base.NullModel.generate` from
:class:`~sparrow.null.base.NullModel`, documented above. Underneath, they share
how a sequence is cut into blocks, how a background's k-mers are counted, and
how a background is tilted to a sequence's composition.

.. automodule:: sparrow.null.blocks
   :members: check_block_sizes, plan_blocks

.. automodule:: sparrow.null.kmers
   :members: count_background_kmers, kmer_residues, weighted_choice, MAX_KMER_SIZE, MIN_MEAN_KMER_COUNT

.. automodule:: sparrow.null.tilt
   :members: fit_tilt, TILT_TOLERANCE

The warning
-----------

The advisory warning the null models give,
:class:`~sparrow.sparrow_exceptions.NullModelWarning`, is documented with
sparrow's exceptions, in :doc:`exceptions`.
