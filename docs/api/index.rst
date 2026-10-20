API Reference
=============

Every public module, class and function in sparrow, documented from its
docstring, one page for each package. The other sections explain how to use
them; this one is the full detail.

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Page
     - Modules
   * - :doc:`sparrow`
     - :mod:`sparrow` itself: :func:`sparrow.batch`, :func:`sparrow.read_fasta`
       and :func:`sparrow.data_path`, and :mod:`sparrow.batching`.
   * - :doc:`protein`
     - :mod:`sparrow.protein`: the :class:`~sparrow.protein.Protein` class.
   * - :doc:`properties`
     - :mod:`sparrow.properties` and its modules: every sequence parameter and
       profile, for a batch.
   * - :doc:`predictors`
     - :mod:`sparrow.predictors`: the ``Predictor`` accessor, batch prediction,
       devices, and the machinery every predictor shares.
   * - :doc:`predictor_networks`
     - The predictor class of each of sparrow's networks.
   * - :doc:`polymer`
     - :mod:`sparrow.polymer`: the ``Polymeric`` accessor and the polymer
       models.
   * - :doc:`nardini`
     - :mod:`sparrow.nardini`: NARDINI+ vectors, backgrounds, GIN assignment
       and clustering.
   * - :doc:`sequence_analysis`
     - :mod:`sparrow.sequence_analysis`: ELMs, PLAAC, low-complexity domains,
       phosphoisoforms, alignment, compression and plugins.
   * - :doc:`data`
     - :mod:`sparrow.data`: residue sets and scales, AAindex, settings.
   * - :doc:`tools`
     - :mod:`sparrow.tools`: reading and fetching sequences, and utilities.
   * - :doc:`visualize`
     - :mod:`sparrow.visualize`: sequence display.
   * - :doc:`exceptions`
     - :mod:`sparrow.sparrow_exceptions`: the exceptions and warnings sparrow
       raises.

.. toctree::
   :maxdepth: 1
   :hidden:

   sparrow
   protein
   properties
   predictors
   predictor_networks
   polymer
   nardini
   sequence_analysis
   data
   tools
   visualize
   exceptions
