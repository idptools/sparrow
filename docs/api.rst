API Documentation
=================

The API docs are organized around common workflows instead of a flat module list.
Start with the guide that matches how you want to use sparrow, then drill into
module/class references from each page.

.. toctree::
   :maxdepth: 2
   :caption: API Guides

   api_guides/protein
   api_guides/sequence_analysis
   api_guides/patterning

Choosing a Workflow
-------------------

.. list-table::
   :header-rows: 1

   * - Workflow
     - Start Here
     - Best For
     - Primary Modules
   * - Object-oriented analysis
     - :doc:`api_guides/protein`
     - Most routine sequence analysis directly from a ``Protein`` object.
     - ``sparrow.protein``
   * - Functional analysis
     - :doc:`api_guides/sequence_analysis`
     - Pipeline code and utilities that should run outside the ``Protein`` wrapper.
     - ``sparrow.calculate_parameters``, ``sparrow.sequence_analysis.alignment``, ``sparrow.sequence_analysis.grammar``, ``sparrow.sequence_analysis.patching``
   * - Extension and community plugins
     - :doc:`plugins`
     - Using and authoring contributed analyses exposed via ``Protein.plugin``.
     - ``sparrow.sequence_analysis.plugins``
   * - Polymeric model workflows
     - :doc:`polymeric`
     - Sequence-level polymer scaling, dimensions, and distribution-based estimates via ``Protein.polymeric``.
     - ``sparrow.polymer``
   * - Patterning and Cython-backed metrics
     - :doc:`api_guides/patterning`
     - Direct access to high-performance sequence patterning functions.
     - ``sparrow.patterning.kappa``, ``sparrow.patterning.iwd``, ``sparrow.patterning.patterning``

Focused Guides
--------------

Detailed implementation-focused guides:

* :doc:`predictors` for predictor training and integration.
* :doc:`polymeric` for polymeric property workflows and model caveats.
* :doc:`plugins` for plugin usage and contribution workflow.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
