Plugins
=======

.. note::
   This page is a focused guide for plugin usage and contribution.
   For core API workflows, start with :doc:`api`.

Plugins extend Sparrow analyses through ``Protein.plugin``.
The plugin manager discovers subclasses of
:class:`sparrow.sequence_analysis.plugins.BasePlugin` from
``sparrow.sequence_analysis.community_plugins.contributed``.

Using Plugins
-------------

.. code-block:: python

   from sparrow import Protein

   p = Protein("MEEEKKKKSSSTTTDDD")

   # discover available plugin callables
   available = [name for name in dir(p.plugin) if not name.startswith("_")]

   # run a plugin
   value = p.plugin.MultiplicativeFCR(factor=2.5)

Caching and Call Semantics
--------------------------

Plugin calls are memoized by argument signature in ``PluginWrapper`` using
``(args, frozenset(kwargs.items()))``. Plugin argument values must therefore
be hashable.

Plugin lifecycle:

1. ``PluginManager`` discovers valid plugin subclasses.
2. First attribute access lazily loads and instantiates the plugin.
3. Calling the wrapper dispatches to ``calculate(*args, **kwargs)``.
4. Repeated calls with identical arguments return cached results.

Plugin Contribution Tutorial
----------------------------

This tutorial describes the minimum steps to add a community plugin.

Step 1: Create a plugin class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add a subclass of ``BasePlugin`` in:

* ``sparrow/sequence_analysis/community_plugins/contributed.py``

.. code-block:: python

   from sparrow.sequence_analysis.plugins import BasePlugin


   class ExampleMetric(BasePlugin):
       """Return a scaled FCR as a simple plugin example."""

       def calculate(self, scale: float = 1.0) -> float:
           return scale * self.protein.FCR

Step 2: Verify discovery and execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sparrow import Protein

   p = Protein("MEEEKKKKSSSTTTDDD")
   assert "ExampleMetric" in dir(p.plugin)
   result = p.plugin.ExampleMetric(scale=2.0)

Step 3: Add tests
^^^^^^^^^^^^^^^^^

Add focused tests to ``sparrow/tests/test_plugins.py`` for:

* plugin result correctness
* cache behavior for repeated calls
* invalid plugin access behavior (if relevant)

Step 4: Add docs
^^^^^^^^^^^^^^^^

Document plugin purpose, parameters, and return values in class/method docstrings.
Plugin class names are user-facing API names via ``protein.plugin.<ClassName>``.

Reference: Plugin Infrastructure
--------------------------------

Key user entrypoint:

* :attr:`sparrow.protein.Protein.plugin`

.. autoclass:: sparrow.sequence_analysis.plugins.PluginManager
   :no-index:

.. autoclass:: sparrow.sequence_analysis.plugins.BasePlugin
   :no-index:

.. automodule:: sparrow.sequence_analysis.plugins
   :no-index:

Reference: Contributed Plugins Module
-------------------------------------

.. autoclass:: sparrow.sequence_analysis.community_plugins.contributed.MultiplicativeFCR
   :no-index:

.. automodule:: sparrow.sequence_analysis.community_plugins.contributed
   :no-index:
