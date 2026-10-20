Writing a Plugin
================

.. note::
   This page is a guide to **writing** a plugin. To **use** plugins from a
   ``Protein``, see :doc:`protein/plugins`.

Plugins extend sparrow through ``Protein.plugin``. The plugin manager,
:class:`~sparrow.sequence_analysis.plugins.PluginManager`, discovers the
subclasses of :class:`~sparrow.sequence_analysis.plugins.BasePlugin` in
``sparrow.sequence_analysis.community_plugins.contributed``, and offers each by
its class name.

How a plugin is run
-------------------

1. ``PluginManager`` discovers the plugin classes.
2. The first time a plugin is asked for, it is loaded and given the
   ``Protein``.
3. Calling it runs its ``calculate(*args, **kwargs)``.
4. A repeated call with the same arguments returns the cached result, so a
   plugin should be a pure function of the protein and its arguments.

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

Reference
---------

:mod:`sparrow.sequence_analysis.plugins` and the contributed plugins are
documented in full in :doc:`api/sequence_analysis`.
