Community Plugins
=================

Community plugins extend sequence analysis through the ``Protein.plugin`` accessor.
The plugin system discovers subclasses of :class:`sparrow.sequence_analysis.plugins.BasePlugin`
in ``sparrow.sequence_analysis.community_plugins.contributed``.

Using Plugins
-------------

Discover and execute available plugin methods:

.. code-block:: python

   from sparrow import Protein

   p = Protein("MEEEKKKKSSSTTTDDD")

   available = [name for name in dir(p.plugin) if not name.startswith("_")]
   value = p.plugin.MultiplicativeFCR(factor=2.5)

Caching and Argument Semantics
------------------------------

Plugin calls are memoized by argument signature. The cache key is built from
``(args, frozenset(kwargs.items()))`` in ``PluginWrapper``, so argument values
must be hashable.

Plugin Lifecycle
----------------

1. Discovery: ``PluginManager`` inspects the contributed module for subclasses.
2. Lazy load: plugin class is imported/instantiated on first attribute access.
3. Wrapper call: ``PluginWrapper`` forwards to ``calculate(*args, **kwargs)``.
4. Memoization: repeated calls with the same arguments return cached results.

Authoring a Contributed Plugin
------------------------------

Place contributed implementations in:

* ``sparrow/sequence_analysis/community_plugins/contributed.py``

Minimal template:

.. code-block:: python

   from sparrow.sequence_analysis.plugins import BasePlugin


   class ExampleMetric(BasePlugin):
       """Contributed plugin example.

       Keep class names descriptive because they become plugin attribute names.
       """

       def calculate(self, scale: float = 1.0) -> float:
           return scale * self.protein.FCR

Authoring expectations:

* Keep class names clear and stable (they are user-facing via ``protein.plugin.<Name>``).
* Add docstrings documenting arguments and return types for autodoc output.
* Keep ``calculate`` deterministic for consistent memoization behavior.

Reference: Plugin Infrastructure
--------------------------------

.. automodule:: sparrow.sequence_analysis.plugins
   :no-index:

Reference: Contributed Plugins Module
-------------------------------------

.. automodule:: sparrow.sequence_analysis.community_plugins.contributed
   :no-index:
