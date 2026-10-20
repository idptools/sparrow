Plugins (``protein.plugin``)
============================

Plugins are analyses contributed to sparrow that are not part of its core. Each
is a class in ``sparrow.sequence_analysis.community_plugins.contributed``, and
``protein.plugin`` finds them all and offers each by its class name:

.. code-block:: python

   from sparrow import Protein

   p = Protein("MEEEKKKKSSSTTTDDD")

   # what is available
   [name for name in dir(p.plugin) if not name.startswith("_")]

   # run one
   p.plugin.MultiplicativeFCR(factor=2.5)

A plugin is loaded the first time you use it, and each call is cached under its
arguments, so a repeated call with the same ones is free. A call with an
argument that cannot be hashed - a list, a dict or a NumPy array - cannot be
cached, so it is computed afresh every time. Asking for a plugin that does not
exist raises an ``AttributeError`` that lists the ones that do.

:doc:`../writing_a_plugin` explains how to contribute one.

Reference
---------

.. autoproperty:: sparrow.protein.Protein.plugin
