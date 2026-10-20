Every Analysis ``sparrow.batch`` Can Compute
============================================

:func:`sparrow.batch` takes the name of any property or method of a ``Protein``,
or of its ``predictor``, ``polymeric`` and ``nardini`` accessors, that computes
a value. These pages list every one of them. They are written from sparrow
itself each time the documentation is built, by the same code ``batch`` uses to
check a name, so they list exactly the names ``batch`` accepts.

How to read an entry
--------------------

* **The name** is what you pass to ``batch``. On its own it is computed with
  its defaults; to give it arguments, pair it with them in a dict:
  ``{"charge_at_pH": {"pH": 6.0}}``.
* **Read it back as** is the call that returns the value from each ``Protein``
  afterwards, without computing anything - as long as you call it with the
  arguments you named it with. The defaults are shown.
* **Computed** says how ``batch`` computes it: for the whole batch at once,
  where sparrow has a function that does that, or one protein at a time, by the
  method itself.
* **Arguments** lists each argument, with its type, its default (or that it is
  required, with no default), and what it does. ``recompute`` is left out:
  ``batch`` computes each value once, and refuses it.

.. code-block:: python

   import sparrow

   proteins = sparrow.batch(
       "idrs.fasta",
       [
           "FCR",                                              # a property
           {"charge_at_pH": {"pH": 6.0}},                      # a method, with arguments
           {"low_complexity_domains": {"residue_selector": "Q"}},  # a required argument
           "predictor.radius_of_gyration",                     # an accessor's method, with its defaults
           {"nardini.vector": {"seed": 42}},
       ],
   )

   protein = proteins["IDR1"]
   protein.charge_at_pH(pH=6.0)                     # read back with the same arguments
   protein.low_complexity_domains(residue_selector="Q")

The groups
----------

.. toctree::
   :maxdepth: 1

   analyses_protein_properties
   analyses_protein_methods
   analyses_predictor
   analyses_polymeric
   analyses_nardini

Every name
----------

"For the whole batch at once" is *yes* where ``batch`` computes the value for
every sequence together, *no* where it computes it one protein at a time, and
*what it reads* for the ``polymeric.predicted_*`` values, which are cheap to
compute from an ALBATROSS prediction that is itself predicted for the whole
batch.

.. include:: _analyses_summary.rst
