Many Sequences (``sparrow.batch``)
==================================

For one sequence, make a :class:`~sparrow.protein.Protein`. For many - fifty or
fifty thousand - use :func:`sparrow.batch`: it makes a ``Protein`` for every
sequence, with the values you name already computed, so reading them back
costs nothing. Everything you do afterwards is the ordinary ``Protein`` API
(:doc:`../protein/index`), and any value you did not name is computed on demand, as for
any ``Protein``.

Where sparrow can compute a value for a whole batch at once, ``batch`` does,
so you get the speed of the batch functions without having to use them.

:doc:`analyses` lists every name ``batch`` accepts, with its arguments and their
defaults, what it returns, and how ``batch`` computes it.

.. toctree::
   :maxdepth: 2
   :hidden:

   analyses

Quick start
-----------

.. code-block:: python

   import sparrow

   proteins = sparrow.batch("idrs.fasta", ["FCR", "kappa", "predictor.radius_of_gyration"])

   proteins["IDR1"].FCR                              # already computed
   proteins["IDR1"].predictor.radius_of_gyration()   # already computed
   proteins["IDR1"].predictor.disorder()             # computed now, as usual

Naming what you want
--------------------

A name is the path you would read the value back by: a ``Protein`` property or
method (``"FCR"``, ``"charge_at_pH"``, ``"linear_sequence_profile"``), or one of
the ``predictor``, ``polymeric`` and ``nardini`` accessors followed by its
method (``"predictor.disorder"``, ``"polymeric.mean_afrc_radius_of_gyration"``,
``"nardini.gin_cluster"``). Every property and method that computes something
can be named; :doc:`analyses` lists them all.

A name on its own is computed with its method's defaults. To give a name
arguments, pair it with them in a dict; a list can mix plain names and such
dicts:

.. code-block:: python

   proteins = sparrow.batch(
       "idrs.fasta",
       [
           "FCR",
           {"charge_at_pH": {"pH": 6.0}},
           {"linear_sequence_profile": {"mode": "NCPR", "window_size": 9}},
           {"nardini.vector": {"seed": 42, "num_scrambles": 10000}},
       ],
       n_processes=8,
       show_progress=True,
   )

   p = proteins["IDR1"]
   p.charge_at_pH(pH=6.0)                         # already computed
   p.nardini.vector(seed=42, num_scrambles=10000) # already computed
   p.charge_at_pH(pH=7.0)                         # computed now

``names`` can also be one dict, in which a name mapped to ``None`` uses its
defaults: ``{"FCR": None, "charge_at_pH": {"pH": 6.0}}`` asks for the same
values as ``["FCR", {"charge_at_pH": {"pH": 6.0}}]``. A name can be asked for
with several sets of arguments - ``[{"charge_at_pH": {"pH": 6.0}},
{"charge_at_pH": {"pH": 7.0}}]`` - and each is computed.

A value is free to read with the same arguments it was computed with; with any
others, it is computed then, and cached in turn. Every name and its arguments
are checked before anything is computed, so a typo fails at once - with a
suggestion, for a name - rather than an hour into a run.

What goes in, and what comes out
--------------------------------

``sequences`` can be:

* a **dict** of amino acid strings or ``Protein`` objects - you get a dict back,
  with the same keys in the same order;
* a **list** (or any iterable) of them - you get a list back, in the same order;
* the **path to a FASTA file** - read with ``sparrow.read_fasta``, and you
  get a dict keyed by header. To pass ``read_fasta`` options (for example
  ``invalid_sequence_action="convert"``), call it yourself and pass the dict.

A ``Protein`` you pass in is filled in and returned, not copied. A single
sequence is not accepted, because a single string is read as a path; for one
sequence, use ``Protein(sequence)``.

What is computed for the whole batch
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Names
     - Computed with
     - On the yeast proteome
   * - The 20 scalar properties (``"FCR"``, ``"kappa"``, ``"SCD"``, ...; see
       :data:`~sparrow.properties.PROPERTY_NAMES`)
     - :func:`sparrow.properties.compute`
     - All 20 for 6,067 proteins in 0.46 s, against 2.8 s one ``Protein`` at a
       time
   * - Every network prediction: ``predictor.radius_of_gyration``,
       ``end_to_end_distance``, ``asphericity``, ``prefactor`` and
       ``scaling_exponent`` (and the ``polymeric.predicted_*`` values that
       read them), the three DSSP methods, ``transmembrane_regions``,
       ``mitochondrial_targeting_sequence``, ``nuclear_export_signal``,
       ``nuclear_import_signal``, ``transactivation_domains``, ``pscore`` and
       the three phosphorylation methods
     - The batch harness, :func:`~sparrow.predictors.batch_predictor.run_network`, once
       for each network, whatever its methods are asked
     - The 5 ALBATROSS values for 2,000 proteins in 28 s on a CPU, against
       227 s
   * - ``predictor.disorder``, ``disorder_domains``, ``binary_disorder`` and
       ``pLDDT``
     - metapredict's own batch functions, once for each set of options
     - All 21 deep-learning predictions, these and the networks', for 300
       proteins in 10.8 s, against 103 s
   * - ``nardini.vector``, ``patterning``, ``gin_cluster``, ``gin_report``
     - :func:`~sparrow.nardini.batch.nardini_proteins`, in ``n_processes``
       processes
     - Scrambled once for each set of options; the GIN methods share the
       default vector's scrambles

Every other value is computed one protein at a time, by the method itself, so
the call is the same if it gains a batch function later; none of them runs a
network. ``show_progress=True`` shows a progress bar for each, and for each
network.

How exact the values are
------------------------

The scalar properties and the NARDINI values are the ones each ``Protein``
computes alone, to the last bit. A deep-learning prediction made in a batch
can differ from one made for the sequence alone at the networks' 32-bit
precision, because a batch is summed in a different order: about 1e-7
relative for the ALBATROSS values, and one place in the last rounded decimal
for the per-residue probabilities and metapredict's scores (see
:doc:`../power_user/networks` for the measurements). A prediction that a ``Protein`` you pass in
has already made is kept, so a value you have already read does not change.

Options
-------

* ``n_processes`` -- worker processes for the NARDINI scrambles (default: every
  core). Nothing else runs in parallel.
* ``show_progress`` -- progress bars for each ALBATROSS network, the NARDINI
  scrambles, and each value computed one protein at a time.
* ``batch_size`` -- sequences per forward pass, for the networks and for
  metapredict, if given.
* ``device`` and ``gpuid`` -- where to run the networks, sparrow's and
  metapredict's, as :func:`~sparrow.predictors.batch_predictor.batch_predict`
  takes them (:doc:`../power_user/networks`): ``device="auto"`` (the default) runs sparrow's
  networks on an NVIDIA GPU where there is one and on the CPU otherwise, and
  leaves metapredict to choose for itself unless ``gpuid`` is given; any other
  device is used by both.

When you want arrays instead
----------------------------

``sparrow.batch`` returns ``Protein`` objects. The functions underneath it
return arrays, which is what a feature matrix for machine learning, or a
clustering, wants:

* :func:`sparrow.properties.compute` -- any of the scalar properties as a dict
  of arrays (:doc:`../power_user/properties`);
* :func:`~sparrow.predictors.batch_predictor.batch_predict` -- any one of
  sparrow's networks over a set of sequences (:doc:`../power_user/networks`);
* :func:`~sparrow.nardini.vectors.nardini_vectors` -- NARDINI+ vectors as a
  labelled matrix, with GIN assignment and clustering (:doc:`../power_user/nardini`).

Reference
---------

.. autofunction:: sparrow.batch
