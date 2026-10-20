The Protein Object
==================

:class:`~sparrow.protein.Protein` is the one object you need for a sequence.
Make it from the sequence and read everything off it: sequence parameters,
profiles, annotations, deep-learning predictions, polymer dimensions and
molecular grammar. Each value is computed the first time you ask for it and
cached, so a ``Protein`` is cheap to make and you only pay for what you use.

.. code-block:: python

   from sparrow import Protein

   p = Protein("MGSQSSRSSSQQQQQQQ")
   p.FCR                                  # a property
   p.compute_kappa_x("ED", "KR")          # a method
   p.predictor.disorder()                 # a prediction
   p.polymeric.predicted_nu               # a polymer-model property

For many sequences, :func:`sparrow.batch` makes a ``Protein`` for each, with
the values you name already computed; see :doc:`../batch/index`.

.. toctree::
   :maxdepth: 1
   :hidden:

   composition
   charge
   hydrophobicity
   conformational
   clustering
   profiles
   annotations
   predictor
   polymeric
   nardini
   visualization
   plugins

What a Protein gives you
------------------------

Each kind of value has a page of its own, which explains it, shows it in use,
and documents every property and method in full.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - What it covers
   * - :doc:`composition`
     - The sequence, its length and molecular weight, and the fraction of each
       residue and residue class.
   * - :doc:`charge`
     - FCR, NCPR, the fractions of positive and negative residues, kappa and
       SCD, the isoelectric point and the charge at any pH.
   * - :doc:`hydrophobicity`
     - The mean hydrophobicity and the sequence hydropathy decoration (SHD).
   * - :doc:`conformational`
     - The chain-expanding and disorder-promoting fractions, polyproline II
       propensity, and the R/K and E/D ratios.
   * - :doc:`clustering`
     - SEG complexity, the inverse weighted distance (IWD) of any residues, and
       patch fractions.
   * - :doc:`profiles`
     - Per-residue windowed profiles of a parameter, a composition or any of
       569 published amino-acid scales.
   * - :doc:`annotations`
     - Low-complexity domains, prion-like domains (PLAAC), eukaryotic linear
       motifs (ELMs) and phosphoisoforms.
   * - :doc:`predictor`
     - ``protein.predictor``: disorder, pLDDT, secondary structure, the
       ALBATROSS polymer dimensions, phosphorylation, localization signals,
       transactivation domains, transmembrane regions, mitochondrial targeting
       and phase-separation propensity.
   * - :doc:`polymeric`
     - ``protein.polymeric``: polymer-model dimensions and distance
       distributions, and the ALBATROSS predictions they start from.
   * - :doc:`nardini`
     - ``protein.nardini``: the NARDINI+ feature vector, and assignment to the
       human GIN grammars.
   * - :doc:`visualization`
     - Showing the sequence as colored HTML.
   * - :doc:`plugins`
     - ``protein.plugin``: analyses contributed as plugins.

Creating a protein
------------------

.. code-block:: python

   from sparrow import Protein

   p = Protein("MGSQSSRSSSQQQQQQQ")
   cleaned = Protein("MGSQXU--", validate=True)   # convert non-standard residues

   len(p)            # sequence length
   p.sequence        # the (upper-cased) sequence string

A sequence is stored upper-cased. The calculations accept only the 20 standard
amino acids and raise a :class:`~sparrow.sparrow_exceptions.SparrowException`
naming any other residue; ``validate=True`` converts the common non-standard
ones first, as the class documentation below lists. To read sequences from a
FASTA file, use :func:`sparrow.read_fasta`, which gives a dict of ``Protein``
objects keyed by header.

.. autoclass:: sparrow.protein.Protein
   :no-members:
   :no-inherited-members:
   :no-undoc-members:

How values are computed and cached
----------------------------------

* A property is computed the first time you read it, and a method the first
  time you call it with a given set of arguments. Each result is cached under
  its name and arguments, so asking again is free, and asking with different
  arguments computes afresh - and is cached in turn.
* Each call hands back its own copy of the cached result, so you can change a
  returned list or array without changing what the next call returns.
* The ``predictor``, ``polymeric``, ``nardini`` and ``plugin`` accessors are
  made on first use and reused; a predictor's network is loaded the first time
  it is needed. The prediction methods take ``recompute=True`` to run the
  network again, apart from the three DSSP methods, which read one cached
  prediction with options of their own.
* :func:`sparrow.batch` fills in this same cache for many proteins at once, so
  a value it computed is read back exactly as if the ``Protein`` had computed
  it itself.

.. _protein-shared-with-nardini:

One implementation, shared with NARDINI+
----------------------------------------

Every sequence parameter a ``Protein`` gives is calculated in
:mod:`sparrow.properties`, which takes a batch of sequences; a ``Protein`` is a
batch of one. Most of these parameters are *also* NARDINI+ compositional
features, and NARDINI+ calls the same functions, so a ``Protein`` property and
its corresponding feature are bit-identical by construction, and a test pins
that for every one of them. The same functions compute any of these properties
for a whole list of sequences at once -
``sparrow.properties.compute(sequences, ["FCR", "kappa"])`` gives them by name;
see :doc:`../power_user/properties`.

.. warning::
   The single exception is **polar**. NARDINI+ counts cysteine as polar
   (``'Frac Polar'`` = QNSTGCH); :attr:`~sparrow.protein.Protein.fraction_polar`
   does not (GHNQST). Both conventions are used in the literature, and the bundled
   per-species backgrounds and GIN centroids are defined against the NARDINI one,
   so neither can be quietly changed to match the other. The two differ by exactly
   the cysteine content, and only the residue set differs - the implementation is
   shared like everything else.

Good to know
------------

* ``kappa`` and ``compute_kappa_x`` return ``-1`` when they are undefined: for a
  sequence shorter than the window, or one missing a residue group they need.
* ``hydrophobicity`` is the **mean** over the sequence; for a per-residue track,
  use ``linear_sequence_profile(mode="hydrophobicity")``.
* Positions are 0-indexed throughout, with an exclusive end, so
  ``sequence[start:end]`` is the region a method reports.
* Every exception sparrow raises deliberately derives from
  :class:`~sparrow.sparrow_exceptions.SparrowException`, so one
  ``except SparrowException`` covers all of them.
