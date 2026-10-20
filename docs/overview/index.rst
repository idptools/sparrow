Overview
========

sparrow (*Sequence PARameters for RegiOns in Windows*) is a Python package for
analyzing and predicting features of protein sequences, with a particular focus
on intrinsically disordered regions (IDRs). It brings together direct sequence
calculations - composition, charge, patterning, hydrophobicity - with
deep-learning predictions trained with `PARROT <https://elifesciences.org/articles/70576>`_,
polymer models, sequence annotations and a native implementation of the
NARDINI+ molecular-grammar analysis.

.. toctree::
   :maxdepth: 1

   organization
   installation

Two ways in
-----------

For **one sequence**, make a :class:`~sparrow.protein.Protein` and read what you
want off it. Nothing is computed until you ask for it, and everything you ask
for is cached, so a ``Protein`` is cheap to make and you only pay for what you
use.

.. code-block:: python

   from sparrow import Protein

   p = Protein("MEEEKKKKSSSTTTDDDQQQQNNNN")
   p.FCR                                   # a property
   p.compute_kappa_x("ED", "KR")           # a method with options
   p.predictor.disorder()                  # a deep-learning prediction
   p.polymeric.predicted_nu                # a polymer property
   p.nardini.vector(seed=42)               # a NARDINI+ vector

For **many sequences** - fifty or fifty thousand - use :func:`sparrow.batch`. It
makes a ``Protein`` for every sequence with the values you name already
computed, using a function that computes each value for the whole batch at once
wherever sparrow has one:

.. code-block:: python

   import sparrow

   proteins = sparrow.batch("idrs.fasta", ["FCR", "kappa", "predictor.radius_of_gyration"])
   proteins["IDR1"].kappa                  # already computed

Everything else in sparrow is what these two are built on, and you can call it
directly when you want arrays rather than ``Protein`` objects, or more control;
see :doc:`../power_user/index`.

What sparrow can do
-------------------

* **Sequence parameters** - composition, molecular weight, charge (FCR, NCPR,
  kappa, SCD, isoelectric point, charge at any pH), hydrophobicity and SHD,
  conformational propensity (chain-expanding and disorder-promoting fractions,
  polyproline II), R/K and E/D ratios, sequence complexity, residue clustering
  (IWD) and patches. See :doc:`../protein/index`.
* **Linear profiles** - per-residue windowed tracks of any of these, and of
  569 published amino-acid scales (AAindex). See :doc:`../protein/profiles`.
* **Deep-learning predictions**, through ``Protein.predictor`` - disorder and
  pLDDT (from metapredict), DSSP secondary structure, polymer dimensions (the
  ALBATROSS networks: Rg, Re, scaling exponent, asphericity, prefactor),
  phosphorylation, nuclear import and export signals, transactivation domains,
  transmembrane regions, mitochondrial targeting and phase-separation
  propensity. See :doc:`../protein/predictor`.
* **Polymer models**, through ``Protein.polymeric`` - the Analytical Flory
  Random Coil, self-avoiding walks and empirical scaling relationships, as
  dimensions and as distance distributions. See :doc:`../protein/polymeric`.
* **Domains and motifs** - low-complexity domains, prion-like domains (a
  pure-Python port of PLAAC), eukaryotic linear motifs (ELM), and
  phosphoisoforms. See :doc:`../protein/annotations`.
* **Molecular grammars** - NARDINI+ feature vectors, with compositional
  backgrounds you can build yourself, assignment to the 30 human GIN grammars,
  clustering to derive your own, and parallel computation across a proteome.
  See :doc:`../protein/nardini` and :doc:`../power_user/nardini`.
* **Scale** - :func:`sparrow.batch` for ``Protein`` objects, and underneath it,
  every sequence parameter for a whole batch at once
  (:mod:`sparrow.properties`), batched network predictions, on the CPU or a
  GPU, and fixed-length feature vectors for machine learning. See
  :doc:`../batch/index`.
* **Extensibility** - a plugin system for adding analyses, and a template for
  adding predictors trained with PARROT. See :doc:`../writing_a_plugin` and
  :doc:`../adding_a_predictor`.

Principles worth knowing
------------------------

* **Lazy and cached.** A ``Protein`` computes a value the first time you ask
  for it and keeps it, under the arguments you asked with, so asking again is
  free and asking with different arguments computes afresh.
* **Each call hands back its own copy.** You can change a list or array a
  ``Protein`` returned without changing what the next call returns.
* **One implementation of each calculation.** Every sequence parameter is
  calculated in one place, :mod:`sparrow.properties`, which ``Protein``,
  :func:`sparrow.batch` and NARDINI+ all call. A value is therefore the same,
  to the last bit, whichever way you reach it, and whether a sequence is
  scored alone or in a batch of a million.
* **Failing loudly.** sparrow raises a
  :class:`~sparrow.sparrow_exceptions.SparrowException` rather than returning a
  value it cannot stand behind - a residue it does not know, an option that
  does not exist, a device the machine does not have - and every exception it
  raises deliberately derives from that one class.

Where to start
--------------

* :doc:`installation` - set up an environment and install sparrow.
* :doc:`../examples` - short walkthroughs that end in a figure or a
  result.
* :doc:`../protein/index` - everything you can do with one sequence.
* :doc:`../batch/index` - the same for many.

Citing sparrow
--------------

If you use sparrow, please cite the original work behind each tool, metric or
predictor you use, as well as sparrow; :doc:`../citing` lists them.
