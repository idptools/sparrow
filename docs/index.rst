sparrow
=======

**sparrow** (*Sequence PARameters for RegiOns in Windows*) is a lightweight,
object-oriented toolkit for analyzing and predicting features of protein
sequences -- with a particular focus on intrinsically disordered regions (IDRs).

What is sparrow?
----------------

Everything in sparrow hangs off a single :class:`~sparrow.protein.Protein`
object: you create one from a sequence and then read parameters, properties, and
predictions directly off it. Calculations are lazy and cached, so a ``Protein``
is cheap to make and you only compute what you ask for.

.. code-block:: python

   from sparrow import Protein

   p = Protein("MEEEKKKKSSSTTTDDDQQQQNNNN")
   p.FCR                          # fraction of charged residues
   p.kappa                        # charge patterning
   p.predictor.disorder()         # per-residue disorder prediction
   p.predictor.radius_of_gyration()   # ALBATROSS Rg

What can it do?
^^^^^^^^^^^^^^^

* **Sequence parameters** -- composition, charge (FCR, NCPR, kappa, SCD),
  hydrophobicity, complexity, residue clustering and patches.
* **Linear profiles** -- per-residue windowed tracks for any of the above, plus
  500+ published amino-acid property scales (AAindex).
* **Deep-learning predictions** (via ``Protein.predictor``) -- disorder and
  pLDDT, DSSP secondary structure, polymer dimensions (Rg, Re, scaling exponent,
  asphericity), phosphorylation, localization signals, transactivation domains,
  transmembrane regions, and phase-separation propensity. These use the
  `ALBATROSS <https://www.nature.com/articles/s41592-023-02159-5>`_ networks and
  related PARROT-trained models.
* **Polymer-model properties** (via ``Protein.polymeric``) -- analytical and
  simulation-derived dimensions and distance distributions.
* **Scale-up tools** -- batch prediction over whole sequence sets, and
  fixed-length feature vectors for machine learning.

Where to start
--------------

* :doc:`installation` -- set up an environment and install sparrow.
* :doc:`examples` -- runnable walkthroughs (including plotting a linear NCPR
  profile).
* :doc:`api_guides/protein` -- **the complete, organized reference for everything
  you can do with a protein.**

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   installation
   examples

.. toctree::
   :maxdepth: 2
   :caption: The Protein Object
   :hidden:

   api_guides/protein

.. toctree::
   :maxdepth: 1
   :caption: More Reference
   :hidden:

   api_guides/batch
   api_guides/properties
   plugins
   api_guides/sequence_analysis
   api_guides/patterning
   predictors

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
