How sparrow is organized
========================

sparrow has three layers. At the bottom are the calculations, each a plain
function or class that takes sequences. Above them, a
:class:`~sparrow.protein.Protein` holds one sequence and calls those
calculations for it, caching what they return. At the top,
:func:`sparrow.batch` fills in many ``Protein`` objects at once, by calling the
calculations for the whole batch.

.. code-block:: text

   sparrow.batch(sequences, names)           many Protein objects, values filled in
          |
   Protein(sequence)                          one sequence, every value cached
     .predictor  .polymeric  .nardini  .plugin  .visualize
          |
   sparrow.properties    sparrow.predictors    sparrow.polymer
   sparrow.nardini       sparrow.sequence_analysis    sparrow.null
                                              the calculations, which take sequences
          |
   sparrow.data                               bundled scales, networks and reference data

You can use any layer directly. Most work needs only the top two, which the
:doc:`../protein/index` and :doc:`../batch/index` sections cover; the
:doc:`../power_user/index` section covers the calculations.

The packages
------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Package
     - Holds
   * - :mod:`sparrow.protein`
     - :class:`~sparrow.protein.Protein`, the object everything hangs off.
   * - :mod:`sparrow.batching`
     - :func:`sparrow.batch`, many sequences in, ``Protein`` objects out.
   * - :mod:`sparrow.properties`
     - Every sequence parameter and linear profile, each a function that takes a
       batch of sequences and returns one value per sequence. ``Protein``
       calls these with a batch of one.
   * - :mod:`sparrow.predictors`
     - The deep-learning predictors. :class:`~sparrow.predictors.Predictor` is
       ``Protein.predictor``; each network has a predictor class in its own
       subpackage, and :mod:`~sparrow.predictors.batch_predictor` runs any of
       them over many sequences at once.
   * - :mod:`sparrow.polymer`
     - The polymer models, each a module of functions, and
       :class:`~sparrow.polymer.Polymeric`, which is ``Protein.polymeric``.
   * - :mod:`sparrow.nardini`
     - NARDINI+: feature vectors, compositional backgrounds, GIN cluster
       assignment, clustering and parallel computation.
       :class:`~sparrow.nardini.protein_interface.ProteinNardini` is
       ``Protein.nardini``.
   * - :mod:`sparrow.sequence_analysis`
     - Sequence annotations - ELMs, PLAAC, low-complexity domains,
       phosphoisoforms - multiple sequence alignment, and the plugin system,
       :class:`~sparrow.sequence_analysis.plugins.PluginManager`, which is
       ``Protein.plugin``.
   * - :mod:`sparrow.null`
     - Null models: shuffles, and sequences rebuilt from a background's
       k-mers or Markov chain, each with a sequence's composition, behind one
       parent class, :class:`~sparrow.null.base.NullModel`. See
       :doc:`../null/index`.
   * - :mod:`sparrow.data`
     - The data sparrow ships: amino-acid scales and residue sets, the AAindex
       databases, the trained networks, the ELM classes, and the NARDINI+
       backgrounds and GIN centroids.
   * - :mod:`sparrow.tools`
     - Reading sequences (:func:`sparrow.read_fasta`, ``uniprot_fetch``) and
       small utilities.
   * - :mod:`sparrow.visualize`
     - Sequence display (``Protein.show_sequence``), and plots of proteins
       along their length (:mod:`sparrow.visualize.proteinvis`).
       :class:`~sparrow.visualize.protein_visualizer.ProteinVisualizer` is
       ``Protein.visualize``, which draws them from a protein's own analyses.
   * - :mod:`sparrow.sparrow_exceptions`
     - :class:`~sparrow.sparrow_exceptions.SparrowException`, which every
       exception sparrow raises deliberately derives from, and the warnings
       NARDINI+ and the null models give.

How a value is computed
-----------------------

**A sequence parameter.** ``p.FCR`` calls
:func:`sparrow.properties.composition.fraction_charged` on the sequence's
residue counts, which the ``Protein`` computes once and shares with every
composition property. The value is cached under the property's name, so
reading it again does not recompute it. A method, such as
``p.charge_at_pH(6.0)``, is cached under its name and the arguments it was
called with, so a different ``pH`` is computed afresh, and cached in turn.
NARDINI+ calls the same functions, which is why a ``Protein`` property and the
NARDINI+ feature of the same name agree to the last bit.

**A prediction.** ``p.predictor`` is made on first use. Each prediction method
reads a network's prediction, which the network's predictor class makes: the
class loads the trained network from ``sparrow/data/networks/`` the first time
it is needed, runs it with PARROT, and turns its output into the prediction.
A network runs once for a sequence however many of its methods are asked - the
three DSSP methods share one run, for instance. Disorder and pLDDT come from
`metapredict <https://github.com/idptools/metapredict>`_ rather than a network
of sparrow's own.

**A polymer property.** ``p.polymeric`` calls the polymer models in
:mod:`sparrow.polymer` (the AFRC model comes from the
`afrc <https://github.com/idptools/afrc>`_ package), and its ``predicted_*``
properties read the ALBATROSS predictions from ``p.predictor``.

**A NARDINI+ vector.** ``p.nardini.vector()`` scrambles the sequence, scores
its patterning against its scrambles, and its composition against a reference
set of IDRs - the human IDRome by default - using :mod:`sparrow.nardini`.

**Many sequences.** :func:`sparrow.batch` reads the names you give it, checks
them all before computing anything, and then computes each value for the
whole batch where sparrow can: the scalar properties with
:func:`sparrow.properties.compute`, every network's predictions in batched
forward passes, metapredict's with its own batch functions, and the NARDINI+
scrambles in several processes. It stores each result in its ``Protein``
exactly where that ``Protein`` would have cached it, so reading it back
computes nothing. Every other value is computed one ``Protein`` at a time.

**A plot.** ``p.visualize.show_protein(...)`` reads the analyses it is given
by the names :func:`sparrow.batch` uses, with the same code, and has the
``Protein`` compute each one, so each is cached as if you had called it
yourself. :mod:`sparrow.visualize.protein_visualizer` turns what each returns
into a track - a line, boxes or sites - beside the tracks for any values,
sites and domains of your own, and :mod:`sparrow.visualize.proteinvis` draws
them. Each is a package whose code is in private modules by what it does:
in ``proteinvis``, the look of a figure, the checks of its arguments, the
features and tracks it is made of, drawing each track, laying out the
sequence, and laying out the whole figure; in ``protein_visualizer``, the
class, reading what it is given, and turning sparrow's analyses into tracks.
Everything public is imported from the package itself.

What sparrow depends on
-----------------------

* `NumPy <https://numpy.org>`_ and `SciPy <https://scipy.org>`_ for the
  calculations; sparrow is pure Python, with nothing to compile.
* `PARROT <https://github.com/idptools/parrot>`_ (``idptools-parrot``), and
  through it `PyTorch <https://pytorch.org>`_, to run the trained networks.
* `metapredict <https://github.com/idptools/metapredict>`_ for disorder and
  pLDDT.
* `afrc <https://github.com/idptools/afrc>`_ for the Analytical Flory Random
  Coil.
* `protfasta <https://github.com/idptools/protfasta>`_ to read and write FASTA
  files.
* `pyFAMSA <https://github.com/althonos/pyfamsa>`_, only with the
  ``alignment`` extra, for multiple sequence alignment (see
  :doc:`installation`).

Importing sparrow imports none of PyTorch, metapredict or the networks: each is
imported the first time something needs it.
