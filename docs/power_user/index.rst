Power-User APIs
===============

:class:`~sparrow.protein.Protein` and :func:`sparrow.batch` cover most work.
Underneath them are the functions and classes that do the calculating, and you
can call those directly - when you want arrays rather than ``Protein`` objects,
when you want an option a ``Protein`` method does not expose, or when you are
building something of your own on top of sparrow. Everything here gives the
same answers as the ``Protein`` it sits under, to the last bit.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - When you want
   * - :doc:`properties`
     - Any sequence parameter or linear profile for a whole list of sequences
       at once, as NumPy arrays - a feature matrix for machine learning, say -
       with :func:`sparrow.properties.compute` and the batch functions of
       :mod:`sparrow.properties`.
   * - :doc:`patterning`
     - The patterning statistics - kappa, delta, the patterning percentile,
       IWD, SCD, SHD - for a batch, and how they are computed.
   * - :doc:`networks`
     - To run one of sparrow's networks directly over many sequences
       (:func:`~sparrow.predictors.batch_predictor.batch_predict`), on the CPU
       or a GPU, to pin a network version, or to use the network itself as a
       PyTorch module.
   * - :doc:`nardini`
     - NARDINI+ over a proteome: building compositional backgrounds, choosing a
       patterning backend, assigning GIN grammars and judging the
       assignments, and clustering your own grammars.
   * - :doc:`polymer_models`
     - The polymer models as plain functions of a sequence.
   * - :doc:`annotations`
     - The annotation functions: the motifs a substitution gains or loses,
       PLAAC's full scores and its command line, and the low-complexity domain
       searches.
   * - :doc:`alignment`
     - A multiple sequence alignment of a set of proteins.
   * - :doc:`aaindex`
     - Any of the 569 AAindex amino-acid scales, as values or as a profile.
   * - :doc:`data`
     - The data sparrow ships: residue sets and scales, the pairwise AAindex
       matrices, the network files and the reference data.
   * - :doc:`utilities`
     - Reading sequences from FASTA files and UniProt, comparing sequences, and
       the other small tools.
   * - :doc:`plotting`
     - A figure of one or more proteins along their length: domains,
       per-residue values, heatmaps and sequence, on a shared residue axis.

.. toctree::
   :maxdepth: 1
   :hidden:

   properties
   patterning
   networks
   nardini
   polymer_models
   annotations
   alignment
   aaindex
   data
   utilities
   plotting

``Protein`` or the functions?
-----------------------------

* Use ``Protein`` (or ``sparrow.batch``) for interactive analysis, and whenever
  you want values computed once and cached.
* Use the functions for explicit data flow - arrays in, arrays out - and when a
  ``Protein`` for each sequence would only be thrown away.
* Mix them freely: they share one implementation of every calculation, so a
  value is the same whichever you use.

Every function and class on these pages is documented in full in the
:doc:`../api/index`.
