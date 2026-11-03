Null Models
===========

A null model makes sequences that match a natural sequence in what you choose
to keep - its composition, its pairs of neighbouring residues, the local
grammar of a set of natural sequences - and are random in everything else, so
whatever evolution or function wrote into the order of its residues is gone.
Those sequences are the sequence's *nulls*. Comparing a feature of a sequence
with the same feature of its nulls asks whether the feature is more than what
was kept would give by itself.

:mod:`sparrow.null` has four null models, which differ in what they keep. Two
are shuffles, made from the sequence alone; two learn from a *background*, a
large set of natural sequences such as a proteome or its IDRs, and rebuild the
sequence from the background's grammar.

.. code-block:: python

   from sparrow.null import BlockShuffle, KletShuffle, KmerBackground, MarkovBackground

   sequence = "MEEEKKKKSSSTTTDDDQQQQEKEKRRRDDDESSGGKKKEEE"

   BlockShuffle(k=1).generate(sequence, count=100, seed=1)        # plain shuffles
   KletShuffle(k=2).generate(sequence, count=100, seed=1)         # every dipeptide kept

   chain = MarkovBackground(order=2).fit("yeast_proteome.fasta")
   chain.generate(sequence, count=100, seed=1)                    # the yeast proteome's grammar

Every null has exactly the length and composition of the sequence it was made
from, whichever model made it.

The notebook ``demos/null_models/null_models.ipynb``, in sparrow's repository,
works through every model on yeast IDRs: the sequences each one makes, what
they keep and what they don't, measured, and how the choice of null changes
how unusual a sequence looks.

.. toctree::
   :maxdepth: 1
   :hidden:

   using
   block_shuffle
   klet_shuffle
   backgrounds
   kmer_background
   markov_background
   composition
   writing_a_null_model

The four null models
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 12 18

   * - Null model
     - Keeps
     - Background
     - Option (default)
   * - :doc:`BlockShuffle <block_shuffle>`
     - The composition, and the neighbours each residue has inside its block
       of ``k`` residues; the blocks are put in a random order. ``k=1`` is a
       plain shuffle.
     - None
     - ``k`` (1): a block size, or several
   * - :doc:`KletShuffle <klet_shuffle>`
     - The count of every run of ``k`` residues - every dipeptide, with
       ``k=2`` - and so the composition, with no residues carried around as
       rigid blocks.
     - None
     - ``k`` (2)
   * - :doc:`KmerBackground <kmer_background>`
     - The composition, in blocks of ``k`` drawn from the background's k-mers,
       so with the background's local grammar in place of the sequence's own.
     - Needed
     - ``k`` (3): a block size up to 4, or several
   * - :doc:`MarkovBackground <markov_background>`
     - The composition, drawn a residue at a time from the background's
       Markov chain of order ``order``: the background's grammar again, with
       no joins between blocks.
     - Needed
     - ``order`` (2): from 0 to 3

What these pages cover
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - What it covers
   * - :doc:`using`
     - What every null model shares: making nulls, the seed and how it makes
       nulls reproducible, comparing a sequence with its nulls, and scoring
       nulls with the rest of sparrow.
   * - :doc:`block_shuffle`
     - :class:`~sparrow.null.block_shuffle.BlockShuffle`: how a sequence is cut
       into blocks of one size or several, how many neighbours each keeps, and
       why the cuts start at a random point.
   * - :doc:`klet_shuffle`
     - :class:`~sparrow.null.klet_shuffle.KletShuffle`: what a k-let shuffle
       keeps, how a uniformly random one is made, and the sequences that have
       few or none.
   * - :doc:`backgrounds`
     - What to use as a background, the forms it can be given in, how large it
       needs to be for each k-mer length, and what fitting does with
       non-standard residues.
   * - :doc:`kmer_background`
     - :class:`~sparrow.null.kmer_background.KmerBackground`: how a null is
       built from the background's k-mers, and what happens when the
       background has none that fit.
   * - :doc:`markov_background`
     - :class:`~sparrow.null.markov_background.MarkovBackground`: how a null is
       drawn from the background's Markov chain, how it starts and ends, and
       the one place it is not quite uniform along its length.
   * - :doc:`composition`
     - How the background models give every null the sequence's exact
       composition without piling the residues the background disfavours up at
       the end: the tilt, the conditioned chain, and how well they work.
   * - :doc:`writing_a_null_model`
     - Writing a null model of your own on
       :class:`~sparrow.null.base.NullModel`, and the pieces the built-in ones
       are made from.

Every class and function in :mod:`sparrow.null` is documented in full in the
API reference, :doc:`../api/null`.

Choosing a null model
---------------------

The null decides the question. Its nulls share with the sequence whatever the
model keeps, so a feature that differs from the nulls' is one that what was
kept does not explain. A different model asks a different question of the same
sequence, and can give a very different answer. Here, for the charge
patterning (:math:`\kappa`) of one sequence, with 1,000 nulls from each model
and the yeast proteome as the background:

.. code-block:: python

   import numpy as np
   from sparrow.null import BlockShuffle, KletShuffle, KmerBackground, MarkovBackground
   from sparrow.properties.patterning import kappa

   sequence = "MEEEKKKKSSSTTTDDDQQQQEKEKRRRDDDESSGGKKKEEE"
   observed = kappa([sequence])[0]                  # 0.232

   models = [
       BlockShuffle(k=1),
       BlockShuffle(k=3),
       KletShuffle(k=2),
       KmerBackground(k=3).fit("yeast_proteome.fasta"),
       MarkovBackground(order=2).fit("yeast_proteome.fasta"),
   ]
   for model in models:
       null_kappas = kappa(model.generate(sequence, count=1000, seed=1))
       percentile = np.mean(null_kappas <= observed)
       z_score = (observed - null_kappas.mean()) / null_kappas.std()

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Null model
     - Mean null kappa
     - Percentile
     - z-score
   * - ``BlockShuffle(k=1)``
     - 0.127
     - 0.92
     - 1.5
   * - ``BlockShuffle(k=3)``
     - 0.238
     - 0.51
     - -0.1
   * - ``KletShuffle(k=2)``
     - 0.289
     - 0.19
     - -0.9
   * - ``KmerBackground(k=3)``
     - 0.157
     - 0.84
     - 1.0
   * - ``MarkovBackground(order=2)``
     - 0.202
     - 0.65
     - 0.3

Against plain shuffles, the sequence's charges look segregated. Its runs of
three or four like charges explain all of that: shuffles that keep blocks of
three, or every dipeptide, segregate them as much or more. A background model
asks something in between - whether the sequence is more segregated than
yeast's own grammar would make a sequence of its composition - and the answer
is a little.

As a rule:

* **A plain shuffle**, ``BlockShuffle(k=1)``, asks whether the order of the
  residues matters at all. It is the usual null, and the weakest: almost any
  natural sequence differs from it in something.
* **A block shuffle with larger blocks**, or **a k-let shuffle**, asks whether
  a feature is more than the sequence's own local order - its runs, its
  dipeptides - would give. The k-let shuffle keeps that order exactly and
  makes nothing new; the block shuffle keeps whole stretches, so suits a
  feature you think is carried by short motifs.
* **A background model** asks whether a feature is more than the grammar of
  natural sequences in general would give a sequence of this composition.
  Its nulls look natural - with yeast as the background, a Q-rich sequence
  gets runs of Q - so it is the one to use to ask whether a sequence is
  unusual *for a protein*, rather than for an arrangement of its residues.
* Between the two background models, ``KmerBackground`` keeps each block of
  ``k`` exactly as the background has it and joins blocks at random, and
  ``MarkovBackground`` has no joins, so every residue follows on from the ones
  before it. :doc:`markov_background` says when to prefer each.
