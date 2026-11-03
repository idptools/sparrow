Background K-mer Reconstruction
===============================

:class:`KmerBackground(k=3) <sparrow.null.kmer_background.KmerBackground>`
counts every k-mer in a background of natural sequences, and rebuilds a
sequence from blocks of ``k`` residues, each a k-mer drawn according to how
often the background has it, with exactly the sequence's composition. The null
has the background's local grammar - its favoured tripeptides, with ``k=3`` -
in place of the sequence's own, and no trace of where along the sequence each
residue was.

.. code-block:: python

   from sparrow.null import KmerBackground

   model = KmerBackground(k=3).fit("yeast_proteome.fasta")

   sequence = "MSDSNQGNNQQNYQQYSQNGNQQQGNNRYQGYQAYNAQAQPAGGYYQNYQGYSGYQQGGY"   # Sup35, 1-60
   model.generate(sequence, count=2, seed=1)
   # ['QNNQQQNNYYNGQQQQQQGYNGYSYGYYDYSQQAANGRNGNGGNPQQMGQGAQYQSYYSA',
   #  'YAYQNNNGYRAGYGQANSQYNQQQQQQYGNSGGQSGYQQAGQMGQQQQYYSYDNYPNGNN']

The yeast proteome puts Q next to Q, and N next to N, more often than chance,
and the nulls do too.

The option
----------

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Option
     - Default
     - What it is
   * - ``k``
     - 3
     - The block size, from 1 to :data:`~sparrow.null.kmers.MAX_KMER_SIZE`
       (4), or a list of different ones, each of which then covers about an
       equal share of the residues. ``k=1`` is a plain shuffle.

The sizes are kept, smallest first, as the ``block_sizes`` attribute. Fitting
counts every k-mer up to the largest size, since a block that cannot be drawn
at its size is drawn as smaller ones (below).

How a null is built
-------------------

1. **The blocks.** The sequence's length is split into blocks just as for
   :doc:`the block shuffle <block_shuffle>`: with ``k=3``, blocks of 3 and
   one shorter block for any residues left over; with ``k=[2, 3]``, blocks of
   2 and 3 with about half the residues in each.
2. **Drawing each block.** The blocks are drawn one at a time, in a random
   order of their sizes. Each is a k-mer of its size, picked with a chance
   proportional to

   * its background frequency, tilted towards the sequence's composition -
     ``P(b) t(b1) t(b2) ... t(bk)`` for a k-mer ``b`` with residues ``b1`` to
     ``bk`` - by a weight ``t`` for each residue, the same wherever it is,
     found for each sequence so that k-mers drawn this way have on average the
     sequence's composition; times
   * the chance of drawing the k-mer's residues from those of the sequence not
     yet in a block, over the chance of drawing them from the sequence's
     composition. This is close to 1 while the residues left are in the
     sequence's proportions, and pulls them back when they stray, so the last
     block uses up exactly what is left.

   A block of one residue has no grammar to draw on, and is drawn from the
   residues left in proportion to their number.
3. **The order.** Once every block is drawn, the blocks are put in a random
   order, so no part of the null is made differently from any other.

:doc:`composition` explains the weights, why they are needed, and how well
they work. Without them the blocks would follow the background's composition
until the residues it favours ran out, and the last blocks drawn would be made
of whatever the background disfavours.

The k-mers are grouped by the residues they are made of, since that decides
everything about a k-mer's chance but its background frequency. A block is
drawn as a group first - one of 8,855 groups of four residues, rather than one
of 160,000 4-mers - and then as an order of its residues within the group, by
their background frequencies.

What it keeps
-------------

Every null has the sequence's composition. Inside each block, the residues
are in the background's proportions of that k-mer, tilted to the sequence's
composition; at the joins between blocks, the pairs are random. So
``KmerBackground(k=3)`` gives a null the background's tripeptides, but
only within blocks: about two thirds of its pairs of neighbours are the
background's, and a third are chance. With no grammar in the background -
every k-mer as frequent as its residues make it - the nulls are plain
shuffles, which a test checks.

When the background falls short
-------------------------------

Near the end of a null, few residues are left, and the background may have no
k-mer of a block's size made only of them - more often with a small
background, or with a sequence of residues the background seldom puts
together. That block is drawn as two smaller ones instead, one residue shorter
and a single residue, and the model warns how many were. Here, for 100 nulls of
Sup35's N-terminal region from 4-mers, with the 100 IDRs in sparrow's tests -
43,502 residues - as the background:

.. code-block:: text

   NullModelWarning: In 5 of 100 nulls, 5 of the 6400 blocks were drawn as
   smaller ones, because the background has no k-mer of their size made of the
   residues left. The nulls still have the sequence's composition.

With the yeast proteome as the background, no block of any sequence tested
has needed splitting. A residue the background never has at all is refused
outright (see :doc:`backgrounds`).

The model also warns, as both background models do, if the background has too
few k-mers for their frequencies to mean much, or residues it cannot count
(:doc:`backgrounds`), and if it cannot weight the background to the sequence's
composition (:doc:`composition`).

Speed
-----

Each null is built a block at a time. With the yeast proteome as the
background, a null of the 254 residues of Sup35's N-terminal region takes about
2.5 ms from 3-mers and 5.5 ms from 4-mers, and a null of all 1,081 residues of
Med15 about 12 ms and 32 ms. Finding the weights for a sequence takes about
1 ms from 3-mers and 3 ms from 4-mers, once for all its nulls.

When to use it
--------------

``KmerBackground`` asks whether a feature of a sequence is more than the local
grammar of natural sequences would give one of its composition, with that
grammar held exactly within blocks of ``k``. It suits a feature you think is
carried by short motifs: the background's motifs of up to ``k`` residues are
kept whole, as the background has them. For a null in which every residue
follows on from the ones before it, with no joins, use
:doc:`MarkovBackground <markov_background>`; :doc:`markov_background`
compares the two.

The class is documented in full in :doc:`../api/null`.
