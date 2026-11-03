The Block Shuffle
=================

:class:`BlockShuffle(k=1) <sparrow.null.block_shuffle.BlockShuffle>` cuts the
sequence into blocks of ``k`` residues and puts the blocks in a random order.
With ``k=1`` that is a plain shuffle, which keeps the composition and nothing
else. With larger blocks, every residue keeps the neighbours it had inside its
block, so the null keeps the sequence's short stretches - its runs and motifs
up to ``k`` long - while the order of the stretches is random. It needs no
background: it is an empirical null, made from the sequence alone.

.. code-block:: python

   from sparrow.null import BlockShuffle

   sequence = "MEEEKKKKSSSTTTDDDQQQQEKEKRRRDDDESSGGKKKEEE"

   BlockShuffle(k=1).generate(sequence, seed=1)
   # ['EKSEREKDKRKTEQDMKSTDEDQSGGKDETESEKKSEQDQRK']

   BlockShuffle(k=3).generate(sequence, seed=1)
   # ['EKKDDQEEEKRRQQQSGGMEEKKSSSTRDDDESEKEKKKTTD']

   BlockShuffle(k=[2, 3]).generate(sequence, seed=1)
   # ['TTSGGEEKEKSSTDDDMEEKKSKKDDEEKEDESQQKKQQRRR']

The option
----------

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Option
     - Default
     - What it is
   * - ``k``
     - 1
     - The block size, a positive integer, or a list of different ones, each
       of which then covers about an equal share of the residues. ``k=1`` is a
       plain shuffle.

The sizes are kept, smallest first, as the ``block_sizes`` attribute:
``BlockShuffle(k=[3, 2]).block_sizes`` is ``(2, 3)``.

What it keeps
-------------

Every null has the sequence's composition. A block of ``k`` residues keeps the
``k - 1`` pairs of neighbours inside it, so a block shuffle keeps about
``(k - 1) / k`` of the sequence's pairs of neighbouring residues, and every
run of residues no longer than ``k`` that falls inside one block. Measured over
2,000 nulls of a 300-residue sequence:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - ``k``
     - Pairs of neighbours kept
     - (k - 1) / k
   * - 1
     - none
     - 0
   * - 2
     - 0.500
     - 1/2
   * - 3
     - 0.667
     - 2/3
   * - 4
     - 0.750
     - 3/4
   * - ``[2, 3]``
     - 0.583
     - the mean of 1/2 and 2/3

The pairs at the joins between blocks are new: nothing about the sequence
decides which block follows which. A block shuffle can therefore make pairs of
residues the sequence never has, where :doc:`the k-let shuffle <klet_shuffle>`
cannot.

How the sequence is cut
-----------------------

**One size.** With ``k=3``, the sequence is cut into blocks of 3, and any
residues left over - fewer than 3 - make one shorter block. A sequence of 10
residues is three blocks of 3 and one of 1.

**Several sizes.** With ``k=[2, 3]``, the sequence is cut into blocks of 2 and
3, with each size covering as close to an equal share of the residues as whole
blocks allow, and with no residues left over whenever some mix of the sizes
adds up to the length. Only if none does is there one shorter block.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Length
     - ``k``
     - Blocks
   * - 10
     - 3
     - three of 3, one of 1
   * - 11
     - 2
     - five of 2, one of 1
   * - 7
     - ``[2, 3]``
     - two of 2, one of 3
   * - 100
     - ``[2, 3]``
     - 26 of 2 (52 residues), 16 of 3 (48 residues)
   * - 100
     - ``[2, 3, 4]``
     - 16 of 2, 12 of 3, 8 of 4 (32, 36 and 32 residues)
   * - 101
     - ``[2, 4]``
     - 26 of 2, 12 of 4, one of 1

The same blocks are used for every null of a sequence; only their order, and
where the cuts fall, change. :func:`sparrow.null.blocks.plan_blocks` gives the
blocks for any length and sizes.

**Where the cuts fall.** Cutting from the first residue every time would keep
the same pairs in every null - with ``k=2``, always residues 1 and 2, 3 and 4,
and so on, and never 2 and 3. So for each null the blocks are put in a random
order, and the cuts are then moved along by a random number of residues within
the first block, with what that cuts off its start becoming a block at the
end. Every pair of neighbours is then as likely as any other to be kept: with
``k=2``, the pairs 1-2, 3-4, ... half the time, and 2-3, 4-5, ... the other
half, with a block of one residue at either end. The price is that the
sequence's first and last residues are in shorter blocks more often than the
rest - on their own half the time, with ``k=2``.

When it refuses
---------------

A sequence no longer than one block - three residues with ``k=3``, say - has
only one arrangement, itself, so ``generate`` raises
:class:`~sparrow.sparrow_exceptions.SparrowException` rather than return the
sequence as its own null. ``k`` must be a positive integer or a list of
different ones; anything else - 0, a float, ``True``, an empty list, a size
given twice - raises when the model is made.

Speed
-----

A null of a 254-residue sequence takes about 0.1 ms, so a million nulls take
under two minutes.

When to use it
--------------

``BlockShuffle(k=1)`` is the usual null for asking whether the order of a
sequence's residues matters at all. With larger blocks it asks whether a
feature is carried by the sequence's short stretches, or by how the stretches
are arranged: a feature that survives a block shuffle with ``k=3`` is one its
runs and motifs of up to three residues give it, wherever they are. A
patterning feature that changes as ``k`` grows says over what lengths the
patterning lies - ``k=[2, 3]`` and ``k=[3, 4]`` sit between the whole sizes.

The class is documented in full in :doc:`../api/null`.
