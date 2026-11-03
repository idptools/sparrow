The K-let Shuffle
=================

:class:`KletShuffle(k=2) <sparrow.null.klet_shuffle.KletShuffle>` makes
sequences with exactly as many of each *k-let* - each run of ``k``
consecutive residues - as the original. With ``k=2``, a null has the same
count of every pair of neighbouring residues, every dipeptide, and so also the
same composition. It is the protein form of the dinucleotide shuffle of
Altschul & Erickson (1985), the usual null for asking whether a sequence's
order means more than its local statistics. It needs no background.

.. code-block:: python

   from sparrow.null import KletShuffle

   sequence = "MEEEKKKKSSSTTTDDDQQQQEKEKRRRDDDESSGGKKKEEE"

   KletShuffle(k=2).generate(sequence, seed=1)
   # ['MEKKKSGGKRRRDEEKKEEEESSSSTTTDDDDDQQQQEKKKE']

   KletShuffle(k=3).generate(sequence, seed=1)
   # ['MEEEEKRRRDDESSGGKKKKKSSSTTTDDDDQQQQEKKEKEE']

The option
----------

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Option
     - Default
     - What it is
   * - ``k``
     - 2
     - The length of the runs whose counts are kept, a positive integer
       shorter than the sequence. ``k=1`` keeps only the composition: a plain
       shuffle.

What it keeps
-------------

Every k-let the sequence has, as many times as it has it, and so every shorter
run too: a dipeptide shuffle keeps the composition, and a tripeptide shuffle
keeps every dipeptide as well. Unlike :doc:`the block shuffle <block_shuffle>`,
it carries no residues around as rigid blocks, and makes no k-let the sequence
does not have - there are no joins.

The first and last ``k - 1`` residues stay where they are, as in Altschul &
Erickson and in uShuffle (Jiang et al. 2008): both nulls above start with M and
end with E. Every sequence with the same k-lets and the same ends is equally
likely, which a test checks by counting the nulls of sequences short enough to
list every arrangement.

How a null is made
------------------

Write the sequence as a walk through a graph whose points are its
(k-1)-lets, with an edge for each k-let, from the (k-1)-let it starts with to
the one it ends with: with ``k=2``, the points are residues, and each
dipeptide is an edge from its first residue to its second. Every walk that
starts at the first (k-1)-let and uses each edge once - an Eulerian trail -
spells out a sequence with the same k-lets, and every such sequence is one of
these walks. A uniformly random one is made as Kandel et al. (1996) describe:

1. For each point but the last, pick the edge by which the walk leaves it for
   the last time, so that these edges form a tree leading to the last point.
   Wilson's (1996) algorithm picks the tree uniformly at random, by random
   walks with their loops erased.
2. Put each point's other edges in a random order, with its last edge last.
3. Walk from the first point, leaving each point by its next unused edge.

The tree in step 1 means the walk can never be stuck at a point while edges
remain elsewhere, and choosing the tree and the orders uniformly makes every
trail - every sequence - equally likely.

Sequences with few k-let shuffles
---------------------------------

A k-let shuffle can only be as varied as the sequence's k-lets allow. A short
sequence, or one in which each residue comes in a single run, may have few
other sequences with the same k-lets, or none, and then its nulls are often,
or always, the sequence itself. In 2,000 nulls each:

.. list-table::
   :header-rows: 1
   :widths: 46 10 22 22

   * - Sequence
     - ``k``
     - Distinct nulls
     - Nulls that are the sequence
   * - ``MKMKM``
     - 2
     - 1
     - all
   * - ``MEEEKKKKSSSTTTDDDQQQQ``
     - 2
     - 1
     - all
   * - ``MEEEKKKKSSSTTTDDDQQQQEKEKRRRDDDESSGGKKKEEE``
     - 2
     - 1,998
     - none
   * - the same
     - 3
     - 948
     - 1
   * - the same
     - 4
     - 2
     - 971

``MEEEKKKKSSSTTTDDDQQQQ`` has each residue in one run, so its only dipeptides
are the ones inside the runs and one joining each run to the next, and there
is just one way to spell them out. The model does not refuse such sequences - they are a true
answer, that the k-lets fix the sequence - but a null distribution with one
member is no distribution at all. ``len(set(nulls))`` says how many different
nulls there are.

When it refuses
---------------

A sequence no longer than ``k`` has only itself as a k-let shuffle - its one
k-let, or none - so ``generate`` raises
:class:`~sparrow.sparrow_exceptions.SparrowException`. ``k`` must be a positive
integer; anything else raises when the model is made.

Speed
-----

A dipeptide shuffle of a 254-residue sequence takes about 0.15 ms. Each null is
a walk of single random choices, which the model makes with Python's own
random number generator, seeded from NumPy's, since that is far faster for one
choice at a time.

When to use it
--------------

The k-let shuffle is the strictest of the shuffles: the nulls share everything
local with the sequence, so a feature that differs from theirs is one that only
the arrangement of the sequence over longer distances gives it. A dipeptide
shuffle is the usual choice; ``k=3`` keeps so much that most sequences of a few
dozen residues have few nulls, as above.

References
----------

Altschul SF, Erickson BW (1985) Significance of nucleotide sequence alignments:
a method for random sequence permutation that preserves dinucleotide and codon
usage. *Mol Biol Evol* 2:526-538.

Kandel D, Matias Y, Unger R, Winkler P (1996) Shuffling biological sequences.
*Discrete Appl Math* 71:171-185.

Wilson DB (1996) Generating random spanning trees more quickly than the cover
time. *Proc 28th ACM Symp Theory Comput* 296-303.

Jiang M, Anderson J, Gillespie J, Mayo M (2008) uShuffle: a useful tool for
shuffling biological sequences while preserving the k-let counts. *BMC
Bioinformatics* 9:192.

The class is documented in full in :doc:`../api/null`.
