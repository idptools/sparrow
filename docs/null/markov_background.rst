The Background Markov Chain
===========================

:class:`MarkovBackground(order=2) <sparrow.null.markov_background.MarkovBackground>`
learns, from a background of natural sequences, how likely each residue is to
follow each run of ``order`` residues - a Markov chain of that order - and
draws a null of a sequence one residue at a time, each given the ``order``
residues before it, with exactly the sequence's composition. Like
:doc:`KmerBackground <kmer_background>`, it gives the null the background's
local grammar in place of the sequence's own; unlike it, there are no joins
between blocks, so every residue follows on from the ones before it.

.. code-block:: python

   from sparrow.null import MarkovBackground

   model = MarkovBackground(order=2).fit("yeast_proteome.fasta")

   sequence = "MSDSNQGNNQQNYQQYSQNGNQQQGNNRYQGYQAYNAQAQPAGGYYQNYQGYSGYQQGGY"   # Sup35, 1-60
   model.generate(sequence, count=2, seed=1)
   # ['NNNSYGAYGAGYGQGAMGYAYQQQQQQQQQSNSNYYYNGNYQQQSRGNGQGPDQQQNYYN',
   #  'AYGYGAYMYSGNGYGNSNGNNNQQQQQQGNNNADYYGYPGYSQQQQQQQQQQRYYGANSQ']

The yeast proteome is fond of runs of Q, and the chain makes them: runs of
nine and ten Qs here, where the sequence's longest is three.

The option
----------

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Option
     - Default
     - What it is
   * - ``order``
     - 2
     - How many residues before each one it is drawn given, from 0 to
       :data:`~sparrow.null.markov_background.MAX_ORDER` (3). Order ``n`` is
       learnt from the background's (n+1)-mers, so the default uses 3-mers,
       and ``order=0``, with no residues before, is a plain shuffle.

How a null is drawn
-------------------

Drawn from the background's chain alone, a null would follow the background's
composition until the residues it favours ran out, and leave the rest to the
end. So the null is drawn from the chain *given* that it ends up with exactly
the sequence's composition. Over a long sequence, that conditioning turns the
background's chain into another Markov chain (Csiszár, Cover & Choi 1987), in
which residue ``a`` follows the run ``c`` with a chance proportional to

.. math::

   P(a \mid c)\, t(a)\, h(c + a)

* :math:`P(a \mid c)` is the background's chance of ``a`` after ``c``;
* :math:`t(a)` is a weight for each residue, the same wherever it is, found
  for each sequence as the one that gives the new chain the sequence's
  composition - so that the chain does not, for instance, use up early the
  residues the background is fond of; and
* :math:`h(c + a)`, the Perron eigenvector of the weighted chain, is how well
  the run that ``a`` makes leads on to the rest. It lets the chain look ahead:
  each residue is weighted not only by its own chance, but by the weight of
  everything that can follow it, as conditioning on the composition of the
  whole null requires.

Each null's composition is then made exact, as for ``KmerBackground``, by
multiplying each residue's chance by the number of it left and dividing by its
fraction of the sequence - close to 1 while the residues left are in the
sequence's proportions, and pulling them back when they stray. :doc:`composition`
explains all of this, and how well it works.

**The start.** The first ``order`` residues have no run before them, so they
are drawn as a run of their own, from the runs the conditioned chain spends
its time in - its stationary distribution - as if the null were a stretch
from the middle of a long sequence. That way the start of a null is made no
differently from the middle.

**The end.** The last few residues are whatever is left, so the composition
forces them, as it must for any null with an exact composition.

The nulls of a sequence are drawn together, a residue of every null at a time,
which makes many of them quick to draw.

What it keeps
-------------

Every null has the sequence's composition, and every residue in it is drawn
given the ``order`` residues before it, from the background's chain weighted
towards the sequence's composition. So the null has the background's
(order+1)-mers throughout, with no joins. That includes the background's runs:
from the yeast proteome at order 3, the longest run of Q and N in nulls of the
first 254 residues of Sup35 is 16 long on average, and in nulls of Med15 30 -
longer than Med15's own, 23. ``KmerBackground(k=4)`` gives runs of 5 to 7, since
a run can only be as long as its block, or two blocks that happen to meet.

With no grammar in the background - every residue as likely after any run as
overall - the nulls are plain shuffles, which a test checks.

Where it is not quite uniform
-----------------------------

The correction that makes each null's composition exact assumes that the
residues still to place vary about as much as a random sample of the
sequence's would. A residue the chain puts in long runs varies more, and
inside a run the correction pulls against carrying it on, so a little of that
residue moves towards the end of each null. It only shows when the runs are a
good part of the sequence. With the yeast proteome as the background, over
10,000 nulls, the first quarter of each null holds this much of the sequence's
Q and N, relative to the rest:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Null model
     - Sup35, 1-254
     - Med15, all 1,081
   * - ``MarkovBackground(order=1)``
     - 1.001
     -
   * - ``MarkovBackground(order=2)``
     - 0.989
     - 1.000
   * - ``MarkovBackground(order=3)``
     - 0.939
     - 0.996
   * - ``KmerBackground(k=4)``
     - 1.005
     - 1.001

So at order 3, the first quarter of a null of Sup35's N-terminal region has
about 6% fewer Qs and Ns than the rest; at order 2, about 1%. For a feature
that depends on where along a sequence its residues are, order 2 - or
``KmerBackground``, which draws its blocks in a random order and has no such
effect - is the safer choice for short sequences rich in a residue the
background runs together.

When the background falls short
-------------------------------

Near the end of a null, or with a small background, it can happen that none
of the residues left ever follows the run before it in the background. That
residue is then drawn given a shorter run, dropping the earliest residue until
one of those left does follow it, by how much more likely each is after that
run than overall, times the number of it left; with no run at all, it is drawn
from what is left, so a null can always be finished. The model warns how many
residues were. Here, for 100 nulls of Sup35's N-terminal region at order 3,
with the 100 IDRs in sparrow's tests as the background:

.. code-block:: text

   NullModelWarning: In 97 of 100 nulls, 428 of the 25400 residues were drawn
   given fewer of the residues before them than the chain's order of 3, because
   the background never has any of the residues left after those. The nulls
   still have the sequence's composition.

With the yeast proteome as the background it is rare: in 200 nulls each of
Gal4, Med15 and Pub1 at order 3, one to four residues in all, and none at order
2. The model refuses, with
:class:`~sparrow.sparrow_exceptions.SparrowException`:

* a sequence with a residue the background never has (see :doc:`backgrounds`);
* a sequence shorter than ``order``;
* a sequence whose residues the background never has follow one another in a
  way that can go on - so that there is no chain through them at all, as when
  the background's only dipeptide of A and K is AK, so that nothing ever
  follows K, and the sequence is ``AKAK``; and
* in the rare case that the eigenvector search does not converge.

It also warns, as both background models do, if the background has too few
k-mers for their frequencies to mean much, or residues it cannot count
(:doc:`backgrounds`), and if it cannot weight the background to the
sequence's composition (:doc:`composition`).

Speed
-----

Finding the conditioned chain for a sequence takes most of the time: with the
yeast proteome as the background, about 0.05 s at order 2 and 0.3 s at order
3, once for all the sequence's nulls. Each null then takes well under a
millisecond, so 100 nulls of Sup35's N-terminal region take about 0.04 s at
order 2. For a few nulls of each of many sequences, ``KmerBackground`` is
quicker; for many nulls of each, ``MarkovBackground`` is.

``MarkovBackground`` or ``KmerBackground``?
-------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - ``KmerBackground(k)``
     - ``MarkovBackground(order)``
   * - The background's grammar
     - Exactly, within each block of ``k``; the joins between blocks are
       random.
     - Throughout, each residue given the ``order`` before it.
   * - Runs and long motifs
     - No longer than a block, or two that happen to meet.
     - As long as the background makes them.
   * - Along the null
     - The same everywhere, since the blocks are put in a random order.
     - The same everywhere but for a little drift at order 3, for short
       sequences rich in a residue the background runs together.
   * - Speed
     - A few milliseconds a null.
     - A fraction of a second a sequence, then well under a millisecond a
       null.

``MarkovBackground`` makes the more natural-looking nulls, and is the one to
use to ask whether a sequence is unusual for a protein of its composition.
``KmerBackground`` keeps the background's motifs of up to ``k`` residues
exactly, but breaks everything longer at its joins.

Reference
---------

Csiszár I, Cover TM, Choi BS (1987) Conditional limit theorems under Markov
conditioning. *IEEE Trans Inf Theory* 33:788-801.

The class is documented in full in :doc:`../api/null`.
