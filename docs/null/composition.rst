Matching the Composition Exactly
================================

Every null has exactly the composition of the sequence it was made from. For
the shuffles that comes for free, since they only rearrange the sequence's own
residues. The background models draw their nulls from a background's grammar
instead, and the background has a composition of its own, so they have to be
made to give the sequence's. This page explains how, why the obvious way is
not good enough, and how well the way sparrow uses works.

The obvious way, and where it goes wrong
----------------------------------------

The obvious way to give a null the sequence's composition is to draw from the
residues still to place: weight each residue's background chance by how many
of it are left, as if drawing from an urn of the sequence's residues. The
composition is then exact, since the last residue is whatever is left.

But the background still decides which residues are drawn early. A background
favours some residues - more than the sequence has of them, or in runs - and
the urn only pushes back once they have been used up. So a null drawn this way
uses the favoured residues up early, and leaves the rest to pile up at its
end. With the yeast proteome as the background, which is fond of runs of Q and
N, an order-3 chain weighted only by the residues left put Q and N in 33% of
the first quarter of each of 200 nulls of Med15, and 20% of the last; the
sequence has 27%. Any feature that depends on where along a sequence its
residues are is then compared with nulls that are not made the same way along
their length, and the comparison says more about the null than the sequence.

Conditioning on the composition
-------------------------------

What is wanted is a null drawn from the background *given* that it has exactly
the sequence's composition: of all the sequences the background could make,
only those with the right composition, each as likely as the background makes
it. Drawn that way, nothing about the start of a null differs from its end,
except what the composition forces.

Over a long sequence, conditioning like this changes the background in one
simple way - the conditional limit theorem (Cover & Thomas 2006, section 11.6;
Csiszár, Cover & Choi 1987 for Markov chains). Each residue ``a`` gets a weight
:math:`t(a)`, the same wherever it is, and everything the background draws is
weighted by the product of the weights of its residues. The weights are the
ones that give the weighted background the sequence's composition: a residue
the background makes too often is weighted down, and one it makes too seldom
up.

Writing :math:`t(a) = e^{\theta(a)}`, the weights are the :math:`\theta` that
minimize

.. math::

   A(\theta) - \theta \cdot q

where :math:`q` is the sequence's composition and :math:`A(\theta)` is the log
of the weighted background's normalizer, per residue. :math:`A` is convex, and
its gradient is the weighted background's composition, so the minimum is
exactly where the weighted background's composition is the sequence's, and
there is only one. :func:`sparrow.null.tilt.fit_tilt` finds it with L-BFGS,
which takes 14 to 34 iterations with the yeast proteome as the background.

**For k-mers**, the weighted background gives the k-mer :math:`b`, with
residues :math:`b_1 \ldots b_k`, a frequency proportional to

.. math::

   P(b)\, t(b_1)\, t(b_2) \cdots t(b_k)

for its background frequency :math:`P(b)`, and :math:`A(\theta)` is the log of
the sum of these over all k-mers, divided by ``k``.

**For a Markov chain**, the weighted chain is a matrix between runs of
``order`` residues, with :math:`P(a \mid c)\, t(a)` for going from the run
:math:`c` to the run that :math:`a` makes on the end of it. :math:`A(\theta)` is
the log of its Perron root :math:`\rho`, its largest eigenvalue. The chain
conditioned on the composition is then the Doob h-transform of the weighted
one, in which ``a`` follows ``c`` with a chance

.. math::

   \frac{P(a \mid c)\, t(a)\, h(c + a)}{\rho\, h(c)}

where :math:`h`, the right Perron eigenvector, says how much weight lies ahead
of each run. This is a Markov chain of its own, and it spends its time at each
run in proportion to :math:`l(c)\, h(c)`, for the left Perron eigenvector
:math:`l` - its stationary distribution, which is what the first ``order``
residues of a null are drawn from. Both eigenvectors come from ARPACK, through
:func:`scipy.sparse.linalg.eigs`, over only the runs made of the sequence's
residues.

Each null's composition made exact
----------------------------------

The weights give the sequence's composition on average, over a long null. To
make each null's composition exact, every chance is also multiplied by the
number of each residue left and divided by the residue's fraction of the
sequence: for ``a`` with :math:`n(a)` of it left and fraction :math:`q(a)` of
the sequence, by :math:`n(a) / q(a)`, and for a block, by the product over its
residues, taking one fewer of a residue each time it comes again. While the
residues left are in the sequence's proportions, this factor is the same for
every residue and changes nothing; when they stray, it pulls them back, and it
makes the last residues whatever is left.

So, together, a ``KmerBackground`` block :math:`b` is drawn with a chance
proportional to

.. math::

   P(b) \prod_{i=1}^{k} \frac{t(b_i)\, \bigl(n(b_i) - r_i\bigr)}{q(b_i)}

with :math:`r_i` the number of times :math:`b_i` comes earlier in the block,
and a ``MarkovBackground`` residue :math:`a` after the run :math:`c` with a
chance proportional to

.. math::

   P(a \mid c)\, t(a)\, h(c + a)\, \frac{n(a)}{q(a)}

With no grammar in the background - every k-mer as frequent as its residues
make it, every residue as likely after any run as overall - the weights are
:math:`t(a) = q(a) / f(a)` for the residue's fraction :math:`f(a)` of the
background, :math:`h` is constant, and both chances reduce to drawing from
the residues left: a plain shuffle. A test checks this for each model.

How well it works
-----------------

With the conditioning, an order-3 chain from the yeast proteome puts 26-27% Q
and N in every quarter of 2,000 nulls of Med15, where the residues left alone
gave 33% in the first quarter and 20% in the last. Over 10,000 nulls, the first
quarter of each null holds this much of the sequence's Q and N, relative to the
rest:

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

sparrow's tests check the same thing for both models: with a background made
of poly-Q runs, the fraction of Q in each quarter of the nulls is within 3
percentage points of the sequence's. Two more check that the conditioned chain,
and the weighted k-mers, have the sequence's composition. With the weights
switched off, the Markov chain fails both its positional test and its
composition test.

**What is left.** The factor that makes each null exact assumes that the
residues still to place vary about as much as a random sample of the
sequence's would. A residue the chain puts in long runs varies more, and inside
a run the factor pulls against carrying it on, so a little of that residue
moves towards the end of each null. That is the 6% at order 3 for Sup35's
N-terminal region, whose nulls have runs of Q and N about 16 long - a good part
of 254 residues. ``KmerBackground`` has no such effect, since it draws its
blocks first and then puts them in a random order. Taking it out of
``MarkovBackground`` too would mean polishing each null with a Markov chain
Monte Carlo pass, which sparrow does not do; for short sequences rich in a
residue the background runs together, order 2 is the safer choice.

When the weights cannot be found
--------------------------------

Some backgrounds cannot be weighted to some compositions at all. A background
that only ever alternates A and K has the same number of each in every
stretch, whatever the weights, so it can never give a sequence with three A
to every K. The weights are bounded - a million-fold either way from where
they start, the weights a background with no grammar would need - and if the
weighted composition is still further than
:data:`~sparrow.null.tilt.TILT_TOLERANCE` (0.1%) from the sequence's for any
residue, the model warns, and makes its nulls anyway:

.. code-block:: text

   NullModelWarning: The background could only be tilted to within 100.0% of the
   sequence's composition (the aim is 0.1%), so the residues placed early in each
   null may not be in quite the same proportions as those placed late. The nulls
   still have the sequence's composition.

The nulls still have exactly the sequence's composition - the factor from the
residues left sees to that - but the residues placed early may not be in quite
the same proportions as those placed late, as with the residues left alone.
With a natural background, such as a proteome, the weights are found to
within the tolerance.

References
----------

Cover TM, Thomas JA (2006) *Elements of Information Theory*, 2nd ed. Wiley.

Csiszár I, Cover TM, Choi BS (1987) Conditional limit theorems under Markov
conditioning. *IEEE Trans Inf Theory* 33:788-801.
