Writing a Null Model
====================

Every null model derives from :class:`~sparrow.null.base.NullModel`, which
does everything the models share: it checks the sequence, the count and the
seed, seeds the random number generator from the seed and the sequence,
refuses to make nulls before a background model is fit, and turns the nulls
back into sequences. A new model only has to say whether it needs a
background, and make the nulls.

A model that needs no background
--------------------------------

Set ``needs_background`` to False, and implement ``_generate``. Here, a model
that shuffles the residues within consecutive windows, so each stays near
where it was - a null for asking whether a feature depends on the order of
the residues within a region, rather than which residues the region has:

.. code-block:: python

   import numpy as np
   from sparrow.null import NullModel


   class WindowShuffle(NullModel):
       """Shuffles within consecutive windows, so each residue stays near where it was.

       Parameters
       ----------
       window : int
           The number of residues in each window; the last may be shorter.
       """

       needs_background = False

       def __init__(self, window: int = 10) -> None:
           super().__init__()
           self.window = window

       def _generate(self, codes, count, rng):
           nulls = []
           for _ in range(count):
               windows = [codes[start : start + self.window] for start in range(0, len(codes), self.window)]
               nulls.append(np.concatenate([rng.permutation(window) for window in windows]))
           return nulls


   WindowShuffle(window=5).generate("MEEEKKKKSSSTTTDDDQQQQ", count=2, seed=1)
   # ['EMEEKKKSSKTTDTSQQDDQQ', 'EEEKMKSSKKTTDSTDQQDQQ']

It works like any other null model: ``count``, ``seed``, a ``Protein`` in place
of a sequence, and the errors for a bad sequence, count or seed all come from
:class:`~sparrow.null.base.NullModel`.

What ``_generate`` is given, and returns
----------------------------------------

``_generate(codes, count, rng)`` is given:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Argument
     - What it is
   * - ``codes``
     - The sequence as a NumPy ``int8`` array of residue codes, never empty:
       each residue's index in :data:`sparrow.properties.encoding.RESIDUES`,
       ``ACDEFGHIKLMNPQRSTVWY``, so A is 0 and Y is 19.
   * - ``count``
     - How many nulls to make, a positive integer.
   * - ``rng``
     - The :class:`numpy.random.Generator` to make every random choice with.
       It is seeded from the seed and the sequence, so making every choice
       with it is what makes the model's nulls reproducible, and different
       for different sequences.

It returns a list of ``count`` arrays of residue codes, each the length of
``codes``. The model is responsible for what its nulls keep: every built-in
model gives each null exactly the sequence's composition, and a new one should
too, unless it has a reason not to that its docstring says.

A sequence the model cannot make a null of should raise
:class:`~sparrow.sparrow_exceptions.SparrowException`, saying why, rather than
return something that is not a null. Nulls the model can make, but not quite as
asked, should come with a :class:`~sparrow.sparrow_exceptions.NullModelWarning`
saying how many were affected.

A model that learns from a background
-------------------------------------

Set ``needs_background`` to True, and implement ``_fit`` as well. ``fit`` reads
the background, from a FASTA file, a dict or a list, passes ``_fit`` its
sequences as a list of upper-case strings - exactly as given, with any
non-standard residues still in them - and marks the model as fit. ``_fit``
stores whatever the model learns on the model. Here is the shape of one - plain
shuffles that refuse a residue the background never has:

.. code-block:: python

   import numpy as np
   from sparrow.null import NullModel
   from sparrow.null.kmers import count_background_kmers
   from sparrow.properties import encoding
   from sparrow.sparrow_exceptions import SparrowException


   class SeenResidueShuffle(NullModel):
       """Plain shuffles of sequences made only of residues the background has."""

       needs_background = True

       def _fit(self, sequences):
           counts = count_background_kmers(sequences, 1, type(self).__name__)
           self.frequencies = counts[1] / counts[1].sum()

       def _generate(self, codes, count, rng):
           missing = [encoding.RESIDUES[code] for code in np.unique(codes) if self.frequencies[code] == 0]
           if missing:
               raise SparrowException(f"The background has no {', '.join(missing)}")
           return [rng.permutation(codes) for _ in range(count)]


   model = SeenResidueShuffle().fit(["MEEEKKKKSSSTTTDDDQQQQ"] * 300)
   model.generate("MEEKK", count=2, seed=1)    # ['KEKME', 'KEKEM']
   model.generate("MEEKW")                     # SparrowException: The background has no W

:func:`~sparrow.null.kmers.count_background_kmers` counts every k-mer up to a
length in one pass, leaving out any that span a non-standard residue or the
join between two sequences, and warns about those residues and about a
background too small for its k-mers, as the built-in models do.

The pieces the built-in models are made from
--------------------------------------------

The built-in models are built from a few shared pieces, which a new model can
use too. Each is documented in full in :doc:`../api/null`.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - What it does
   * - :func:`sparrow.null.blocks.check_block_sizes`
     - Checks a block size, or a list of them, as ``BlockShuffle`` and
       ``KmerBackground`` take ``k``.
   * - :func:`sparrow.null.blocks.plan_blocks`
     - The blocks a sequence of some length is cut into, with each size
       covering an equal share of the residues.
   * - :func:`sparrow.null.kmers.count_background_kmers`
     - The count of every k-mer in a background, for each length up to the
       longest asked for.
   * - :func:`sparrow.null.kmers.kmer_residues`
     - The residues of each k-mer, from its index.
   * - :func:`sparrow.null.kmers.weighted_choice`
     - An index, picked with a chance proportional to its weight.
   * - :func:`sparrow.null.tilt.fit_tilt`
     - The weights that tilt a background to a sequence's composition, given
       the log normalizer of the weighted background (see :doc:`composition`).

The four built-in models are worked examples of all of this, from the short
:mod:`sparrow.null.block_shuffle` to :mod:`sparrow.null.markov_background`.

Checking a new model
--------------------

Two things are worth a test for any null model: that every null has the
length and composition the model promises, and that the same seed gives the
same nulls while different sequences get different ones. The tests of the
built-in models, in ``sparrow/tests/test_null.py``, run those checks over every
model, and are a starting point.
