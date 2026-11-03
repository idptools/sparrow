Making and Using Nulls
======================

Every null model is used the same way, since every one derives from
:class:`~sparrow.null.base.NullModel`:

1. make it with its options, such as ``BlockShuffle(k=2)`` or
   ``MarkovBackground(order=3)``;
2. if it learns from a background, give it one with ``fit``;
3. ask it for nulls of any sequence with ``generate``.

.. code-block:: python

   from sparrow.null import BlockShuffle, MarkovBackground

   shuffle = BlockShuffle(k=2)
   nulls = shuffle.generate("MEEEKKKKSSSTTTDDDQQQQ", count=100, seed=1)

   chain = MarkovBackground(order=2).fit("human_idrs.fasta")
   nulls = chain.generate("MEEEKKKKSSSTTTDDDQQQQ", count=100, seed=1)

A model is made once and used for as many sequences as you like. Making nulls
changes nothing about it, so the same call gives the same nulls however many
other calls come between.

Making nulls
------------

:meth:`NullModel.generate(sequence, count=1, seed=None) <sparrow.null.base.NullModel.generate>`
takes:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Argument
     - Default
     - What it is
   * - ``sequence``
     - (required)
     - The natural sequence, as a string or a
       :class:`~sparrow.protein.Protein`. It is upper-cased, and must be made of
       the 20 standard amino acids.
   * - ``count``
     - 1
     - How many nulls to make: a positive integer.
   * - ``seed``
     - None
     - A non-negative integer, for nulls you can make again; None for nulls
       that are different every time.

and returns a list of ``count`` sequences, each a string - a list even when
``count`` is 1. Every null has exactly the length of ``sequence`` and its
composition, the count of every residue.

Learning from a background
--------------------------

A model that learns from a background has ``needs_background`` set to True,
and must be given one with
:meth:`fit(background) <sparrow.null.base.NullModel.fit>` before it can make
nulls. The background can be the path to a FASTA file, a dict of sequences, or
a list of them (:doc:`backgrounds` has the details). ``fit`` returns the model
itself, so a model can be made and fit in one line, and fitting again replaces
what it learnt before. Its ``is_fitted`` property says whether it has been fit.

.. code-block:: python

   from sparrow.null import KmerBackground

   model = KmerBackground(k=3)
   model.needs_background      # True
   model.is_fitted             # False

   model.fit("human_idrs.fasta")
   model.is_fitted             # True

The shuffles need no background, and refuse one rather than ignore it:
``BlockShuffle().fit(...)`` raises an error.

Seeds and reproducible nulls
----------------------------

With a seed, the nulls of a sequence depend on the seed and the sequence alone:
the random number generator is seeded with both, the sequence by a hash of its
residues. That has some useful consequences.

* **One seed for many sequences.** A loop over many sequences with one seed
  gives each its own nulls, not the same arrangement of each:

  .. code-block:: python

     import protfasta

     idrs = protfasta.read_fasta("my_idrs.fasta")
     nulls = {name: model.generate(idr, count=100, seed=1) for name, idr in idrs.items()}

* **The same nulls however they are batched.** A sequence's nulls do not depend
  on which other sequences were made alongside it, or in what order, so a run
  split across jobs gives what one run would.
* **The same sequence, the same nulls.** Two identical sequences given the same
  seed get identical nulls, even under different names. Give them different
  seeds if you need them to differ.
* **A different background, different nulls.** A background model's nulls
  also depend on what it was fit to, so the same seed and sequence with a
  different background give different nulls.

The nulls from a smaller ``count`` are the first of those from a larger one for
the shuffles and :class:`~sparrow.null.kmer_background.KmerBackground`, but not
for :class:`~sparrow.null.markov_background.MarkovBackground`, which draws all
its nulls together. Rather than rely on it, ask for as many nulls as you need in
one call.

Comparing a sequence with its nulls
-----------------------------------

The nulls are plain sequences, so anything sparrow computes for a sequence can
be computed for them, and the sequence's value set against theirs. Three
summaries are common:

* the **percentile** of the sequence's value among the nulls' - the fraction of
  nulls with a value no larger;
* the **z-score**, how many standard deviations the sequence's value is from
  the nulls' mean;
* an **empirical p-value**, the fraction of nulls at least as extreme as the
  sequence. Counting the sequence itself among them - ``(1 + extreme) / (1 +
  count)`` - keeps it from being 0, which no finite number of nulls can show.

.. code-block:: python

   import numpy as np
   from sparrow.null import KletShuffle
   from sparrow.properties.patterning import kappa

   sequence = "MEEEKKKKSSSTTTDDDQQQQEKEKRRRDDDESSGGKKKEEE"
   nulls = KletShuffle(k=2).generate(sequence, count=1000, seed=1)

   observed = kappa([sequence])[0]
   null_kappas = kappa(nulls)

   percentile = np.mean(null_kappas <= observed)
   z_score = (observed - null_kappas.mean()) / null_kappas.std()
   p_value = (1 + np.sum(null_kappas >= observed)) / (1 + len(nulls))

The z-score is only as good as the nulls' spread is close to a normal
distribution; the percentile and p-value assume nothing about it. A feature
that takes few values - a count of motifs, say - often ties with many of its
nulls, so it is worth counting the nulls equal to the sequence apart from
those below it.

A feature of the composition alone is the same in every null, since every null
has the sequence's composition: the fraction of charged residues, the net
charge per residue, the mean hydropathy, the molecular weight. There is nothing
to compare for those - the nulls' spread is zero, to rounding, and a z-score
divides by it.

Scoring nulls with the rest of sparrow
--------------------------------------

For the sequence parameters, :func:`sparrow.properties.compute` gives any of
them for every null in one call, as arrays (see
:doc:`../power_user/properties`):

.. code-block:: python

   from sparrow.properties import compute

   observed = compute([sequence], ["kappa", "SCD", "SHD"])
   null_values = compute(nulls, ["kappa", "SCD", "SHD"])

   for name in observed:
       z_score = (observed[name][0] - null_values[name].mean()) / null_values[name].std()

For anything else a :class:`~sparrow.protein.Protein` computes - a network's
predictions, a polymer model, NARDINI+ - :func:`sparrow.batch` makes a
``Protein`` for every null with the values you name already computed (see
:doc:`../batch/index`):

.. code-block:: python

   import numpy as np
   import sparrow

   null_proteins = sparrow.batch(nulls, ["predictor.radius_of_gyration"])
   null_rg = np.array([protein.predictor.radius_of_gyration() for protein in null_proteins])

When nulls cannot be made
-------------------------

Every model raises :class:`~sparrow.sparrow_exceptions.SparrowException`,
before making anything, if:

* the sequence is empty, is not a string or ``Protein``, or has a residue other
  than the 20 standard amino acids;
* ``count`` is not a positive integer, or ``seed`` is not None or a
  non-negative integer;
* the model needs a background and has not been fit to one, or the model needs
  none and is given one.

Each model also refuses the sequences it cannot make a null of - one too short
for its blocks, or with a residue its background never has - as its own page
describes. A model that can make nulls, but not quite as asked, warns with
:class:`~sparrow.sparrow_exceptions.NullModelWarning` and makes them; its
nulls still have exactly the sequence's length and composition. To make those
warnings errors, or to silence them:

.. code-block:: python

   import warnings
   from sparrow.sparrow_exceptions import NullModelWarning

   warnings.filterwarnings("error", category=NullModelWarning)     # or "ignore"
