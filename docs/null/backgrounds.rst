Backgrounds
===========

:doc:`KmerBackground <kmer_background>` and
:doc:`MarkovBackground <markov_background>` learn from a *background*: a large
set of natural sequences, such as a proteome or its IDRs. They count how often
every short run of residues - every k-mer - occurs in it, and give each null
the background's local grammar in place of the sequence's own. The background
is given to :meth:`~sparrow.null.base.NullModel.fit`, once, and the fitted
model then makes nulls of any sequence.

.. code-block:: python

   from sparrow.null import KmerBackground, MarkovBackground

   kmers = KmerBackground(k=3).fit("human_idrs.fasta")
   chain = MarkovBackground(order=2).fit("human_idrs.fasta")

Choosing a background
---------------------

The background is what the nulls are meant to look like, so it is part of the
question a null asks. Its grammar is what a null keeps: nulls from a proteome
have the local grammar of the proteome's proteins, folded domains and all, and
nulls from its IDRs the grammar of its IDRs. A few things are worth keeping in
mind:

* **Like with like.** For IDRs, a background of IDRs - from the same organism,
  if the question is about that organism - asks whether a sequence is unusual
  for an IDR. A whole proteome mixes in the grammar of folded domains, with
  their hydrophobic runs and periodic patterns.
* **The sequence in its own background.** If the sequence, or many close
  relatives of it, is in the background, its k-mers are part of what the
  nulls are drawn from. Over a proteome that hardly matters; over a small or
  family-specific background it can, and the sequence is better left out.
* **Size.** Longer k-mers carry more of the background's grammar, and need a
  larger background to count well; see `How large a background needs to be`_.

What a background can be
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Background
     - How it is read
   * - The path to a FASTA file, as a string or :class:`pathlib.Path`
     - Read with :func:`protfasta.read_fasta`, keeping every record: two
       records with the same header are both counted, as are two with the
       same sequence, and a record with a residue other than the 20 standard
       amino acids is kept, for the counting to deal with (below).
   * - A dict
     - Its values: sequences as strings, or
       :class:`~sparrow.protein.Protein` objects. The keys are not used.
   * - A list, or any other iterable
     - Its items, each a string or a ``Protein``.

Every sequence is upper-cased. A string on its own is always read as the path
to a FASTA file, never as one sequence, so a background of one sequence is a
list of one. ``fit`` raises :class:`~sparrow.sparrow_exceptions.SparrowException`
if the path is not a file, if there are no sequences, or if an item is neither
a string nor a ``Protein``.

.. code-block:: python

   import protfasta
   from sparrow import Protein
   from sparrow.null import KmerBackground

   model = KmerBackground(k=3)
   idrs = protfasta.read_fasta("human_idrs.fasta")

   model.fit("human_idrs.fasta")                           # a FASTA file
   model.fit(idrs)                                         # a dict of sequences
   model.fit(list(idrs.values()))                          # a list of sequences
   model.fit([Protein(idr) for idr in idrs.values()])      # or of Proteins

What is counted
---------------

Fitting counts every overlapping run, of every length from 1 up to the longest
the model needs, in every sequence: ``max(k)`` for ``KmerBackground(k)``, and
``order + 1`` for ``MarkovBackground(order)``. A run is not counted if it spans
the end of one sequence and the start of the next, or a residue other than the
20 standard amino acids. Those residues are not counted at all, and fitting
warns, with :class:`~sparrow.sparrow_exceptions.NullModelWarning`, how many
there were of each:

.. code-block:: text

   NullModelWarning: 800 residues in the background are not one of the 20 standard
   amino acids (B: 400, X: 400); they, and every k-mer that spans one, were not counted.

Nothing else is done to them - they are not dropped from the sequence, joining
what is either side, or replaced - so a run of X in a background breaks it in
two, as far as the counts are concerned.

Fitting the yeast proteome, 6,067 proteins and 2.9 million residues, takes
under 0.1 s for 4-mers. A fitted model can be used for as many sequences as you
like, and fitting it again replaces what it learnt. A fitted model can also be
pickled, but fitting is quick enough that it is usually simpler to fit again.

How large a background needs to be
-----------------------------------

There are 20\ :sup:`k` possible k-mers: 400 dipeptides, 8,000 tripeptides,
160,000 4-mers. Counted in a background, each k-mer's count is roughly
Poisson, so a k-mer seen 10 times has a frequency uncertain by about a third
(1/√10). Fitting warns when the background has, on average, fewer than
:data:`~sparrow.null.kmers.MIN_MEAN_KMER_COUNT` (10) of each possible k-mer of
the longest length it counts:

.. list-table::
   :header-rows: 1
   :widths: 14 22 22 42

   * - k-mer length
     - Possible k-mers
     - Residues needed
     - Models that count it
   * - 2
     - 400
     - about 4,000
     - ``KmerBackground(k=2)``, ``MarkovBackground(order=1)``
   * - 3
     - 8,000
     - about 80,000
     - ``KmerBackground(k=3)``, ``MarkovBackground(order=2)``: the defaults
   * - 4
     - 160,000
     - about 1.6 million
     - ``KmerBackground(k=4)``, ``MarkovBackground(order=3)``

The yeast proteome has 18 of each 4-mer on average, with 4% of them never seen;
the human proteome, about 11 million residues, about 70. The 100 IDRs in
sparrow's tests, 43,502 residues, are too few for 3-mers:

.. code-block:: text

   NullModelWarning: The background has 43302 3-mers, an average of 5.4 for each of
   the 8000 possible ones (at least 10 is recommended), so most of their frequencies
   rest on a handful of counts. Fit KmerBackground to a larger background, or use
   shorter k-mers.

The warning is advice: the model is fit, and makes nulls, from whatever counts
there are. A k-mer the background never has is never drawn, so a sparse
background gives nulls that keep to the few k-mers it happens to have. 4-mers,
:data:`~sparrow.null.kmers.MAX_KMER_SIZE`, are the longest counted: there are
3.2 million 5-mers, which even the human proteome would see about 3 times
each.

Residues the background never has
---------------------------------

A null can only be built from residues the background has. Making nulls of a
sequence with a residue the background never has raises
:class:`~sparrow.sparrow_exceptions.SparrowException`, naming the residue:

.. code-block:: text

   SparrowException: The background has no W, so KmerBackground(k=2) cannot place
   it; fit it to a background with every residue the sequence has.

A background that has every residue, but not in the runs a sequence needs -
no k-mer made only of the residues it has left, near the end of a null - is
dealt with as each model's page describes, with a warning.
