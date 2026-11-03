"""Null models: sequences that match a natural sequence with its evolutionary and functional information stripped out.

Each null model keeps something about a sequence and makes the rest random,
and they differ in what they keep:

* :class:`BlockShuffle` - the sequence cut into blocks of ``k`` residues,
  shuffled. Keeps the composition, and the neighbours each residue has inside
  its block. Needs no background.
* :class:`KletShuffle` - a shuffle with exactly the sequence's count of every
  run of ``k`` residues (every dipeptide, with ``k=2``). Needs no background.
* :class:`KmerBackground` - the sequence's composition, rebuilt from blocks
  drawn from the k-mers of a background of natural sequences, so with the
  background's local grammar in place of its own.
* :class:`MarkovBackground` - the sequence's composition, drawn residue by
  residue from a background's Markov chain: the background's grammar again,
  with no joins between blocks.

They all work the same way (see :class:`~sparrow.null.base.NullModel`):

.. code-block:: python

   from sparrow.null import BlockShuffle, KmerBackground

   nulls = BlockShuffle(k=2).generate(sequence, count=10, seed=1)

   model = KmerBackground(k=3).fit("human_idrs.fasta")
   nulls = model.generate(sequence, count=10, seed=1)

Every null has exactly the length and composition of the sequence it was made
from.
"""

from sparrow.null.base import NullModel
from sparrow.null.block_shuffle import BlockShuffle
from sparrow.null.klet_shuffle import KletShuffle
from sparrow.null.kmer_background import KmerBackground
from sparrow.null.markov_background import MarkovBackground

__all__ = ["NullModel", "BlockShuffle", "KletShuffle", "KmerBackground", "MarkovBackground"]
