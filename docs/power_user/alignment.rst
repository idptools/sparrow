Multiple Sequence Alignment
===========================

:class:`~sparrow.sequence_analysis.alignment.SequenceAlignment` aligns a set of
proteins with `FAMSA <https://github.com/refresh-bio/FAMSA>`_ (Deorowicz *et
al.*, 2016), through its Python binding, pyFAMSA. It needs sparrow's
``alignment`` extra, which installs pyFAMSA:

.. code-block:: bash

   pip install "idptools-sparrow[alignment] @ git+https://github.com/idptools/sparrow.git"

Without it, importing :mod:`sparrow.sequence_analysis.alignment` raises an error
saying how to install it; nothing else in sparrow needs pyFAMSA.

.. code-block:: python

   from sparrow import Protein
   from sparrow.sequence_analysis.alignment import SequenceAlignment

   proteins = {
       "seq1": Protein("MGSQSSRSSSQQQ"),
       "seq2": Protein("MGSQSSRNNNQQQ"),
       "seq3": Protein("MGSQSSRSSSAAA"),
   }

   msa = SequenceAlignment(proteins, scoring_matrix="BLOSUM62")
   alignment = msa.alignment                   # aligned once, then cached
   msa.save_msa("example_alignment.fasta")     # the aligned sequences, with gaps
   msa.display_msa()                           # printed, one sequence per line

``SequenceAlignment`` takes a dict of ``Protein`` objects or the path to a FASTA
file. Its other arguments are FAMSA's: the guide tree (``guide_tree``,
``tree_heuristic``, ``medoid_threshold``), the refinement (``refine``,
``n_refinements``), whether to keep duplicate sequences, and the number of
threads.

If you use it, please cite FAMSA: Deorowicz S, Debudaj-Grabysz A, Gudyś A.
FAMSA: Fast and accurate multiple sequence alignment of huge protein families.
*Sci Rep.* 2016;6:33964.

Reference
---------

:class:`~sparrow.sequence_analysis.alignment.SequenceAlignment` is documented in
full in :doc:`../api/sequence_analysis`.
