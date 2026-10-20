Reading, Fetching and Comparing Sequences
=========================================

Reading FASTA files
-------------------

:func:`sparrow.read_fasta` reads a FASTA file into a dict of ``Protein``
objects, keyed by header. It is a wrapper around `protfasta
<https://protfasta.readthedocs.io/>`_'s ``read_fasta``, and takes the same
keyword arguments, for duplicate headers and sequences and for invalid
residues:

.. code-block:: python

   from sparrow import read_fasta

   proteins = read_fasta("my_sequences.fasta")
   proteins = read_fasta("my_sequences.fasta", invalid_sequence_action="convert")

:func:`sparrow.batch` reads a FASTA file this way when it is given a path; to
pass ``read_fasta`` options, read the file yourself and pass the dict.

Fetching from UniProt
---------------------

:func:`~sparrow.tools.io.uniprot_fetch` fetches one sequence from UniProt by
accession and returns it as a ``Protein``, or None if it cannot be fetched. It
makes one web request, so it is for interactive use; for many sequences,
download them from UniProt in bulk.

.. code-block:: python

   from sparrow.tools.io import uniprot_fetch

   p = uniprot_fetch("P04637")       # human p53

Comparing sequences
-------------------

:func:`~sparrow.tools.general_tools.compare_sequence` counts the positions at
which two sequences of the same length differ, or lists them, optionally
ignoring gaps, as in an alignment:

.. code-block:: python

   from sparrow.tools.general_tools import compare_sequence

   compare_sequence("MEEEKKKK", "MEDEKRKK")                          # 2
   compare_sequence("MEEEKKKK", "MEDEKRKK", return_positions=True)   # [2, 5]
   compare_sequence("ME-EKKKK", "MEDEKRKK", ignore_gaps=True)        # 1

Compressing sequences
---------------------

:class:`~sparrow.sequence_analysis.compression.KMerCompressor` compresses an
amino-acid sequence with an LZ78-style dictionary of variable-length k-mers,
and decompresses it again:

.. code-block:: python

   from sparrow.sequence_analysis.compression import KMerCompressor

   compressor = KMerCompressor()
   tokens = compressor.compress("MEEEKKKKSSSTTTDDDMEEEKKKK")
   compressor.decompress(tokens)      # 'MEEEKKKKSSSTTTDDDMEEEKKKK'

Reference
---------

These are documented in full in :doc:`../api/sparrow` and :doc:`../api/tools`.
