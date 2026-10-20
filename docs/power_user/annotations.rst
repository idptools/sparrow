Annotation Functions
====================

The annotations a ``Protein`` gives - low-complexity domains, prion-like domains,
eukaryotic linear motifs and phosphoisoforms (:doc:`../protein/annotations`) -
each come from a module in :mod:`sparrow.sequence_analysis`. Its functions take
a sequence, and some of them do more than the ``Protein`` methods do.

.. list-table::
   :header-rows: 1

   * - What
     - Module
     - Functions
   * - Eukaryotic linear motifs
     - :mod:`sparrow.sequence_analysis.elm`
     - ``find_all_elms``, ``compute_lost_elms``, ``compute_gained_elms``,
       ``compute_retained_elms``, ``parse_hgvs``
   * - Prion-like domains (PLAAC)
     - :mod:`sparrow.sequence_analysis.plaac.plaac`
     - ``score_sequence``, ``score_sequences``
   * - Low-complexity domains
     - :mod:`sparrow.sequence_analysis.sequence_complexity`
     - ``low_complexity_domains_holt``, ``low_complexity_domains_holt_permissive``
   * - Phosphoisoforms
     - :mod:`sparrow.sequence_analysis.phospho_isoforms`
     - ``phosphoisoforms``

What a substitution does to the motifs
--------------------------------------

:func:`~sparrow.sequence_analysis.elm.find_all_elms` is what ``Protein.elms``
calls. To see what a point substitution does to the motifs, give it in HGVS
notation (1-indexed, one- or three-letter codes) or as a ``(position, residue)``
tuple (0-indexed):

.. code-block:: python

   from sparrow import Protein
   from sparrow.sequence_analysis.elm import compute_gained_elms, compute_lost_elms, compute_retained_elms

   p = Protein("MKKRPKSPSGSPTSRRPLL")

   lost = compute_lost_elms(p, "p.S7A")       # or "p.Ser7Ala", or (6, "A")
   sorted(motif.identifier for motif in lost)
   # ['DOC_WW_Pin1_4', 'MOD_DYRK1A_RPxSP_1', 'MOD_GSK3_1', 'MOD_ProDKin_1']

   gained = compute_gained_elms(p, "p.S7A")
   kept = compute_retained_elms(p, "p.S7A")

An HGVS substitution's wild-type residue is checked against the sequence, so
``"p.K7A"`` raises here, where position 7 is S. Two matches are the same motif
when they have the same regular expression, functional site and start position,
so a motif is lost when no match of its class starts in the same place in the
mutant, and every motif lost or gained covers the substituted residue.
:func:`~sparrow.sequence_analysis.elm.parse_hgvs` reads a substitution on its
own.

.. _power-user-plaac:

PLAAC in full
-------------

:func:`~sparrow.sequence_analysis.plaac.plaac.score_sequence` gives every score
the PLAAC program reports - the HMM's prion-like domains and core, FoldIndex
disorder, PAPA prion propensity and the Michelitsch-Weissman Q/N score - as a
:class:`~sparrow.sequence_analysis.plaac.plaac.PLAACResult`, with the 1-based
positions of the original program. For many sequences,
:func:`~sparrow.sequence_analysis.plaac.plaac.score_sequences` builds the HMMs
once and scores them all:

.. code-block:: python

   from sparrow.sequence_analysis.plaac.plaac import score_sequence, score_sequences

   result = score_sequence("MFKSGNQNN" * 20)
   round(result.core_score, 2), result.prd_regions[0].prd_start, result.prd_regions[0].prd_end
   # (33.98, 1, 180)

   results = score_sequences({"A": "MFKSGNQNN" * 20, "B": "ACDEFGHIKLMNPQRSTVWY" * 10})

To score against a different background than the *S. cerevisiae* proteome, pass
``bg_freqs`` and lower ``alpha`` below 1: at ``alpha=1`` the *S. cerevisiae*
background is used whatever ``bg_freqs`` says, so ``score_sequence`` raises
rather than ignore it. PLAAC also runs from the command line on a FASTA file,
as the original does::

   python -m sparrow.sequence_analysis.plaac.plaac -i my_sequences.fasta

If you use PLAAC, please cite it: Lancaster AK, Nutter-Upham A, Lindquist S, King
OD. PLAAC: a web and command-line application to identify proteins with prion-like
amino acid composition. *Bioinformatics.* 2014;30(17):2501-2502.

Low-complexity domains
----------------------

The two searches ``Protein.low_complexity_domains`` chooses between with
``mode`` are functions of a sequence:

.. code-block:: python

   from sparrow.sequence_analysis.sequence_complexity import (
       low_complexity_domains_holt,
       low_complexity_domains_holt_permissive,
   )

   sequence = "QQQQQQQQ" + "AAAAAAAA" + "QQQQQQQQ"
   low_complexity_domains_holt(sequence, "Q", minimum_length=5)
   # [['QQQQQQQQ', 0, 8], ['QQQQQQQQ', 16, 24]]  - the eight As split them
   low_complexity_domains_holt_permissive(sequence, "Q", minimum_length=5)
   # [['QQQQQQQQAAAAAAAAQQQQQQQQ', 0, 24]]  - 16/24 = 67% Q passes 0.25
   low_complexity_domains_holt_permissive(sequence, "Q", minimum_length=5, fractional_threshold=0.7)
   # [['QQQQQQQQ', 0, 8], ['QQQQQQQQ', 16, 24]]  - but not 0.7

:mod:`sparrow.sequence_analysis.sequence_complexity` describes the search both
of them run.

Phosphoisoforms
---------------

:func:`~sparrow.sequence_analysis.phospho_isoforms.phosphoisoforms` is what
``Protein.generate_phosphoisoforms`` calls, and takes the ``Protein``:

.. code-block:: python

   from sparrow.sequence_analysis.phospho_isoforms import phosphoisoforms

   isoforms = phosphoisoforms(Protein("MKSPEEKTPRYS"), mode="custom", phosphosites=[2, 7])
   # the 4 isoforms of two sites

Reference
---------

The four modules are documented in full in :doc:`../api/sequence_analysis`.
