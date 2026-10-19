Motifs, Domains and Phosphoisoforms
===================================

Beyond whole-sequence properties, sparrow finds four kinds of local sequence
feature. Each is available from a :class:`~sparrow.protein.Protein` and as plain
functions in ``sparrow.sequence_analysis``:

.. list-table::
   :header-rows: 1

   * - What
     - From a ``Protein``
     - Module
   * - Eukaryotic linear motifs (ELMs)
     - ``p.elms``
     - :mod:`sparrow.sequence_analysis.elm`
   * - Prion-like domains (PLAAC)
     - ``p.plaac_prion_like_domains()``
     - :mod:`sparrow.sequence_analysis.plaac.plaac`
   * - Low-complexity domains
     - ``p.low_complexity_domains()``, ``p.low_complexity_mask()``
     - :mod:`sparrow.sequence_analysis.sequence_complexity`
   * - Phosphoisoforms
     - ``p.generate_phosphoisoforms()``
     - :mod:`sparrow.sequence_analysis.phospho_isoforms`

Positions are 0-indexed with an exclusive end, so ``sequence[start:end]`` is the
feature -- except in PLAAC's own result objects, which keep the 1-based positions
of the original PLAAC program. ``Protein.plaac_prion_like_domains()`` converts
them.

Eukaryotic linear motifs
------------------------

ELMs are short linear motifs -- degrons, docking sites, modification sites,
targeting signals -- each defined by a regular expression in the
`ELM resource <http://elm.eu.org>`_. sparrow bundles the ELM class table
(download version 1.4, 2 April 2024, 349 classes) and reports every match of
every class:

.. code-block:: python

   from sparrow import Protein

   p = Protein("MKKRPKSPSGSPTSRRPLL")
   for motif in sorted(p.elms, key=lambda m: (m.start, m.identifier))[:3]:
       print(motif.identifier, motif.start, motif.end, motif.sequence)

.. code-block:: text

   DEG_Nend_UBRbox_1 0 4 MKKR
   TRG_NLS_MonoExtC_3 0 6 MKKRPK
   DOC_USP7_UBL2_3 1 6 KKRPK

A match is a candidate, not evidence of a functional site: many ELM patterns are
short enough to match by chance, and each class's ``probability`` says how likely
that is. ``p.elms`` is a set, computed once and cached. Each class is tried at
every start position, so matches of one class that overlap are all reported.

To see what a point substitution does to the motifs, give it in HGVS notation
(1-indexed, one- or three-letter codes) or as a ``(position, residue)`` tuple
(0-indexed):

.. code-block:: python

   from sparrow.sequence_analysis.elm import compute_gained_elms, compute_lost_elms, compute_retained_elms

   lost = compute_lost_elms(p, "p.S7A")       # or "p.Ser7Ala", or (6, "A")
   sorted(motif.identifier for motif in lost)
   # ['DOC_WW_Pin1_4', 'MOD_DYRK1A_RPxSP_1', 'MOD_GSK3_1', 'MOD_ProDKin_1']

An HGVS substitution's wild-type residue is checked against the sequence, so
``"p.K7A"`` raises here, where position 7 is S. Two matches are the same motif
when they have the same regular expression, functional site and start position,
so a motif is lost when no match of its class starts in the same place in the
mutant, and every motif lost or gained covers the substituted residue.

If you use these annotations, please cite the ELM resource: Kumar M, et al.
ELM -- the Eukaryotic Linear Motif resource -- 2024 update. *Nucleic Acids Res.*
2024;52(D1):D442-D455.

Prion-like domains (PLAAC)
--------------------------

PLAAC finds prion-like domains with a hidden Markov model whose prion-like state
is trained on the composition of 28 *S. cerevisiae* prion domains (Alberti et al.
2009), against a background of the *S. cerevisiae* proteome by default. sparrow
carries a pure-Python port of the PLAAC program, which also reports FoldIndex
disorder, PAPA prion propensity and the Michelitsch-Weissman Q/N score.

.. code-block:: python

   from sparrow import Protein
   from sparrow.sequence_analysis.plaac.plaac import score_sequence

   p = Protein("MFKSGNQNN" * 20)
   p.plaac_prion_like_domains()          # [[sequence, 0, 180]]: 0-indexed, end exclusive

   result = score_sequence(p.sequence)   # every PLAAC score, 1-based positions
   round(result.core_score, 2), result.prd_regions[0].prd_start, result.prd_regions[0].prd_end
   # (33.98, 1, 180)

``p.plaac_prion_like_domains(simple=False)`` returns the same
:class:`~sparrow.sequence_analysis.plaac.plaac.PLAACResult`. To use a different
background, pass ``bg_freqs`` and lower ``alpha`` below 1 -- at ``alpha=1`` the
*S. cerevisiae* background is used whatever ``bg_freqs`` says, so
:func:`~sparrow.sequence_analysis.plaac.plaac.score_sequence` raises rather than
ignore it. PLAAC also runs from the command line on a FASTA file, as the original
does::

   python -m sparrow.sequence_analysis.plaac.plaac -i my_sequences.fasta

If you use PLAAC, please cite it: Lancaster AK, Nutter-Upham A, Lindquist S, King
OD. PLAAC: a web and command-line application to identify proteins with prion-like
amino acid composition. *Bioinformatics.* 2014;30(17):2501-2502.

Low-complexity domains
----------------------

A low-complexity domain (LCD) here is a region rich in residues from a chosen
set: it starts and ends at a chosen residue, is at least ``minimum_length``
long, and the chosen residues make up more than ``fractional_threshold`` of its
length. The approach follows Gutierrez et al. from the Holt lab (*eLife* 2022),
hence the mode names:

.. code-block:: python

   p = Protein("MKKRPKSPSGEEEQQQQQQQQQQQQQQQQASDLEKRK")
   p.low_complexity_domains(residue_selector="Q", minimum_length=10)
   # [['QQQQQQQQQQQQQQQQ', 13, 29]]

In ``mode='holt'`` (the default) a run of more than ``max_interruption`` other
residues (5 by default; Gutierrez et al. used 17) always ends a domain. In
``mode='holt-permissive'`` there is no such limit, so a run of any length can sit
inside a domain, and ``fractional_threshold`` alone decides how far a domain
extends:

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

``p.low_complexity_mask()`` takes the same arguments and gives the same domains
as a 1 or 0 for each residue, which is easier to plot along the sequence or to
line up with a per-residue prediction:

.. code-block:: python

   p = Protein("SSQQAQQQSS")
   p.low_complexity_mask(residue_selector="Q", minimum_length=5)
   # [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  - the A between the Qs is inside the domain

:meth:`~sparrow.protein.Protein.low_complexity_mask` sets out how the domains
are found, step by step.

Phosphoisoforms
---------------

A phosphoisoform replaces some of the phosphosites with glutamate, the usual
phosphomimetic. Computing a property over every isoform shows how far
phosphorylation could move it:

.. code-block:: python

   p = Protein("MKSPEEKTPRYS")
   isoforms = p.generate_phosphoisoforms()     # 16: four S/T/Y sites, each on or off
   ncpr = [Protein(isoform).NCPR for isoform in isoforms]
   round(p.NCPR, 3), round(min(ncpr), 3)
   # (0.083, -0.25)

By default every S, T and Y is a candidate site. ``mode='predict'`` uses the sites
sparrow's phosphorylation predictors call, and ``mode='custom'`` takes a list of
0-indexed positions. A sequence with ``n`` sites has ``2**n`` isoforms, so for
long sequences restrict the sites, or cap how many may be phosphorylated at once
with ``phospho_rate``.

Reference: ELMs
---------------

.. automodule:: sparrow.sequence_analysis.elm
   :exclude-members: ELM

.. The dataclass fields are documented by the Attributes section of each class
   docstring, so they are excluded here rather than documented twice.

.. autoclass:: sparrow.sequence_analysis.elm.ELM
   :exclude-members: regex, identifier, functional_site_name, description,
                     probability, start, end, sequence

Reference: PLAAC
----------------

.. py:module:: sparrow.sequence_analysis.plaac.plaac

The programmatic entry points and the objects they return. The rest of the module
-- the HMM, FoldIndex and PAPA machinery and the command-line program -- follows
the original Java program.

.. autofunction:: sparrow.sequence_analysis.plaac.plaac.score_sequence

.. autofunction:: sparrow.sequence_analysis.plaac.plaac.score_sequences

.. autoclass:: sparrow.sequence_analysis.plaac.plaac.PLAACResult
   :exclude-members: sequence, protein_length, mw_score, mw_start, mw_end,
                     mw_length, llr, llr_start, llr_end, llr_length, nllr,
                     viterbi_max_run, core_score, core_start, core_end,
                     core_length, core_seq, prd_score, prd_start, prd_end,
                     prd_length, prd_seq, prd_start_seq, prd_end_seq, hmm_all,
                     hmm_vit, fi_num_disordered, fi_mean_hydro, fi_mean_charge,
                     fi_mean_combo, fi_max_run, papa_combo, papa_prop, papa_fi,
                     papa_llr, papa_llr2, papa_center, papa_seq, disorder,
                     viterbi_path, map_path, posterior, prd_regions

.. autoclass:: sparrow.sequence_analysis.plaac.plaac.PRDRegion
   :exclude-members: prd_start, prd_end, prd_length, prd_seq, prd_score,
                     core_start, core_end, core_length, core_seq, core_score

.. autoclass:: sparrow.sequence_analysis.plaac.plaac.DisorderReport
   :exclude-members: charge, hydro, fi, plaac_llr, papa, fix2, plaacllr_x2,
                     papa_x2, mean_charge, mean_hydro, mean_fi,
                     num_disordered_strict, max_disorder_run, papa_max_score,
                     papa_max_prop, papa_max_fi, papa_max_llr, papa_max_llr2,
                     papa_max_center

Reference: Low-complexity domains
---------------------------------

.. automodule:: sparrow.sequence_analysis.sequence_complexity
   :members: low_complexity_domains_holt, low_complexity_domains_holt_permissive

Reference: Phosphoisoforms
--------------------------

.. automodule:: sparrow.sequence_analysis.phospho_isoforms
   :members: phosphoisoforms
