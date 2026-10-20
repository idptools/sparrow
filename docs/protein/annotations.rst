Domains, Motifs and Isoforms
============================

Beyond whole-sequence parameters, a ``Protein`` finds four kinds of local
feature:

.. list-table::
   :header-rows: 1

   * - What
     - From a ``Protein``
   * - Low-complexity domains
     - ``p.low_complexity_domains()``, ``p.low_complexity_mask()``
   * - Prion-like domains (PLAAC)
     - ``p.plaac_prion_like_domains()``
   * - Eukaryotic linear motifs (ELMs)
     - ``p.elms``
   * - Phosphoisoforms
     - ``p.generate_phosphoisoforms()``

Positions are 0-indexed with an exclusive end, so ``sequence[start:end]`` is the
feature. Each is also a plain function in :mod:`sparrow.sequence_analysis`,
which :doc:`../power_user/annotations` covers - along with what the functions
can do that the ``Protein`` methods do not, such as finding the motifs a
substitution gains or loses.

Low-complexity domains
----------------------

A low-complexity domain (LCD) here is a region rich in residues from a chosen
set: it starts and ends at a chosen residue, is at least ``minimum_length``
long, and the chosen residues make up more than ``fractional_threshold`` of its
length. The approach follows Gutierrez *et al.* from the Holt lab (*eLife*
2022), hence the mode names:

.. code-block:: python

   from sparrow import Protein

   p = Protein("MKKRPKSPSGEEEQQQQQQQQQQQQQQQQASDLEKRK")
   p.low_complexity_domains(residue_selector="Q", minimum_length=10)
   # [['QQQQQQQQQQQQQQQQ', 13, 29]]

In ``mode="holt"`` (the default) a run of more than ``max_interruption`` other
residues (5 by default; Gutierrez *et al.* used 17) always ends a domain. In
``mode="holt-permissive"`` there is no such limit, so a run of any length can sit
inside a domain, and ``fractional_threshold`` alone decides how far a domain
extends:

.. code-block:: python

   p = Protein("QQQQQQQQ" + "AAAAAAAA" + "QQQQQQQQ")
   p.low_complexity_domains(residue_selector="Q", minimum_length=5)
   # [['QQQQQQQQ', 0, 8], ['QQQQQQQQ', 16, 24]]  - the eight As split them
   p.low_complexity_domains(mode="holt-permissive", residue_selector="Q", minimum_length=5)
   # [['QQQQQQQQAAAAAAAAQQQQQQQQ', 0, 24]]  - 16/24 = 67% Q passes 0.25

``low_complexity_mask`` takes the same arguments and gives the same domains as a
1 or 0 for each residue, which is easier to plot along the sequence or to line
up with a per-residue prediction:

.. code-block:: python

   p = Protein("SSQQAQQQSS")
   p.low_complexity_mask(residue_selector="Q", minimum_length=5)
   # [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]  - the A between the Qs is inside the domain

:meth:`~sparrow.protein.Protein.low_complexity_mask` sets out how the domains
are found, step by step. ``residue_selector`` has no default, so it must always
be given.

Prion-like domains (PLAAC)
--------------------------

PLAAC finds prion-like domains with a hidden Markov model whose prion-like state
is trained on the composition of 28 *S. cerevisiae* prion domains (Alberti *et
al.* 2009), against a background of the *S. cerevisiae* proteome by default.
sparrow carries a pure-Python port of the PLAAC program.

.. code-block:: python

   p = Protein("MFKSGNQNN" * 20)
   p.plaac_prion_like_domains()             # [[sequence, 0, 180]]: 0-indexed, end exclusive
   p.plaac_prion_like_domains(core_length=60)

   result = p.plaac_prion_like_domains(simple=False)   # every PLAAC score
   round(result.core_score, 2)

``simple=False`` returns the full
:class:`~sparrow.sequence_analysis.plaac.plaac.PLAACResult`, which also reports
FoldIndex disorder, PAPA prion propensity and the Michelitsch-Weissman Q/N
score, and keeps the 1-based positions of the original PLAAC program. Any other
keyword is passed to
:func:`~sparrow.sequence_analysis.plaac.plaac.score_sequence`; see
:ref:`power-user-plaac` for using a different background.

If you use PLAAC, please cite it: Lancaster AK, Nutter-Upham A, Lindquist S, King
OD. PLAAC: a web and command-line application to identify proteins with prion-like
amino acid composition. *Bioinformatics.* 2014;30(17):2501-2502.

Eukaryotic linear motifs
------------------------

ELMs are short linear motifs - degrons, docking sites, modification sites,
targeting signals - each defined by a regular expression in the
`ELM resource <http://elm.eu.org>`_. sparrow bundles the ELM class table
(download version 1.4, 2 April 2024, 349 classes) and reports every match of
every class:

.. code-block:: python

   p = Protein("MKKRPKSPSGSPTSRRPLL")
   for motif in sorted(p.elms, key=lambda m: (m.start, m.identifier))[:3]:
       print(motif.identifier, motif.start, motif.end, motif.sequence)

.. code-block:: text

   DEG_Nend_UBRbox_1 0 4 MKKR
   TRG_NLS_MonoExtC_3 0 6 MKKRPK
   DOC_USP7_UBL2_3 1 6 KKRPK

Each match is an :class:`~sparrow.sequence_analysis.elm.ELM`. A match is a
candidate, not evidence of a functional site: many ELM patterns are short enough
to match by chance, and each class's ``probability`` says how likely that is.
``p.elms`` is a set, computed once and cached. Each class is tried at every start
position, so matches of one class that overlap are all reported.

If you use these annotations, please cite the ELM resource: Kumar M, *et al.*
ELM - the Eukaryotic Linear Motif resource - 2024 update. *Nucleic Acids Res.*
2024;52(D1):D442-D455.

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

By default every S, T and Y is a candidate site. ``mode="predict"`` uses the
sites sparrow's phosphorylation predictors call, and ``mode="custom"`` takes a
list of 0-indexed positions. A sequence with ``n`` sites has ``2**n`` isoforms,
so for long sequences restrict the sites, or cap how many may be phosphorylated
at once with ``phospho_rate``.

Reference
---------

.. autosummary::
   :nosignatures:

   sparrow.protein.Protein.low_complexity_domains
   sparrow.protein.Protein.low_complexity_mask
   sparrow.protein.Protein.plaac_prion_like_domains
   sparrow.protein.Protein.elms
   sparrow.protein.Protein.generate_phosphoisoforms

.. automethod:: sparrow.protein.Protein.low_complexity_domains

.. automethod:: sparrow.protein.Protein.low_complexity_mask

.. automethod:: sparrow.protein.Protein.plaac_prion_like_domains

.. autoproperty:: sparrow.protein.Protein.elms

.. automethod:: sparrow.protein.Protein.generate_phosphoisoforms
