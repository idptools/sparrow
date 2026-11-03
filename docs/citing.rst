Citing sparrow
==============

sparrow gives access to a great deal of other people's work. **If you use a
specific tool, metric, scale or predictor, cite the original paper for it** -
that matters more than citing sparrow itself.

sparrow
-------

If you use the ALBATROSS predictors - the radius of gyration, end-to-end
distance, asphericity, scaling exponent and prefactor - please cite:

  Lotthammer JM, Ginell GM, Griffith D, Emenecker RJ, Holehouse AS. Direct
  prediction of intrinsically disordered protein conformational properties from
  sequence. *Nat Methods.* 2024.

Otherwise, for now, please cite sparrow as "sparrow: **S**\ equence
**PAR**\ ameters for **R**\ egi\ **O**\ ns in **W**\ indows
(https://github.com/idptools/sparrow)".

The work sparrow builds on
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - If you use
     - Please cite
   * - Any deep-learning predictor
     - Griffith D, Holehouse AS. PARROT is a flexible recurrent neural network
       framework for analysis of large protein datasets. *eLife.*
       2021;10:e70576.
   * - Disorder and pLDDT (``predictor.disorder``, ``disorder_domains``,
       ``binary_disorder``, ``pLDDT``)
     - metapredict, as its documentation asks: Emenecker RJ, Griffith D,
       Holehouse AS. Metapredict: a fast, accurate, and easy-to-use predictor
       of consensus disorder and structure. *Biophys J.* 2021;120(20):4312-4319,
       and the paper for the version you use.
   * - ``kappa``, ``compute_kappa_x``
     - Das RK, Pappu RV. Conformations of intrinsically disordered proteins are
       influenced by linear sequence distributions of oppositely charged
       residues. *PNAS.* 2013;110(33):13392-13397.
   * - ``SCD``, ``compute_SCD_x``
     - Sawle L, Ghosh K. A theoretical method to compute sequence dependent
       configurational properties in charged polymers and proteins. *J Chem
       Phys.* 2015;143(8):085101.
   * - ``SHD``, ``compute_SHD_custom``, and ``polymeric.empirical_nu`` and
       ``empirical_radius_of_gyration``
     - Zheng W, Dignon GL, Brown M, Kim YC, Mittal J. Hydropathy patterning
       complements charge patterning to describe conformational preferences of
       disordered proteins. *J Phys Chem Lett.* 2020;11(9):3408-3415.
   * - ``hydrophobicity``
     - Kyte J, Doolittle RF. A simple method for displaying the hydropathic
       character of a protein. *J Mol Biol.* 1982;157(1):105-132.
   * - ``complexity``
     - Wootton JC, Federhen S. Statistics of local complexity in amino acid
       sequences and sequence databases. *Comput Chem.* 1993;17(2):149-163.
   * - ``PPII_propensity``
     - Elam WA, Schrank TP, Campagnolo AJ, Hilser VJ. Evolutionary conservation
       of the polyproline II conformation surrounding intrinsically disordered
       phosphorylation sites. *Protein Sci.* 2013;22(4):405-417.
   * - ``fraction_disorder_promoting``
     - Campen A, Williams RM, Brown CJ, Meng J, Uversky VN, Dunker AK.
       TOP-IDP-scale: a new amino acid scale measuring propensity for intrinsic
       disorder. *Protein Pept Lett.* 2008;15(9):956-963.
   * - An AAindex scale (``linear_property_profile``)
     - The scale's own source, which :doc:`power_user/aaindex` lists for
       each, and Kawashima S, Kanehisa M. AAindex: amino acid index database.
       *Nucleic Acids Res.* 2000.
   * - ``low_complexity_domains``, ``low_complexity_mask``
     - Gutierrez JI, *et al.* SWI/SNF senses carbon starvation with a
       pH-sensitive low-complexity sequence. *eLife.* 2022;11:e70344.
   * - ``plaac_prion_like_domains``
     - Lancaster AK, Nutter-Upham A, Lindquist S, King OD. PLAAC: a web and
       command-line application to identify proteins with prion-like amino
       acid composition. *Bioinformatics.* 2014;30(17):2501-2502.
   * - ``elms``
     - Kumar M, *et al.* ELM - the Eukaryotic Linear Motif resource - 2024
       update. *Nucleic Acids Res.* 2024;52(D1):D442-D455.
   * - The AFRC (``polymeric.afrc_*``)
     - Alston JJ, Ginell GM, Soranno A, Holehouse AS. The Analytical Flory
       Random Coil is a simple-to-use reference model for unfolded and
       disordered proteins. *J Phys Chem B.* 2023;127(21):4746-4760.
   * - The self-avoiding walk (``polymeric.*saw*``)
     - O'Brien EP, Morrison G, Brooks BR, Thirumalai D. How accurate are
       polymer models in the analysis of Forster resonance energy transfer
       experiments on proteins? *J Chem Phys.* 2009;130(12):124903; and, for
       the nu-dependent SAW, Zheng W, Zerze GH, Borgia A, Mittal J, Schuler B,
       Best RB. Inferring properties of disordered chains from FRET transfer
       efficiencies. *J Chem Phys.* 2018;148(12):123329.
   * - NARDINI+ and GIN (``protein.nardini``, :mod:`sparrow.nardini`)
     - The NARDINI, NARDINI+ and GIN papers listed in :ref:`nardini-citations`.
   * - ``SequenceAlignment``
     - Deorowicz S, Debudaj-Grabysz A, Gudyś A. FAMSA: Fast and accurate
       multiple sequence alignment of huge protein families. *Sci Rep.*
       2016;6:33964.
   * - ``KletShuffle`` (:mod:`sparrow.null`)
     - Altschul SF, Erickson BW. Significance of nucleotide sequence
       alignments: a method for random sequence permutation that preserves
       dinucleotide and codon usage. *Mol Biol Evol.* 1985;2(6):526-538; and,
       for the algorithm, Kandel D, Matias Y, Unger R, Winkler P. Shuffling
       biological sequences. *Discrete Appl Math.* 1996;71:171-185, and Jiang
       M, Anderson J, Gillespie J, Mayo M. uShuffle: a useful tool for
       shuffling biological sequences while preserving the k-let counts. *BMC
       Bioinformatics.* 2008;9:192.
   * - ``KmerBackground``, ``MarkovBackground`` (:mod:`sparrow.null`)
     - For conditioning a background on a composition: Csiszár I, Cover TM,
       Choi BS. Conditional limit theorems under Markov conditioning. *IEEE
       Trans Inf Theory.* 1987;33(6):788-801.

Each module's docstring gives the references for what it implements; the
:doc:`api/index` shows them.
