``sparrow.sequence_analysis``
=============================

.. py:module:: sparrow.sequence_analysis

Sequence annotations, alignment, compression and the plugin system.
:doc:`../protein/annotations` and :doc:`../power_user/annotations` explain the
annotations, :doc:`../power_user/alignment` the alignment, and
:doc:`../protein/plugins` and :doc:`../writing_a_plugin` the plugins.

``sparrow.sequence_analysis.elm``
---------------------------------

.. automodule:: sparrow.sequence_analysis.elm
   :exclude-members: ELM

.. The dataclass fields are documented by the Attributes section of each class
   docstring, so they are excluded here rather than documented twice.

.. autoclass:: sparrow.sequence_analysis.elm.ELM
   :exclude-members: regex, identifier, functional_site_name, description,
                     probability, start, end, sequence

``sparrow.sequence_analysis.plaac.plaac``
-----------------------------------------

.. py:module:: sparrow.sequence_analysis.plaac

.. py:module:: sparrow.sequence_analysis.plaac.plaac

A pure-Python port of the PLAAC program (Lancaster *et al.*, 2014). These are
the programmatic entry points and the objects they return. The rest of the
module - the HMM, FoldIndex and PAPA machinery and the command-line program -
follows the original Java program.

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

``sparrow.sequence_analysis.sequence_complexity``
-------------------------------------------------

.. automodule:: sparrow.sequence_analysis.sequence_complexity
   :members: low_complexity_domains_holt, low_complexity_domains_holt_permissive

``sparrow.sequence_analysis.phospho_isoforms``
----------------------------------------------

.. automodule:: sparrow.sequence_analysis.phospho_isoforms
   :members: phosphoisoforms

``sparrow.sequence_analysis.alignment``
---------------------------------------

Needs sparrow's ``alignment`` extra (:doc:`../power_user/alignment`).

.. automodule:: sparrow.sequence_analysis.alignment
   :members: SequenceAlignment

``sparrow.sequence_analysis.compression``
-----------------------------------------

.. automodule:: sparrow.sequence_analysis.compression
   :members: KMerCompressor, Token, AMINO_ALPHABET

``sparrow.sequence_analysis.plugins``
-------------------------------------

.. automodule:: sparrow.sequence_analysis.plugins

``sparrow.sequence_analysis.community_plugins.contributed``
-----------------------------------------------------------

.. py:module:: sparrow.sequence_analysis.community_plugins

.. automodule:: sparrow.sequence_analysis.community_plugins.contributed
