The Network Predictors
======================

Each of sparrow's networks has a predictor class, in its own module, which
loads the network and turns its output into a prediction.
:doc:`../power_user/networks` shows how to use them, and
:data:`~sparrow.predictors.NETWORK_PREDICTORS` maps each network's name to its
class. Every class derives from
:class:`~sparrow.predictors.base.BaseNetworkPredictor` (:doc:`predictors`),
which gives it ``predict`` and ``predict_many``.

Polymer dimensions (ALBATROSS)
------------------------------

.. automodule:: sparrow.predictors.rg.radius_of_gyration_predictor

.. automodule:: sparrow.predictors.scaled_rg.scaled_radius_of_gyration_predictor

.. automodule:: sparrow.predictors.e2e.end_to_end_distance_predictor

.. automodule:: sparrow.predictors.scaled_re.scaled_end_to_end_distance_predictor

.. automodule:: sparrow.predictors.asphericity.asphericity_predictor

.. automodule:: sparrow.predictors.prefactor.prefactor_predictor

.. automodule:: sparrow.predictors.scaling_exponent.scaling_exponent_predictor

Secondary structure
-------------------

.. automodule:: sparrow.predictors.dssp.dssp_predictor

Localization, domains and phase behavior
----------------------------------------

.. automodule:: sparrow.predictors.transmembrane.transmembrane_predictor

.. automodule:: sparrow.predictors.mitochondrial_targeting.mitochondrial_targeting_predictor

.. automodule:: sparrow.predictors.nes.nuclear_export_signal_predictor

.. automodule:: sparrow.predictors.nls.nuclear_import_signal_predictor

.. automodule:: sparrow.predictors.tad.transactivation_domain_predictor

.. automodule:: sparrow.predictors.pscore.pscore_predictor

Phosphorylation
---------------

.. automodule:: sparrow.predictors.phosphorylation.base
   :private-members: _predict_phospho, _from_raw

.. automodule:: sparrow.predictors.phosphorylation.ser_phosphorylation_predictor

.. automodule:: sparrow.predictors.phosphorylation.thr_phosphorylation_predictor

.. automodule:: sparrow.predictors.phosphorylation.tyr_phosphorylation_predictor

.. automodule:: sparrow.predictors.phosphorylation.phospho_predictor_utils
