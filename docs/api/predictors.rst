``sparrow.predictors``
======================

.. automodule:: sparrow.predictors
   :no-members:
   :no-inherited-members:

:doc:`../protein/predictor` explains each prediction, and
:doc:`../power_user/networks` running the networks yourself.
:doc:`../adding_a_predictor` explains how the registries below are used to
add one. Each network's predictor class is on :doc:`predictor_networks`.

The ``Predictor`` accessor
--------------------------

.. autoclass:: sparrow.predictors.Predictor
   :members:
   :no-index:

Loading a network's predictor
-----------------------------

.. autofunction:: sparrow.predictors.load_network_predictor

.. autofunction:: sparrow.predictors.network_predictor_class

The registries
--------------

.. autodata:: sparrow.predictors.NETWORK_PREDICTORS
   :no-value:

.. autodata:: sparrow.predictors.METHOD_NETWORKS
   :no-value:

.. autodata:: sparrow.predictors.METAPREDICT_METHODS

.. autodata:: sparrow.predictors.BATCH_METHODS
   :no-value:

.. autodata:: sparrow.predictors.SCALED_NETWORK_METHODS

.. autodata:: sparrow.predictors.SCALED_NETWORK_PREFIX

``sparrow.predictors.batch_predictor``
--------------------------------------

.. automodule:: sparrow.predictors.batch_predictor
   :members:

``sparrow.predictors.devices``
------------------------------

.. automodule:: sparrow.predictors.devices
   :members:

``sparrow.predictors.base``
---------------------------

.. automodule:: sparrow.predictors.base
   :private-members: _from_raw, _network_input

``sparrow.predictors.network_loader``
-------------------------------------

.. automodule:: sparrow.predictors.network_loader
   :exclude-members: number_of_classes, input_size, number_of_layers, hidden_vector_size, device,
                     saved_by_parrot2, datatype

``sparrow.predictors.outputs``
------------------------------

.. automodule:: sparrow.predictors.outputs

``sparrow.predictors.predictor_template``
-----------------------------------------

.. automodule:: sparrow.predictors.predictor_template
   :private-members: _from_raw

``sparrow.predictors.batch_predict``
------------------------------------

.. automodule:: sparrow.predictors.batch_predict
   :no-members:
   :no-inherited-members:
