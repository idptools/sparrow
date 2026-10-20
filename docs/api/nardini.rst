``sparrow.nardini``
===================

.. automodule:: sparrow.nardini
   :no-members:
   :no-inherited-members:

Everything below can also be imported from :mod:`sparrow.nardini` itself, which
re-exports the public functions, classes and constants of its submodules.
:doc:`../power_user/nardini` explains them and shows them in use, and
:doc:`../protein/nardini` covers ``Protein.nardini``.

The ``Protein.nardini`` accessor
--------------------------------

.. automodule:: sparrow.nardini.protein_interface
   :no-members:
   :no-inherited-members:

.. autoclass:: sparrow.nardini.protein_interface.ProteinNardini
   :members:
   :no-index:

Vector calculation
------------------

.. automodule:: sparrow.nardini.vectors
   :no-members:
   :no-inherited-members:

.. autofunction:: sparrow.nardini.vectors.nardini_vectors

.. autoclass:: sparrow.nardini.vectors.NardiniResult
   :members:

Labelled containers
-------------------

.. automodule:: sparrow.nardini.table
   :no-members:
   :no-inherited-members:

.. autoclass:: sparrow.nardini.table.FeatureVector
   :members:

.. autoclass:: sparrow.nardini.table.FeatureTable
   :members:

Background construction
-----------------------

.. automodule:: sparrow.nardini.background
   :no-members:
   :no-inherited-members:

.. autofunction:: sparrow.nardini.background.build_background

.. autofunction:: sparrow.nardini.background.load_reference_background

.. autoclass:: sparrow.nardini.background.NardiniBackground
   :members:

.. autodata:: sparrow.nardini.background.REFERENCE_SPECIES

.. autodata:: sparrow.nardini.background.MIN_RECOMMENDED_SEQUENCES

.. autodata:: sparrow.nardini.background.MAX_DEGENERATE_FRACTION

GIN cluster assignment
----------------------

.. automodule:: sparrow.nardini.gin
   :no-members:
   :no-inherited-members:

.. autofunction:: sparrow.nardini.gin.assign_gin_clusters

.. autofunction:: sparrow.nardini.gin.gin_cluster_definitions

.. autofunction:: sparrow.nardini.gin.load_gin_reference

.. autodata:: sparrow.nardini.gin.GIN_REFERENCES

.. autodata:: sparrow.nardini.gin.DEFAULT_GIN_REFERENCE

.. autodata:: sparrow.nardini.gin.GIN_CLIP

.. autodata:: sparrow.nardini.gin.TYPICAL_COMPOSITION_RMS_Z

.. autodata:: sparrow.nardini.gin.TYPICAL_MAX_SATURATED

.. autoclass:: sparrow.nardini.gin.GINAssignment
   :members:

.. autoclass:: sparrow.nardini.gin.GINReference
   :members:

Patterning
----------

.. automodule:: sparrow.nardini.patterning
   :members: patterning_zscores, patterning_values, resolve_backend,
             encode_types, type_fractions, PATTERNING_BACKENDS

Composition
-----------

.. automodule:: sparrow.nardini.composition
   :members: composition_values, composition_matrix, composition_feature_names,
             PATCH_INTERRUPTION, PATCH_MIN_TARGET_COUNT, PATCH_RESIDUES,
             RG_PATCH_RESIDUES, RG_MOTIF, RG_MIN_ADJACENT_PAIRS

Clustering
----------

.. automodule:: sparrow.nardini.clustering
   :members: build_gin_reference, compare_gin_references, describe_centroid,
             MIN_SEQUENCES_PER_CLUSTER, DEFAULT_NUM_CLUSTERS

Parallel computation
--------------------

.. automodule:: sparrow.nardini.batch
   :no-members:
   :no-inherited-members:

.. autofunction:: sparrow.nardini.batch.nardini_proteins

.. autofunction:: sparrow.nardini.batch.score_in_parallel

.. autofunction:: sparrow.nardini.batch._default_start_method

Residue groupings
-----------------

.. automodule:: sparrow.nardini.groups
   :members: resolve_typeset, type_names, patterning_feature_names

Warnings
--------

The advisory warnings NARDINI+ gives -
:class:`~sparrow.sparrow_exceptions.BackgroundQualityWarning` and
:class:`~sparrow.sparrow_exceptions.ClusteringQualityWarning` - are documented
with sparrow's exceptions, in :doc:`exceptions`.
