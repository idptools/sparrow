``sparrow.sparrow_exceptions``
==============================

.. automodule:: sparrow.sparrow_exceptions
   :no-members:
   :no-inherited-members:

Exceptions
----------

.. autoexception:: sparrow.sparrow_exceptions.SparrowException
   :no-inherited-members:

.. autoexception:: sparrow.sparrow_exceptions.ProteinException
   :no-inherited-members:

.. autoexception:: sparrow.sparrow_exceptions.PatterningException
   :no-inherited-members:

.. autoexception:: sparrow.sparrow_exceptions.CalculationException
   :no-inherited-members:

Warnings
--------

The advisory warnings are not errors: the call still returns a usable result,
and they flag a result you probably should not trust. They share a base class,
so you can silence them as a group::

    import warnings
    from sparrow.sparrow_exceptions import SparrowWarning

    warnings.filterwarnings("ignore", category=SparrowWarning)

.. autoexception:: sparrow.sparrow_exceptions.SparrowWarning
   :no-inherited-members:

.. autoexception:: sparrow.sparrow_exceptions.BackgroundQualityWarning
   :no-inherited-members:

.. autoexception:: sparrow.sparrow_exceptions.ClusteringQualityWarning
   :no-inherited-members:
