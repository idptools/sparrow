Predictions (``predictor.``)
============================

The methods of ``protein.predictor`` (:doc:`../protein/predictor`). Every one of
them is predicted for the whole batch at once: sparrow's networks in batched
forward passes, each network once however many of its methods are asked for,
and metapredict's disorder and pLDDT with metapredict's own batch functions.
``batch_size``, ``device`` and ``gpuid`` say how (:doc:`index`).

.. include:: _analyses_predictor.rst
