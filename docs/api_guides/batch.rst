Batch Prediction (``batch_predict``)
====================================

:func:`sparrow.predictors.batch_predict.batch_predict` runs a single ALBATROSS
network over a whole set of sequences in batched forward passes. For large
collections this is substantially faster than constructing one ``Protein`` per
sequence and calling its ``predictor``, and it returns identical values.

.. note::
   Batch prediction covers the polymer-dimension networks only:
   ``rg``, ``re``, ``asphericity``, ``scaling_exponent``, ``prefactor``,
   ``scaled_rg``, ``scaled_re``. For other predictors (disorder, DSSP,
   phosphorylation, ...) use the per-protein ``Protein.predictor`` API
   (:doc:`protein`).

Quick start
-----------

.. code-block:: python

   from sparrow.predictors.batch_predict import batch_predict

   sequences = {
       "p1": "MEEEKKKKSSSTTTDDD",
       "p2": "GRGRGGYGGRGGYGGSRGGYGG",
       "p3": "QQQQQQAASSSSTTTTQQQQQ",
   }

   results = batch_predict(sequences, network="rg", batch_size=64)
   #   {"p1": ["MEEE...", 14.7], "p2": [...], "p3": [...]}

Inputs and outputs
------------------

* **Input** -- a ``dict`` (keys preserved in the output) or a ``list`` (mapped to
  integer keys ``0..n``). Values may be sequence strings or
  :class:`sparrow.protein.Protein` objects.
* **Output (default)** -- a ``dict`` mapping each input key to
  ``[sequence, prediction]``.
* **Output (``return_seq2prediction=True``)** -- a ``dict`` mapping each unique
  sequence directly to its prediction.
* **Order** -- the returned dictionary preserves input order.

.. code-block:: python

   # sequence -> prediction
   seq2rg = batch_predict(sequences, network="rg", return_seq2prediction=True)

   # list input -> integer-keyed output
   from_list = batch_predict(["MEEE...", "QQQQ..."], network="asphericity")

Key options
-----------

* ``network`` -- which network to run (see the list above).
* ``batch_size`` -- forward-pass batch size (default 32). Larger values are
  faster on GPU; results are independent of batch size.
* ``version`` -- network version (default 2).
* ``gpuid`` -- GPU index to use when CUDA is available (CPU otherwise).
* ``batch_algorithm`` -- ``"default"`` (recommended) selects ``"pad-n-pack"`` on
  modern PyTorch, which batches mixed-length sequences together via
  ``pack_padded_sequence``; ``"size-collect"`` groups equal-length sequences
  instead. Both give identical values.
* ``safe`` -- for ``rg``/``re``, automatically routes sequences shorter than the
  minimum reliable length to the corresponding scaled network. Leave this
  ``True`` (the default) unless you have a specific reason not to.

Performance notes
-----------------

* ``pad-n-pack`` (the default) is typically several times faster than
  ``size-collect`` on sequence sets with varied lengths, and equal when all
  lengths match.
* Inference runs under ``torch.no_grad()`` in evaluation mode.

API Reference
-------------

.. autofunction:: sparrow.predictors.batch_predict.batch_predict
