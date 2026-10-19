Batch Prediction (``batch_predict``)
====================================

:func:`sparrow.predictors.batch_predictor.batch_predict` runs one of sparrow's
networks over a whole set of sequences in batched forward passes - the
ALBATROSS polymer-dimension networks, which give one value per sequence, and the
per-residue networks for secondary structure, transmembrane regions,
mitochondrial targeting, localization signals, transactivation domains, PScore
and phosphorylation. For large collections this is substantially faster than
constructing one ``Protein`` per sequence and calling its ``predictor``: over 300
yeast proteins, 7 to 16 times faster for each network.

To get ``Protein`` objects with these predictions already made, use
:func:`sparrow.batch` (:doc:`many_sequences`), which runs the same networks the
same way for you, and metapredict's disorder and pLDDT predictions in
metapredict's own batches.

Quick start
-----------

.. code-block:: python

   from sparrow.predictors.batch_predictor import batch_predict

   sequences = {
       "p1": "MEEEKKKKSSSTTTDDD",
       "p2": "GRGRGGYGGRGGYGGSRGGYGG",
       "p3": "QQQQQQAASSSSTTTTQQQQQ",
   }

   results = batch_predict(sequences, network="rg", batch_size=64)
   #   {"p1": ["MEEEKKKKSSSTTTDDD", 10.44...], "p2": [...], "p3": [...]}

   helix = batch_predict(sequences, network="dssp")
   #   {"p1": ["MEEEKKKKSSSTTTDDD", array of shape (17, 3)], ...}

The networks
------------

Each network's prediction is what its predictor makes for one sequence. The
``Protein.predictor`` methods return these, or values made from them.

.. list-table::
   :header-rows: 1
   :widths: 28 42 30

   * - ``network``
     - Prediction for each sequence
     - ``Protein.predictor`` method
   * - ``"rg"``, ``"scaled_rg"``
     - The radius of gyration, in Angstroms
     - ``radius_of_gyration()``
   * - ``"re"``, ``"scaled_re"``
     - The end-to-end distance, in Angstroms
     - ``end_to_end_distance()``
   * - ``"asphericity"``, ``"prefactor"``, ``"scaling_exponent"``
     - One value
     - ``asphericity()``, ``prefactor()``, ``scaling_exponent()``
   * - ``"dssp"``
     - The probability of helix, extended and coil at each residue, an array
       of shape ``(len(sequence), 3)``
     - ``dssp_helicity()``, ``dssp_extended()`` and ``dssp_coil()``, which
       classify one column each
   * - ``"transmembrane"``
     - The predicted class of each residue
     - ``transmembrane_regions()``
   * - ``"mitochondrial_targeting"``
     - 1 for each residue in a predicted targeting sequence, else 0; only the
       first 168 residues are predicted
     - ``mitochondrial_targeting_sequence()``
   * - ``"nes"``, ``"nls"``, ``"tad"``
     - The probability that each residue is in a nuclear export signal, a
       nuclear import signal or a transactivation domain
     - ``nuclear_export_signal()``, ``nuclear_import_signal()``,
       ``transactivation_domains()``
   * - ``"pscore"``
     - The phase-separation propensity of each residue, on its native 0-6 scale
     - ``pscore()``, which rescales it to 0-1 by default
   * - ``"ser_phosphorylation"``, ``"thr_phosphorylation"``,
       ``"tyr_phosphorylation"``
     - The probability that each residue is phosphorylated
     - ``serine_phosphorylation()`` and its relatives, which apply a threshold

How exact the predictions are
-----------------------------

A prediction made in a batch is not always the one made for the sequence alone
to the last bit: the networks compute in 32-bit arithmetic, and a batch is
summed in a different order than one sequence alone. Over 300 yeast proteins
(114,436 residues), with the default algorithm:

* the ALBATROSS values differ by at most 3.3e-7 relative;
* the DSSP probabilities by at most 4.5e-6, and the PScore by at most 8.8e-6;
* the transmembrane and mitochondrial-targeting classes not at all;
* the signal, domain and phosphorylation probabilities, which are rounded to 5
  decimal places, by one rounded place (1e-5) in at most 501 residues. The
  largest difference, 9e-5 in one 632-residue protein, is in a prediction the
  network itself makes only to about 1e-4 in 32-bit arithmetic: alone, it is
  9.4e-5 from the same network run in 64-bit arithmetic, and in its batch,
  1.3e-4.

Inputs and outputs
------------------

* **Input** -- a ``dict`` (keys preserved in the output) or a ``list`` (mapped to
  0-based integer keys ``0..n-1``). Values may be sequence strings or
  :class:`sparrow.protein.Protein` objects, but they must be all one or all the
  other -- a mixture raises a ``SparrowException``.
* **Output (default)** -- a ``dict`` mapping each input key to
  ``[sequence, prediction]``.
* **Output (``return_seq2prediction=True``)** -- a ``dict`` mapping each unique
  sequence directly to its prediction. Note that duplicate sequences collapse
  into a single key here, so this mode loses the key/sequence mapping; use the
  default if you need it.
* **Order** -- the returned dictionary preserves input order.

.. code-block:: python

   # sequence -> prediction
   seq2rg = batch_predict(sequences, network="rg", return_seq2prediction=True)

   # list input -> integer-keyed output
   from_list = batch_predict(["MEEEKKKKSSSTTTDDD", "QQQQQQAASSSSTTTTQQQQQ"], network="asphericity")

Key options
-----------

* ``network`` -- which network to run (see the table above). Required.
* ``batch_size`` -- forward-pass batch size (default 32). Larger values are
  faster on GPU; results depend on it only at the networks' 32-bit precision
  (1.4e-7 relative at most, between batch sizes 8 and 64). Should be no larger than
  the batch size used during training.
* ``version`` -- network version (default None, each network's own default; 2,
  released May 21st 2023, for the ALBATROSS networks). Version 1 weights also
  ship with sparrow for reproducing older results.
* ``device`` -- where to run the network, the same choice PARROT's ``Predictor``
  takes: ``"auto"`` (the default), ``"cpu"``, ``"cuda"`` or ``"mps"``, or a
  device as torch names it, such as ``"cuda:1"``. ``"auto"`` runs on an NVIDIA
  GPU where there is one and on the CPU otherwise; it never chooses an
  Apple-silicon GPU, on which sparrow's networks run 4.5 to 7 times slower than
  on the CPU, but ``"mps"`` runs on one. A device the machine does not have is an
  error, not a silent fall back to the CPU. ``prepare_model``, ``predict_many``
  and :func:`sparrow.batch` take it too, and the predictor classes take it as
  their ``device``.
* ``gpuid`` -- which NVIDIA GPU to run on (default None, the current one), as
  PARROT's ``gpu_id`` chooses. With ``device="auto"`` it is used only where CUDA
  is available, so code written for a GPU runs unchanged on a laptop; with
  ``"cpu"`` or ``"mps"`` it is an error.
* ``batch_algorithm`` -- ``"default"`` (recommended) selects ``"pad-n-pack"`` on
  PyTorch 1.11+, which batches mixed-length sequences together via
  ``pack_padded_sequence``, and falls back to ``"size-collect"`` on older
  versions. ``"size-collect"`` groups equal-length sequences instead. Both give
  the same values, to the precision above. Requesting ``"pad-n-pack"`` explicitly
  on PyTorch older than 1.11 raises a ``SparrowException``.
* ``return_seq2prediction`` -- switch the return shape (see above). Default
  ``False``.
* ``show_progress_bar`` -- print a ``tqdm`` progress bar as batches complete.
  Default ``True``.
* ``safe`` -- for ``rg``/``re``, automatically routes sequences shorter than
  ``sparrow.data.configs.MIN_LENGTH_ALBATROSS_RE_RG`` (35 residues) to the
  corresponding scaled network, which is substantially more accurate there. Leave
  this ``True`` (the default) unless you have a specific reason not to. Has no
  effect for the other networks.

How it works
------------

Every network is run in batches by its predictor's ``predict_many``, which
``batch_predict`` and :func:`sparrow.batch` both call, and is used the same way
whether it was trained with PARROT 1 or PARROT 2. ``predict_many`` gets each
sequence's output from the network in a batch, and the predictor's ``_from_raw``
turns that into its prediction, the same way it does for one sequence, so a
prediction is made identically alone or in a batch apart from the arithmetic
above.

* A network saved by PARROT 1 (every network sparrow ships) is run by sparrow's
  batch harness, :func:`~sparrow.predictors.batch_predictor.run_network`, which
  runs a many-to-one network, with one value per sequence, or a many-to-many
  one, with one per residue, over a list of sequences, and returns each
  sequence's raw output in the shape the network gives it for that sequence
  alone. ``pad-n-pack`` (the default) pads each batch to its longest member and
  packs it, so the LSTM runs over each sequence's own residues: the network's
  forward pass, on packed input. It is typically several times faster than
  ``size-collect`` on sequence sets with varied lengths, and equal when all
  lengths match. Inference runs under ``torch.no_grad()`` in evaluation mode,
  and on an NVIDIA GPU in full float32 precision
  (:func:`~sparrow.predictors.devices.full_float32_precision`), as PARROT 2 runs
  its networks: PyTorch otherwise lets cuDNN compute the LSTM in TF32, which
  PARROT measured moving predictions up to ~1e-2 from the CPU's, against ~1e-4
  in full precision.
* A network saved by PARROT 2 is run by PARROT's own ``Predictor.predict_batch``,
  which batches sequences of similar length together and masks the padding, and
  returns each sequence's prediction as PARROT makes it alone, to the same 32-bit
  precision. PARROT batches these networks its own way, so ``batch_algorithm``
  does not apply to them.

API Reference
-------------

.. autofunction:: sparrow.predictors.batch_predictor.batch_predict

.. autofunction:: sparrow.predictors.batch_predictor.prepare_model

.. autofunction:: sparrow.predictors.batch_predictor.run_network

.. automodule:: sparrow.predictors.devices
   :members: resolve_device, full_float32_precision, DEVICE_CHOICES, DEFAULT_DEVICE
