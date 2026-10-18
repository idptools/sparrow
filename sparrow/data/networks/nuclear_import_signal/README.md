# Nuclear import signals (`networks/nuclear_import_signal`)

The network that predicts, for each residue, the probability that it is in a nuclear import (localization) signal (NLS), rounded to 5 decimal places.

## Files and how sparrow uses them

| File | Version | LSTM layers | Hidden size |
|---|---|---|---|
| `nls_predictor_network_v1.pt` | 1 (default) | 1 | 10 |

Each file holds the weights of a bidirectional LSTM network (BRNN) trained with PARROT 1 on one-hot encoded sequences. It is many-to-many, with outputs at every residue, and a classifier over 2 classes.

`NLSPredictor`, in `sparrow/predictors/nls/nuclear_import_signal_predictor.py`, loads it, and sparrow registers it as the `"nls"` network in `sparrow.predictors.NETWORK_PREDICTORS`. The predictor's `_from_raw` makes the network's prediction from its output, the same way for one sequence or a batch, and sparrow runs it through:

* `Protein(seq).predictor.nuclear_import_signal()`, which returns the prediction or a value made from it, and for many sequences `sparrow.batch(sequences, ["predictor.nuclear_import_signal"])`, which runs the network in batches;
* `batch_predict(sequences, network="nls")`, from `sparrow.predictors.batch_predictor`, which returns the prediction for many sequences, in batches, and takes `version=`;
* `NLSPredictor(version=...).predict(seq)` and `.predict_many(sequences)`, to use a version other than the default directly.

The [networks README](../README.md) describes how the versioned files are named and loaded.
