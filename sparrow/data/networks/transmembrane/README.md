# Transmembrane regions (`networks/transmembrane`)

The network that predicts, for each residue, whether it is in a transmembrane region (1) or not (0). It was trained on the transmembrane regions UniProt annotates.

## Files and how sparrow uses them

| File | Version | LSTM layers | Hidden size |
|---|---|---|---|
| `transmembrane_predictor_network_v1.pt` | 1 | 1 | 10 |
| `transmembrane_predictor_network_v4.pt` | 4 (default) | 2 | 10 |

Each file holds the weights of a bidirectional LSTM network (BRNN) trained with PARROT 1 on one-hot encoded sequences. It is many-to-many, with outputs at every residue, and a classifier over 2 classes.

`TransmembranePredictor`, in `sparrow/predictors/transmembrane/transmembrane_predictor.py`, loads it, and sparrow registers it as the `"transmembrane"` network in `sparrow.predictors.NETWORK_PREDICTORS`. The predictor's `_from_raw` makes the network's prediction from its output, the same way for one sequence or a batch, and sparrow runs it through:

* `Protein(seq).predictor.transmembrane_regions()`, which returns the prediction or a value made from it, and for many sequences `sparrow.batch(sequences, ["predictor.transmembrane_regions"])`, which runs the network in batches;
* `batch_predict(sequences, network="transmembrane")`, from `sparrow.predictors.batch_predictor`, which returns the prediction for many sequences, in batches, and takes `version=`;
* `TransmembranePredictor(version=...).predict(seq)` and `.predict_many(sequences)`, to use a version other than the default directly.

The [networks README](../README.md) describes how the versioned files are named and loaded.
