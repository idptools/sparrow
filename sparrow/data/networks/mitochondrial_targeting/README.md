# Mitochondrial targeting sequences (`networks/mitochondrial_targeting`)

The network that predicts, for each residue, whether it is in a mitochondrial targeting sequence (1) or not (0).

## Files and how sparrow uses them

| File | Version | LSTM layers | Hidden size |
|---|---|---|---|
| `mitochondrial_targeting_predictor_network_v1.pt` | 1 (default) | 2 | 10 |

Each file holds the weights of a bidirectional LSTM network (BRNN) trained with PARROT 1 on one-hot encoded sequences. It is many-to-many, with outputs at every residue, and a classifier over 2 classes.

`MitochondrialTargetingPredictor`, in `sparrow/predictors/mitochondrial_targeting/mitochondrial_targeting_predictor.py`, loads it, and sparrow registers it as the `"mitochondrial_targeting"` network in `sparrow.predictors.NETWORK_PREDICTORS`. The predictor's `_from_raw` makes the network's prediction from its output, the same way for one sequence or a batch, and sparrow runs it through:

* `Protein(seq).predictor.mitochondrial_targeting_sequence()`, which returns the prediction or a value made from it, and for many sequences `sparrow.batch(sequences, ["predictor.mitochondrial_targeting_sequence"])`, which runs the network in batches;
* `batch_predict(sequences, network="mitochondrial_targeting")`, from `sparrow.predictors.batch_predictor`, which returns the prediction for many sequences, in batches, and takes `version=`;
* `MitochondrialTargetingPredictor(version=...).predict(seq)` and `.predict_many(sequences)`, to use a version other than the default directly.

## Notes

Mitochondrial targeting sequences are N-terminal, so only the first 168 residues are run through the network, and every residue after them is reported as 0.

The [networks README](../README.md) describes how the versioned files are named and loaded.
