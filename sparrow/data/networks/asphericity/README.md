# Asphericity (`networks/asphericity`)

The network that predicts a sequence's asphericity, a dimensionless measure of how far its conformational ensemble is from spherical (0 for a sphere, 1 for a rod). It is one of the ALBATROSS networks (Lotthammer et al., *Nat. Methods* 2024), which predict the conformational properties of disordered regions from sequence. Version 2 is the ALBATROSS release of May 21st 2023 and is the default; version 1 is kept so older results can be reproduced.

## Files and how sparrow uses them

| File | Version | LSTM layers | Hidden size |
|---|---|---|---|
| `asphericity_network_v1.pt` | 1 | 2 | 45 |
| `asphericity_network_v2.pt` | 2 (default) | 2 | 55 |

Each file holds the weights of a bidirectional LSTM network (BRNN) trained with PARROT 1 on one-hot encoded sequences. It is many-to-one, with one output per sequence, and a regression network.

`AsphericityPredictor`, in `sparrow/predictors/asphericity/asphericity_predictor.py`, loads it, and sparrow registers it as the `"asphericity"` network in `sparrow.predictors.NETWORK_PREDICTORS`. The predictor's `_from_raw` makes the network's prediction from its output, the same way for one sequence or a batch, and sparrow runs it through:

* `Protein(seq).predictor.asphericity()`, which returns the prediction or a value made from it, and for many sequences `sparrow.batch(sequences, ["predictor.asphericity"])`, which runs the network in batches;
* `batch_predict(sequences, network="asphericity")`, from `sparrow.predictors.batch_predictor`, which returns the prediction for many sequences, in batches, and takes `version=`;
* `AsphericityPredictor(version=...).predict(seq)` and `.predict_many(sequences)`, to use a version other than the default directly.

The [networks README](../README.md) describes how the versioned files are named and loaded.
