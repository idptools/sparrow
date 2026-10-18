# Scaling exponent (`networks/scaling_exponent`)

The network that predicts a sequence's scaling exponent, nu, in the polymer scaling law R_ij = R0 |i - j|^nu: dimensionless, about 0.33 for a compact chain and 0.59 for a self-avoiding one. It is one of the ALBATROSS networks (Lotthammer et al., *Nat. Methods* 2024), which predict the conformational properties of disordered regions from sequence. Version 2 is the ALBATROSS release of May 21st 2023 and is the default; version 1 is kept so older results can be reproduced.

## Files and how sparrow uses them

| File | Version | LSTM layers | Hidden size |
|---|---|---|---|
| `scaling_exponent_network_v1.pt` | 1 | 4 | 20 |
| `scaling_exponent_network_v1.5.pt` | 1.5 | 1 | 30 |
| `scaling_exponent_network_v2.pt` | 2 (default) | 2 | 35 |

Each file holds the weights of a bidirectional LSTM network (BRNN) trained with PARROT 1 on one-hot encoded sequences. It is many-to-one, with one output per sequence, and a regression network.

`ScalingExponentPredictor`, in `sparrow/predictors/scaling_exponent/scaling_exponent_predictor.py`, loads it, and sparrow registers it as the `"scaling_exponent"` network in `sparrow.predictors.NETWORK_PREDICTORS`. The predictor's `_from_raw` makes the network's prediction from its output, the same way for one sequence or a batch, and sparrow runs it through:

* `Protein(seq).predictor.scaling_exponent()`, which returns the prediction or a value made from it, and for many sequences `sparrow.batch(sequences, ["predictor.scaling_exponent"])`, which runs the network in batches;
* `batch_predict(sequences, network="scaling_exponent")`, from `sparrow.predictors.batch_predictor`, which returns the prediction for many sequences, in batches, and takes `version=`;
* `ScalingExponentPredictor(version=...).predict(seq)` and `.predict_many(sequences)`, to use a version other than the default directly.

## Notes

Version 1.5 was never assessed or validated and should not be used.

The [networks README](../README.md) describes how the versioned files are named and loaded.
