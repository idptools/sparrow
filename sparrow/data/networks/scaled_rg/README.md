# Scaled radius of gyration (`networks/scaled_rg`)

The network that predicts a sequence's radius of gyration divided by the square root of its length, Rg/sqrt(N). Its predictor multiplies the prediction by sqrt(N), so sparrow returns the radius of gyration itself, in Angstroms. It is one of the ALBATROSS networks (Lotthammer et al., *Nat. Methods* 2024), which predict the conformational properties of disordered regions from sequence. Version 2 is the ALBATROSS release of May 21st 2023 and is the default; version 1 is kept so older results can be reproduced.

## Files and how sparrow uses them

| File | Version | LSTM layers | Hidden size |
|---|---|---|---|
| `scaled_rg_network_v1.pt` | 1 | 1 | 75 |
| `scaled_rg_network_v2.pt` | 2 (default) | 1 | 55 |

Each file holds the weights of a bidirectional LSTM network (BRNN) trained with PARROT 1 on one-hot encoded sequences. It is many-to-one, with one output per sequence, and a regression network.

`ScaledRgPredictor`, in `sparrow/predictors/scaled_rg/scaled_radius_of_gyration_predictor.py`, loads it, and sparrow registers it as the `"scaled_rg"` network in `sparrow.predictors.NETWORK_PREDICTORS`. The predictor's `_from_raw` makes the network's prediction from its output, the same way for one sequence or a batch, and sparrow runs it through:

* `Protein(seq).predictor.radius_of_gyration()`, which returns the prediction or a value made from it, and for many sequences `sparrow.batch(sequences, ["predictor.radius_of_gyration"])`, which runs the network in batches;
* `batch_predict(sequences, network="scaled_rg")`, from `sparrow.predictors.batch_predictor`, which returns the prediction for many sequences, in batches, and takes `version=`;
* `ScaledRgPredictor(version=...).predict(seq)` and `.predict_many(sequences)`, to use a version other than the default directly.

## Notes

This is the network `radius_of_gyration()` uses by default (`use_scaled=True`), and for every sequence shorter than 35 residues whatever `use_scaled` says, because it is much more accurate than the `rg` network for short sequences.

The [networks README](../README.md) describes how the versioned files are named and loaded.
