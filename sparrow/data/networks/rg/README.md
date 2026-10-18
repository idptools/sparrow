# Radius of gyration (`networks/rg`)

The network that predicts a sequence's radius of gyration, in Angstroms, trained on the radius of gyration itself. It is one of the ALBATROSS networks (Lotthammer et al., *Nat. Methods* 2024), which predict the conformational properties of disordered regions from sequence. Version 2 is the ALBATROSS release of May 21st 2023 and is the default; version 1 is kept so older results can be reproduced.

## Files and how sparrow uses them

| File | Version | LSTM layers | Hidden size |
|---|---|---|---|
| `rg_network_v1.pt` | 1 | 1 | 55 |
| `rg_network_v2.pt` | 2 (default) | 1 | 45 |

Each file holds the weights of a bidirectional LSTM network (BRNN) trained with PARROT 1 on one-hot encoded sequences. It is many-to-one, with one output per sequence, and a regression network.

`RgPredictor`, in `sparrow/predictors/rg/radius_of_gyration_predictor.py`, loads it, and sparrow registers it as the `"rg"` network in `sparrow.predictors.NETWORK_PREDICTORS`. The predictor's `_from_raw` makes the network's prediction from its output, the same way for one sequence or a batch, and sparrow runs it through:

* `Protein(seq).predictor.radius_of_gyration(use_scaled=False)`, which returns the prediction or a value made from it, and for many sequences `sparrow.batch(sequences, {"predictor.radius_of_gyration": {"use_scaled": False}})`, which runs the network in batches;
* `batch_predict(sequences, network="rg")`, from `sparrow.predictors.batch_predictor`, which returns the prediction for many sequences, in batches, and takes `version=`;
* `RgPredictor(version=...).predict(seq)` and `.predict_many(sequences)`, to use a version other than the default directly.

## Notes

`radius_of_gyration()` uses the `scaled_rg` network by default, which is more accurate; it uses this one with `use_scaled=False`, and even then only for sequences of at least 35 residues (`sparrow.data.configs.MIN_LENGTH_ALBATROSS_RE_RG`), since it is much less accurate for shorter ones. `safe=False` uses it for every length, which we do not recommend. `batch_predict(network="rg")` applies the same rule unless given `safe=False`.

The [networks README](../README.md) describes how the versioned files are named and loaded.
