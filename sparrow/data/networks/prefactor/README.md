# Prefactor (`networks/prefactor`)

The network that predicts a sequence's prefactor, R0, in the polymer scaling law R_ij = R0 |i - j|^nu, which describes how the distance between residues i and j grows with their separation; with distances in Angstroms, R0 is in Angstroms too. It is one of the ALBATROSS networks (Lotthammer et al., *Nat. Methods* 2024), which predict the conformational properties of disordered regions from sequence. Version 2 is the ALBATROSS release of May 21st 2023 and is the default; version 1 is kept so older results can be reproduced.

## Files and how sparrow uses them

| File | Version | LSTM layers | Hidden size |
|---|---|---|---|
| `prefactor_network_v1.pt` | 1 | 2 | 15 |
| `prefactor_network_v2.pt` | 2 (default) | 1 | 70 |

Each file holds the weights of a bidirectional LSTM network (BRNN) trained with PARROT 1 on one-hot encoded sequences. It is many-to-one, with one output per sequence, and a regression network.

`PrefactorPredictor`, in `sparrow/predictors/prefactor/prefactor_predictor.py`, loads it, and sparrow registers it as the `"prefactor"` network in `sparrow.predictors.NETWORK_PREDICTORS`. The predictor's `_from_raw` makes the network's prediction from its output, the same way for one sequence or a batch, and sparrow runs it through:

* `Protein(seq).predictor.prefactor()`, which returns the prediction or a value made from it, and for many sequences `sparrow.batch(sequences, ["predictor.prefactor"])`, which runs the network in batches;
* `batch_predict(sequences, network="prefactor")`, from `sparrow.predictors.batch_predictor`, which returns the prediction for many sequences, in batches, and takes `version=`;
* `PrefactorPredictor(version=...).predict(seq)` and `.predict_many(sequences)`, to use a version other than the default directly.

The [networks README](../README.md) describes how the versioned files are named and loaded.
