# End-to-end distance (`networks/re`)

The network that predicts a sequence's end-to-end distance, in Angstroms, trained on the end-to-end distance itself. It is one of the ALBATROSS networks (Lotthammer et al., *Nat. Methods* 2024), which predict the conformational properties of disordered regions from sequence. Version 2 is the ALBATROSS release of May 21st 2023 and is the default; version 1 is kept so older results can be reproduced.

## Files and how sparrow uses them

| File | Version | LSTM layers | Hidden size |
|---|---|---|---|
| `re_network_v1.pt` | 1 | 1 | 45 |
| `re_network_v2.pt` | 2 (default) | 1 | 45 |

Each file holds the weights of a bidirectional LSTM network (BRNN) trained with PARROT 1 on one-hot encoded sequences. It is many-to-one, with one output per sequence, and a regression network.

`RePredictor`, in `sparrow/predictors/e2e/end_to_end_distance_predictor.py`, loads it, and sparrow registers it as the `"re"` network in `sparrow.predictors.NETWORK_PREDICTORS`. The predictor's `_from_raw` makes the network's prediction from its output, the same way for one sequence or a batch, and sparrow runs it through:

* `Protein(seq).predictor.end_to_end_distance(use_scaled=False)`, which returns the prediction or a value made from it, and for many sequences `sparrow.batch(sequences, {"predictor.end_to_end_distance": {"use_scaled": False}})`, which runs the network in batches;
* `batch_predict(sequences, network="re")`, from `sparrow.predictors.batch_predictor`, which returns the prediction for many sequences, in batches, and takes `version=`;
* `RePredictor(version=...).predict(seq)` and `.predict_many(sequences)`, to use a version other than the default directly.

## Notes

`end_to_end_distance()` uses the `scaled_re` network by default, which is more accurate; it uses this one with `use_scaled=False`, and even then only for sequences of at least 35 residues (`sparrow.data.configs.MIN_LENGTH_ALBATROSS_RE_RG`), since it is much less accurate for shorter ones. `safe=False` uses it for every length, which we do not recommend. `batch_predict(network="re")` applies the same rule unless given `safe=False`.

The network is called `re`, but its predictor module is `e2e`: a module called `re` would clash with Python's regular-expression module, `re`.

The [networks README](../README.md) describes how the versioned files are named and loaded.
