# PScore (`networks/pscore`)

The network that predicts, for each residue, its PScore, the phase-separation propensity score of Vernon et al. (*eLife* 2018), on PScore's native 0-6 scale.

## Files and how sparrow uses them

| File | Version | LSTM layers | Hidden size |
|---|---|---|---|
| `pscore_predictor_network_v2.pt` | 2 | 2 | 10 |
| `pscore_predictor_network_v3.pt` | 3 | 2 | 10 |
| `pscore_predictor_network_v4.pt` | 4 (default) | 2 | 10 |

Each file holds the weights of a bidirectional LSTM network (BRNN) trained with PARROT 1 on one-hot encoded sequences. It is many-to-many, with outputs at every residue, and a regression network.

`PScorePredictor`, in `sparrow/predictors/pscore/pscore_predictor.py`, loads it, and sparrow registers it as the `"pscore"` network in `sparrow.predictors.NETWORK_PREDICTORS`. The predictor's `_from_raw` makes the network's prediction from its output, the same way for one sequence or a batch, and sparrow runs it through:

* `Protein(seq).predictor.pscore()`, which returns the prediction or a value made from it, and for many sequences `sparrow.batch(sequences, ["predictor.pscore"])`, which runs the network in batches;
* `batch_predict(sequences, network="pscore")`, from `sparrow.predictors.batch_predictor`, which returns the prediction for many sequences, in batches, and takes `version=`;
* `PScorePredictor(version=...).predict(seq)` and `.predict_many(sequences)`, to use a version other than the default directly.

## Notes

`Protein(seq).predictor.pscore()` rescales the prediction onto 0-1 by default, dividing it by 6 and clipping it to that range; `pscore(normalized=False)` returns the native scale. `batch_predict` returns the native scale.

The [networks README](../README.md) describes how the versioned files are named and loaded.
