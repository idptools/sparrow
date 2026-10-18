# Phosphorylation (`networks/phosphorylation`)

Three networks that predict, for each residue, the probability that it is phosphorylated, rounded to 5 decimal places: one for serine, one for threonine and one for tyrosine.

## `ser_phosphorylation`

| File | Version | LSTM layers | Hidden size |
|---|---|---|---|
| `ser_phosphorylation_predictor_network_v1.pt` | 1 (default) | 2 | 10 |

Each file holds the weights of a bidirectional LSTM network (BRNN) trained with PARROT 1 on one-hot encoded sequences. It is many-to-many, with outputs at every residue, and a classifier over 2 classes.

`SerPhosphorylationPredictor`, in `sparrow/predictors/phosphorylation/ser_phosphorylation_predictor.py`, loads it, and sparrow registers it as the `"ser_phosphorylation"` network in `sparrow.predictors.NETWORK_PREDICTORS`. The predictor's `_from_raw` makes the network's prediction from its output, the same way for one sequence or a batch, and sparrow runs it through:

* `Protein(seq).predictor.serine_phosphorylation()`, which returns the prediction or a value made from it, and for many sequences `sparrow.batch(sequences, ["predictor.serine_phosphorylation"])`, which runs the network in batches;
* `batch_predict(sequences, network="ser_phosphorylation")`, from `sparrow.predictors.batch_predictor`, which returns the prediction for many sequences, in batches, and takes `version=`;
* `SerPhosphorylationPredictor(version=...).predict(seq)` and `.predict_many(sequences)`, to use a version other than the default directly.

## `thr_phosphorylation`

| File | Version | LSTM layers | Hidden size |
|---|---|---|---|
| `thr_phosphorylation_predictor_network_v1.pt` | 1 (default) | 2 | 10 |

Each file holds the weights of a bidirectional LSTM network (BRNN) trained with PARROT 1 on one-hot encoded sequences. It is many-to-many, with outputs at every residue, and a classifier over 2 classes.

`ThrPhosphorylationPredictor`, in `sparrow/predictors/phosphorylation/thr_phosphorylation_predictor.py`, loads it, and sparrow registers it as the `"thr_phosphorylation"` network in `sparrow.predictors.NETWORK_PREDICTORS`. The predictor's `_from_raw` makes the network's prediction from its output, the same way for one sequence or a batch, and sparrow runs it through:

* `Protein(seq).predictor.threonine_phosphorylation()`, which returns the prediction or a value made from it, and for many sequences `sparrow.batch(sequences, ["predictor.threonine_phosphorylation"])`, which runs the network in batches;
* `batch_predict(sequences, network="thr_phosphorylation")`, from `sparrow.predictors.batch_predictor`, which returns the prediction for many sequences, in batches, and takes `version=`;
* `ThrPhosphorylationPredictor(version=...).predict(seq)` and `.predict_many(sequences)`, to use a version other than the default directly.

## `tyr_phosphorylation`

| File | Version | LSTM layers | Hidden size |
|---|---|---|---|
| `tyr_phosphorylation_predictor_network_v1.pt` | 1 (default) | 2 | 10 |

Each file holds the weights of a bidirectional LSTM network (BRNN) trained with PARROT 1 on one-hot encoded sequences. It is many-to-many, with outputs at every residue, and a classifier over 2 classes.

`TyrPhosphorylationPredictor`, in `sparrow/predictors/phosphorylation/tyr_phosphorylation_predictor.py`, loads it, and sparrow registers it as the `"tyr_phosphorylation"` network in `sparrow.predictors.NETWORK_PREDICTORS`. The predictor's `_from_raw` makes the network's prediction from its output, the same way for one sequence or a batch, and sparrow runs it through:

* `Protein(seq).predictor.tyrosine_phosphorylation()`, which returns the prediction or a value made from it, and for many sequences `sparrow.batch(sequences, ["predictor.tyrosine_phosphorylation"])`, which runs the network in batches;
* `batch_predict(sequences, network="tyr_phosphorylation")`, from `sparrow.predictors.batch_predictor`, which returns the prediction for many sequences, in batches, and takes `version=`;
* `TyrPhosphorylationPredictor(version=...).predict(seq)` and `.predict_many(sequences)`, to use a version other than the default directly.

## Notes

By default the three methods return a mask, 1 at each predicted site and 0 elsewhere: a window of 4 residues either side slides along the sequence, and wherever the residue at its centre has a probability above 0.6, every residue of the target type in the window is called a site. `raw_values=True` returns the probabilities instead, and `return_sites_only=True` the positions of the sites. Each method reads its network's cached prediction, so whichever of these it is asked for, the network runs once per sequence.

The [networks README](../README.md) describes how the versioned files are named and loaded.
