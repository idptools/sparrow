# Secondary structure (DSSP) (`networks/dssp`)

The network that predicts, for each residue, the probability that it is helical, extended or coil, as DSSP would assign it. Stephen Plassmeyer trained it on DSSP assignments of AlphaFold2 structures.

## Files and how sparrow uses them

| File | Version | LSTM layers | Hidden size |
|---|---|---|---|
| `dssp_predictor_network_v1.pt` | 1 | 2 | 20 |
| `dssp_predictor_network_v2.pt` | 2 (default) | 2 | 20 |

Each file holds the weights of a bidirectional LSTM network (BRNN) trained with PARROT 1 on one-hot encoded sequences. It is many-to-many, with outputs at every residue, and a classifier over 3 classes.

`DSSPPredictor`, in `sparrow/predictors/dssp/dssp_predictor.py`, loads it, and sparrow registers it as the `"dssp"` network in `sparrow.predictors.NETWORK_PREDICTORS`. The predictor's `_from_raw` makes the network's prediction from its output, the same way for one sequence or a batch, and sparrow runs it through:

* `Protein(seq).predictor.dssp_helicity()`, which returns the prediction or a value made from it, and for many sequences `sparrow.batch(sequences, ["predictor.dssp_helicity"])`, which runs the network in batches;
* `Protein(seq).predictor.dssp_extended()`, which returns the prediction or a value made from it, and for many sequences `sparrow.batch(sequences, ["predictor.dssp_extended"])`, which runs the network in batches;
* `Protein(seq).predictor.dssp_coil()`, which returns the prediction or a value made from it, and for many sequences `sparrow.batch(sequences, ["predictor.dssp_coil"])`, which runs the network in batches;
* `batch_predict(sequences, network="dssp")`, from `sparrow.predictors.batch_predictor`, which returns the prediction for many sequences, in batches, and takes `version=`;
* `DSSPPredictor(version=...).predict(seq)` and `.predict_many(sequences)`, to use a version other than the default directly.

## Notes

The network gives three outputs per residue, which its predictor turns into probabilities with a softmax, in the order helix, extended, coil.

`dssp_helicity`, `dssp_extended` and `dssp_coil` each read one column: with `mode='probability'` they return it, and by default (`mode='class'`) they classify each residue against `threshold` (0.5), keeping regions at least the minimum length (5 residues for helix and extended, 1 for coil). The prediction is cached, so the three methods, with any arguments, run the network once per sequence.

The [networks README](../README.md) describes how the versioned files are named and loaded.
