# pLDDT, version 2 (`networks/plddt2`)

The network that predicts, for each residue, the pLDDT AlphaFold2 would give it: AlphaFold2's own estimate of how accurately it has placed that residue, from 0 to 100. It was trained with PARROT 2 on the per-residue pLDDT of 1.86 million AlphaFold2 structures. `Protein(seq).predictor.pLDDT()` gives metapredict's prediction of the same thing; this is sparrow's own.

## Files and how sparrow uses them

| File | Version | Architecture | LSTM layers | Hidden size | Parameters |
|---|---|---|---|---|---|
| `plddt2_network_v1.pt` | 1 (default) | BRNN, dropout 0.1 | 2 | 64 | 143,489 |

The file is a PARROT 2 checkpoint, which records its own architecture (a bidirectional LSTM with one output per residue), its task (regression) and its encoding (one-hot), so loading it needs PARROT 2 installed; without it, sparrow raises an error saying how to install it. The network predicts pLDDT / 100, the scale it was trained on.

`PLDDT2Predictor`, in `sparrow/predictors/plddt2/plddt2_predictor.py`, loads it, and sparrow registers it as the `"plddt2"` network in `sparrow.predictors.NETWORK_PREDICTORS`. The predictor's `_from_raw` makes the network's prediction from its output, the same way for one sequence or a batch, and sparrow runs it through:

* `Protein(seq).predictor.pLDDT2()`, which returns the prediction, and for many sequences `sparrow.batch(sequences, ["predictor.pLDDT2"])`, which runs the network in batches;
* `batch_predict(sequences, network="plddt2")`, from `sparrow.predictors.batch_predictor`, which returns the prediction for many sequences, in batches, and takes `version=`;
* `PLDDT2Predictor(version=...).predict(seq)` and `.predict_many(sequences)`, to use a version other than the default directly.

## Notes

The prediction is the network's output times 100, so it is on AlphaFold's 0-100 scale, as `pLDDT()` is, clipped to 0-100. The network is a regression, so a few residues are predicted just outside that range: 0.06% of the test set's residues, all above 100.

How version 1 was trained:

* **Data.** Per-residue pLDDT read from the AlphaFold2 models of 1,856,713 training proteins, downloaded from the AlphaFold Protein Structure Database (AFDB) in October 2026: 1,833,171 AFDB version 6 models (Swiss-Prot, the model-organism and global-health proteomes, and about a million TrEMBL entries sampled across AFDB's Foldseek clusters) and 23,542 from the VADB viral collection AFDB distributes. The viral collections predicted with ColabFold (BFVD, Viro3D and NVDA) were left out, because ColabFold's pLDDT is not on quite the same footing as AlphaFold2's; leaving them out lowered the validation error on AlphaFold2 structures.
* **Sets.** Proteins were assigned to training, validation and test sets (90/5/5) as whole connected components of their MMseqs2 30%-identity clusters and Foldseek structural clusters, so no validation or test protein shares either kind of cluster with a training protein. The validation set (99,574 proteins) chose the network; the test set (100,081 proteins) was used only to report its accuracy.
* **Training.** One-hot sequences; mean squared error of pLDDT / 100, averaged over each protein's residues and then over proteins, so long proteins do not dominate; Adam at a learning rate of 3e-4 falling along a cosine to 0 over 20 epochs, gradients clipped to a global norm of 0.25, at most 32 sequences and 16,384 padded residues per batch, seed 1. The network kept is the one from epoch 14, whose validation loss was the lowest. Training took 10 h 23 min on an NVIDIA RTX A4000.
* **Accuracy.** On the 100,081 test proteins (26.2 million residues), the per-residue RMSE was 15.8 pLDDT and the Pearson correlation 0.76, against 20.2 and 0.58 for metapredict's pLDDT network; the median correlation within a protein was 0.72 (metapredict 0.45). On the test proteins with no Swiss-Prot entry in their split component, which metapredict's network (trained on Swiss-Prot) is least likely to have seen relatives of, the error per protein was lower than metapredict's for 81% of proteins.
* **Speed.** The whole human proteome (20,393 sequences) takes about 3.4 s on an NVIDIA RTX A4000 (metapredict 3.7 s) and about 2.5 min on one CPU thread (metapredict 46 s).

The [networks README](../README.md) describes how the versioned files are named and loaded.
