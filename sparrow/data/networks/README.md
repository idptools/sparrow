# Networks (`networks`)

The trained networks behind sparrow's predictors, one directory per predictor. Each directory holds every version of its network's weights, as `<file stem>_network_v<version>.pt`, and a README describing them. The disorder and pLDDT predictions are not here: they come from metapredict, which ships its own networks.

## The networks

| Directory | Network | Predicts | `Protein.predictor` method | Default version |
|---|---|---|---|---|
| [`asphericity`](asphericity/README.md) | `asphericity` | Asphericity (ALBATROSS) | `asphericity()` | 2 |
| [`dssp`](dssp/README.md) | `dssp` | Helix, extended and coil probability per residue | `dssp_helicity()`, `dssp_extended()`, `dssp_coil()` | 2 |
| [`mitochondrial_targeting`](mitochondrial_targeting/README.md) | `mitochondrial_targeting` | Mitochondrial targeting sequence per residue | `mitochondrial_targeting_sequence()` | 1 |
| [`nuclear_export_signal`](nuclear_export_signal/README.md) | `nes` | Nuclear export signal probability per residue | `nuclear_export_signal()` | 1 |
| [`nuclear_import_signal`](nuclear_import_signal/README.md) | `nls` | Nuclear import signal probability per residue | `nuclear_import_signal()` | 1 |
| [`phosphorylation`](phosphorylation/README.md) | `ser_phosphorylation` | Serine phosphorylation probability per residue | `serine_phosphorylation()` | 1 |
| [`phosphorylation`](phosphorylation/README.md) | `thr_phosphorylation` | Threonine phosphorylation probability per residue | `threonine_phosphorylation()` | 1 |
| [`phosphorylation`](phosphorylation/README.md) | `tyr_phosphorylation` | Tyrosine phosphorylation probability per residue | `tyrosine_phosphorylation()` | 1 |
| [`prefactor`](prefactor/README.md) | `prefactor` | Polymer scaling prefactor, R0 (ALBATROSS) | `prefactor()` | 2 |
| [`pscore`](pscore/README.md) | `pscore` | PScore phase-separation propensity per residue | `pscore()` | 4 |
| [`re`](re/README.md) | `re` | End-to-end distance (ALBATROSS) | `end_to_end_distance()` | 2 |
| [`rg`](rg/README.md) | `rg` | Radius of gyration (ALBATROSS) | `radius_of_gyration()` | 2 |
| [`scaled_re`](scaled_re/README.md) | `scaled_re` | End-to-end distance, trained on Re/sqrt(N) (ALBATROSS) | `end_to_end_distance()` | 2 |
| [`scaled_rg`](scaled_rg/README.md) | `scaled_rg` | Radius of gyration, trained on Rg/sqrt(N) (ALBATROSS) | `radius_of_gyration()` | 2 |
| [`scaling_exponent`](scaling_exponent/README.md) | `scaling_exponent` | Polymer scaling exponent, nu (ALBATROSS) | `scaling_exponent()` | 2 |
| [`transactivation_domains`](transactivation_domains/README.md) | `tad` | Transactivation domain probability per residue | `transactivation_domains()` | 1 |
| [`transmembrane`](transmembrane/README.md) | `transmembrane` | Transmembrane region per residue | `transmembrane_regions()` | 4 |

`radius_of_gyration()` and `end_to_end_distance()` read the scaled networks by default, and the `rg` and `re` networks only with `use_scaled=False` (see their READMEs).

## How sparrow loads them

`sparrow.predictors.NETWORK_PREDICTORS` names every network, and maps the name to the predictor class that loads it. Each predictor class (a subclass of `sparrow.predictors.base.BaseNetworkPredictor`, in a package under `sparrow/predictors`) declares where its weights are, as a `NETWORK_PATH` template relative to `sparrow/data` such as `networks/rg/rg_network_v{version}.pt`, and which version it loads by default, `DEFAULT_VERSION`. `sparrow.predictors.network_loader.load_parrot_network` reads the file. A predictor's module is imported, and its weights read, only when the network is first used.

Every network here was trained with PARROT 1, which saves only a bidirectional LSTM's weights, so its predictor also declares its `ARCHITECTURE`: `"MtO"` (many-to-one) for one value per sequence, or `"MtM"` (many-to-many) for one per residue. A network trained with PARROT 2 records its own architecture, task and encoding, and sparrow loads it with PARROT 2's `Predictor`; both kinds are used the same way.

Older versions stay here so earlier results can be reproduced. To load one, give the version: `batch_predict(sequences, network="rg", version=1)`, or `RgPredictor(version="1")`. To add a network, see the developer guide on predictors (`docs/predictors.rst`).
