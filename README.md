# sparrow: a tool for integrative analysis and prediction from protein sequence data 


### Major version 0.2.3


## Overview
SPARROW (Sequence PARameters for RegiOns in Windows) is our next-generation package for calculating and predicting amino acid sequence features. It is meant as a lightweight object-oriented framework for working with protein sequences that integrates both direct sequence calculations and per-residue predictions from deep learning models trained using [PARROT](https://elifesciences.org/articles/70576). 

Our goal is for SPARROW to be easy for anyone to add new sequence analysis plugins into, creating a simple, extendable framework for de novo sequence feature characterization.

SPARROW is in *active* development. As of version 0.2 a few things are considered mature - notably the [ALBATROSS implementation](https://www.biorxiv.org/content/10.1101/2023.05.08.539824v2) in SPARROW is robust and is good to be used, as are many of the bulk sequence property calculators. 

In general, we recommend using [localCIDER](http://pappulab.github.io/localCIDER/) for general sequence analysis, but the ALBATROSS predictors in SPARROW are ready for public use.

## Use and distribution
SPARROW is still under active development until the first major release; however, the core API has mostly been decided upon. 

## Installation
Installation can be done via `pip` directly from GitHub!!!

	pip install git+https://git@github.com/idptools/sparrow.git
	
## Usage

#### General
`sparrow` gives you a window into protein sequence properties in a lightweight, just-in-time way. Standard pattern for analysis is:

	from sparrow import Protein
	
	my_cool_protein = Protein('THISISAMINAACIDSEQWENCE')
	
The object `my_cool_protein` is now available and has a large collection of attributes and functions that can be accessed via standard dot notation, e.g.

	print(my_cool_protein.FCR)
	print(my_cool_protein.NCPR)
	print(my_cool_protein.hydrophobicity)

Much more extensive documentation is coming, and for now, to see the functions take a look at the [code directly here](https://github.com/holehouse-lab/sparrow/blob/main/sparrow/protein.py)

In general, SPARROW is written in a Protein-centric way - i.e., all functions emerge from the Protein object.

#### Reading in FASTA files
One non-obvious thing is if you have a FASTA file you can read it into a dictionary of `sparrow.Protein` objects using:

	from sparrow import read_fasta
	protein_dictionary = read_fasta('my_fasta_file.fasta')
	
protein_dictionary is then a dictionary with key/value pairs for the Protein objects. `read_fasta()` accepts the same keyword arguments as [protfasta.read_fasta](https://protfasta.readthedocs.io/en/latest/read_fasta.html) so check that out.

## ALBATROSS in SPARROW
To use ALBATROSS functionality in SPARROW there are two routes one can take - single protein prediction or batch prediction.


#### Single protein predictions
For single protein prediction, one can predict specific features using the following syntax:



	from sparrow import Protein
	
	P = Protein('MKYLAAYLLLNAAGNTPDATKIKAILESVGIEIEDEKVSSVLSALEGKSVDELITEGNEKLAAVPAAGPASAGGAAAASGDAAAEEEKEEEAAEESDDDMGFGLFD')
	
	print(P.predictor.asphericity())
		
	print(P.predictor.radius_of_gyration())
	print(P.predictor.radius_of_gyration(use_scaled=True))
	
	print(P.predictor.end_to_end_distance(use_scaled=True))
	print(P.predictor.end_to_end_distance(use_scaled=False))
	
	print(P.predictor.scaling_exponent())
	print(P.predictor.prefactor())
	
Note that Rg and Re can be calculated using the `use_scaled` flag, which if used, means we calculate on a network trained on Rg/Sqrt(N) and Re/Sqrt(N) data. In general, we always recommend using use_scaled (which is the default behavior); it provides much better accuracy at shorter sequences, and is the mode used in the main-text ALBATROSS figures. 

#### Batch predictions
ALBATROSS also affords a batch mode which on GPUs enables 1000s of sequences to be predicted in a few seconds.

Batch prediction can be obtained via 

	from sparrow.predictors import batch_predict
	
	P = Protein('MKYLAAYLLLNAAGNTPDATKIKAILESVGIEIEDEKVSSVLSALEGKSVDELITEGNEKLAAVPAAGPASAGGAAAASGDAAAEEEKEEEAAEESDDDMGFGLFD')
	
	# dictionary with one sequence, but in general, you'd probably
	# want to pass in many...xs
	input_seqs = {1:P}
	
	# run batch prediction
	return_dict = batch_predict.batch_predict({1:P}, network='re')
	
The return dict is a dictionary of sequence-to-value mapping, and you can select one of the standard networks for doing batch prediction:

* `rg`
* `scaled_rg`
* `re`
* `scaled_re`
* `prefactor`
* `scaling_exponent`
* `asphericity`

The benefits from parallelization on both GPUs and CPUs, i.e., proteome-scale analysis, is highly accessible. As with single sequence predictions, **we strongly recommend using the `scaled_rg` and `scaled_re` networks**.


If you use ALBATROSS, please cite:

1. Lotthammer, J. M.; Ginell, G. M.; Griffith, D.; Emenecker, R. J.; Holehouse, A. S. Direct Prediction of Intrinsically Disordered Protein Conformational Properties from Sequence. Nat. Methods 2024, 1–12.

### Roadmap
An initial public version of SPARROW was released in June 2022 to enable existing tools developed by the Holehouse lab to use this code. This version is not meant for those outside the Holehouse lab to take and use (see **Use and distribution** for their own safety!).

A full public release is planned for spring of 2025.

## Changelog

#### Nover 2024 (version 0.2.3 release)
* Updated to using pyproject.toml for package data
* Fixed tests to work with metapredict V3

#### June 2024 (version 0.2.3 release)
* Cythonized SHD/SCD and IWD clustering sequence parameters

#### Sept 2023 (version 0.2.2 release)
* Updated low complexity domain identification code. 
* Added ability to grey out sequences in `show_sequence()` [h/t Garrett!].

#### May 2023 (version 0.2.1 release)
* Update to ALBATROSS v2 networks (all networks use the `v2` by default both in individual predictors and batch predictions
* Re-wrote much of `batch_predict()` code. Changes here include 
	* Implementation of the `size-collect` algorithm to ensure the passed batchsize does not impact the accuracy of predictions. Batch prediction can now use larger batch sizes, providing better performance on both GPUs and CPUs
	*  Set default batch size to 32
	*  Improved robustness of input types `batch_predict()` can accept. Can now take dictionaries and lists of sparrow.protein.Protein objects or dictionaries/lists of sequences.
	*  Changed order of input parameters for `batch_predict()`, such that now the only required options are [0] Input list/dictionary and [2] name of the network to be used.
	*  Updated return type for `batch_predict()` such that now the return type by default is a dictionary that maps input IDs (or list positions) to sequence and prediction. The original return behavior (a dictionary that maps sequence to prediction) can be obtained if the `return_seq2prediction` flag is set to True.
	*  Wrote much more extensive tests for all `batch_predict()` code
	*  Ensure `batch_predict()` guarantees the same return order as the input order if possible. The only exception is if return_seq2prediction=True and duplicate sequences are found in the input data, in which case only the first occurrence of a given sequence is included. Also order here refers to the order if the return dictionary had dict.values() called.
*  **Added scaled-network for small sequences**: In the course of testing the networks we noticed that in both V1 and V2, when sequences are short (<30-40 amino acids) the non-scaled Re and Rg predictors can return non-sensical results. In contrast, the `scaled_rg` and `scaled_re` networks show reasonable and reproducible polymeric behavior for these smaller sequences. To address this, in both single sequence predictions and batch predict, by default, even if an `rg` or `re` network is requested, if the sequence is less than 35 residues long, we force the `scaled_rg` or `scaled_re` networks. This can be over-ridden by setting the 'safe' keyword in either `batch_predict()` or the single sequence `radius_of_gyration()` or `end_to_end_distance()` 
*  Technical change: the end-to-end distance predictor module found under sparrow/predictors was renamed from `re` to `e2e` to prevent clashing with Python's regular expression (`re`) package. This does not introduce any errors, but makes debugging predictors challenging. The actual network name is retained as `re`.
*  Default predictor for `Protein.predictor.radius_of_gyration()` and `Protein.predictor.end_to_end_distance()` use `use_scaled=True` as a default, based on accuracy of the V2 networks.
	

#### May 2023 (version 0.2 release)
* First major 'alpha' release to coincide with ALBATROSS preprint

#### Feb 2023 
* Building early version of ALBATROSS

#### Nov 2022 (0.
* Major set of updates to fix errors in how kappa is calculated
* Added SCD and SHD functions
* Added some tests
* Moved `scd.py` into `sparrow.patterning`
* Updated IWD for bivariate charge clustering (h/t Garrett Ginell)
* Fixed some tests(h/t Jeff Lotthammer)
* Improved docstrings

#### July 2022
* Moved to idptools! If you had previously cloned sparrow, you can update your git remote location using:

		git remote set-url origin git@github.com:idptools/sparrow.git


* Updated requirement for Python 3.7 or 3.8


### Copyright
Copyright (c) 2020-2024, Alex Holehouse, Ryan Emenecker, Jeff Lotthammer, Garrett Ginell, and Dan Griffith built in the Holehouse lab. Currently shared under a MIT license. 


#### Acknowledgements
 
