sparrow
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/sparrow/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/sparrow/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/sparrow/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/sparrow/branch/master)

# SPARROW


## Overview
SPARROW (Sequence PARameters for RegiOns in Windows) is our next generation package for calculating and predicting amino acid sequence features. It is meant as a light-weight object-oriented framework for working with protein sequences that integrates both direct sequence calculations and per-residue predictions from deep learning models trained using [PARROT](https://elifesciences.org/articles/70576). 

Our goal is for SPARROW to be easy for anyone to add new sequence analysis plugins into, creating a simple, extendable framework for de novo sequence feature characterization.

SPARROW is in *active* development. **We do not recommend using SPARROW at this time**. If you need to use it in your code or analysis PLEASE reach out to us and let us know, because things are changing rapidly.

## Use and distribution
We plan to finish up SPARROW and publish sometime in fall of 2022. Until that time the code is provided under the Creative Commons Attribution NonCommercial NoDerivs (CC-NC-ND) license (see LICSENE). This is – deliberately – an extremely restrictive license which we are using ONLY until the code is actually released (at which point we'll transition to a GNU-style license). 

The reason for this is to try and discourage you from incorporating code that may change dramatic over the next few months


## Installation
Installation can be done via `pip` directly from GitHub!!!

	pip install git+ssh://git@github.com/holehouse-lab/sparrow.git
	
NOTE that for this to work you must [have set up ssh keys](https://gitlab.com/holehouselab/labbook/-/blob/master/guides/ssh/setting_up_ssh_keys.md) and have [connected your ssh key to GitHub](https://gitlab.com/holehouselab/labbook/-/blob/master/guides/ssh/ssh_keys_for_github.md). This is because **sparrow** is a *private* repository, so installation is limited to those authenticated against the [Holehouse lab GitHub page](https://github.com/holehouse-lab/).

This can be re-run to update when updates are pushed. What a world we live in!

## Usage

#### General
`sparrow` gives you a window into protein sequence properties in a lightweight, just-in-time way. Standard pattern for analysis is:

	from sparrow import Protein
	
	my_cool_protein = Protein('THISISAMINAACIDSEQWENCE')
	
The object `my_cool_protein` is now available and has a large collection of attributes and functions that can be accessed via standard dot notation, e.g.

	print(my_cool_protein.FCR)
	print(my_cool_protein.NCPR)
	print(my_cool_protein.hydrophobicity)

Much more extensive documentation is coming and for now, to see the functions take a look at the [code directly here](https://github.com/holehouse-lab/sparrow/blob/main/sparrow/protein.py)

In general sparrow is written in a Protein-centric way - i.e. all functions emerge from the Protein object.

#### Reading in FASTA files
One non-obvious thing is if you have a FASTA file you can read it in to a dictionary of `sparrow.Protein` objects using:

	from sparrow import read_fasta
	protein_dictionary = read_fasta('my_fasta_file.fasta')
	
protein_dictionary is then a dictionary with key/value pairs for the Protein objects. `read_fasta()` accepts the same keyword arguments as [protfasta.read_fasta](https://protfasta.readthedocs.io/en/latest/read_fasta.html) so check that out.

### Roadmap
An initial public version of SPARROW is released in June 2022 to enable existing tools developed by the Holehouse lab to use this code. This version is not meant for those outside the Holehouse lab to take and use (see **Use and distribution** for their own safety!).

A full public release is planned for Fall of 2022.


### Copyright

Copyright (c) 2020-2022, Alex Holehouse & Ryan Emenecker, built in the Holehouse lab. Currently shared under a CC BY-NC-ND license. 


#### Acknowledgements
 
