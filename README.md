sparrow
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/sparrow/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/sparrow/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/sparrow/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/sparrow/branch/master)


# SPARROW: Sequence PARameters of Regions in disOrdered Windows 

Sparrow is our next generation package for sequence parameter calculation. It's still in development, but, the functionality present works well.

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
We'd like to put an initial public version of sparrow out in the late fall of 2021. I'd also love to include as many people from the lab on this as possible, so, if you have ideas for analysis please talk to me and we can set up how to build them in (which will probably involve Alex writing a stub] first and then you adding code).

The plan if for sparrow to contain 

1. 'native' sequence analysis tools (where properties, either average or linear) are computed directly from the sequence. Many example of sequence that follow this pattern are currently implemented

2. Deep-learning based sequence analysis, whereby PARROT-trained networks are read in and used to analyze sequence. We haven't yet implemented any of these, but  

### How to add sequence parameter

### Copyright

Copyright (c) 2020-2021, Alex Holehouse & Ryan Emenecker, built in the Holehouse lab. 


#### Acknowledgements
 
