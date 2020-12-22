sparrow
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/sparrow/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/sparrow/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/sparrow/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/sparrow/branch/master)


# SPARROW: Sequence PARameters of Regions in disOrdered Windows 

Next generation package for sequence parameter calculation. 

## Installation
For now (i.e. during development) installation should be done by 

1. Creating a local directory where you want the code to live (i.e. somewhere in your directory structure that makes sense, such as 

		/Users/alex/Documents/software/sparrow

	or
	
		/Users/alex/Dropbox/software/sparrow
		
	It doesn't actually matter where this is but having it somewhere sensible will be useful as this code is continually updated and you can use `git pull` to update as and when new features/fixes are added.
	
2. Clone this git repository to your local machine and move into that directory

		git clone git@github.com:holehouse-lab/sparrow.git
		cd sparrow

3. Install sparrow using `pip`

		pip install .

## Usage
`sparrow` gives you a window into protein sequence properties in a lightweight, just-in-time way. Standard pattern is

	from sparrow import Protein
	
	my_cool_protein = Protein('THISISAMINAACIDSEQWENCE')
	
The object `my_cool_protein` is now available and has a large collection of attributes and functions that can be accessed via standard dot notation, e.g.

	print(my_cool_protein.FCR)
	print(my_cool_protein.NCPR)
	print(my_cool_protein.hydrophobicity)

Much more extensive documentation is coming and for now, to see the functions take a look at the [code directly here](https://github.com/holehouse-lab/sparrow/blob/main/sparrow/protein.py)


### Copyright

Copyright (c) 2020, Alex Holehouse & Ryan Emenecker, built in the Holehouse lab.


#### Acknowledgements
 
