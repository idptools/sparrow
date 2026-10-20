Bundled Data
============

sparrow ships the data its calculations need, under ``sparrow/data``. Each
directory there has a README saying what its files hold and how sparrow uses
them.

Residue sets and scales
-----------------------

:mod:`sparrow.data.amino_acids` holds the residue sets the composition
properties count and the per-residue scales they average, each keyed by
one-letter code - the definitions behind the properties on a ``Protein``:

.. code-block:: python

   from sparrow.data.amino_acids import AA_PKA, AA_hydro_KD, ARO, DISORDER_PROMOTING, POLAR

   ARO                    # the aromatic residues, as fraction_aromatic counts them
   AA_hydro_KD["W"]       # Trp on the Kyte-Doolittle scale
   AA_PKA                 # the side-chain pKa values charge_at_pH uses

It also holds the residue masses, the three- and one-letter codes, and the
colors ``show_sequence`` uses. :doc:`../api/data` lists every constant.

AAindex
-------

The AAindex release sparrow ships has three databases: AAindex1, 569 scales
with one value per amino acid, and AAindex2 and AAindex3, 105 substitution
matrices and 50 contact potentials with one value per *pair*. There are two
ways in:

* :mod:`sparrow.data.aaindex` names each AAindex1 scale with a readable
  identifier, such as ``"hydropathy-kyte-1982"``, for
  ``Protein.linear_property_profile``; :doc:`aaindex` covers it and lists every
  scale.
* :mod:`sparrow.data.properties` reads all three databases by accession, and
  is the only way to the pairwise matrices:

.. code-block:: python

   from sparrow.data.properties import available, load

   available()                              # ['aaindex1', 'aaindex2', 'aaindex3']

   scales = load("aaindex1")
   scales.value("KYTJ820101", "W")          # -0.9, Kyte-Doolittle hydropathy of Trp
   scales.list("stickiness")                # [(accession, description), ...]

   matrices = load("aaindex2")
   matrices.pair("EDSSMat62", "W", "W")     # one entry of a substitution matrix
   matrices.matrix("EDSSMat62")             # the whole matrix, as nested dicts

Networks
--------

The trained networks are in ``sparrow/data/networks``, one directory per
predictor, with every version that ships; :doc:`networks` covers choosing one.
Disorder and pLDDT come from metapredict, which ships its own networks.

Reference data
--------------

* ``sparrow/data/elm_classes.tsv`` - the ELM class table (version 1.4, 2 April
  2024, 349 classes) that ``Protein.elms`` matches.
* ``sparrow/data/nardini`` - the NARDINI+ compositional backgrounds of nine
  species and the GIN centroids, as published and as refit, redistributed from
  the GIN resource under the MIT licence (see :doc:`nardini`).

:func:`sparrow.data_path` gives the absolute path of any of these files:

.. code-block:: python

   import sparrow

   sparrow.data_path("elm_classes.tsv")

Settings
--------

:mod:`sparrow.data.configs` holds one setting,
``MIN_LENGTH_ALBATROSS_RE_RG`` (35 residues): the length below which the
radius of gyration and end-to-end distance are predicted with the scaled
networks, whatever was asked for, unless ``safe=False``.

Reference
---------

The data modules are documented in full in :doc:`../api/data`.
