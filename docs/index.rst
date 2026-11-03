sparrow
=======

**sparrow** (*Sequence PARameters for RegiOns in Windows*) is a toolkit for
analyzing and predicting features of protein sequences, with a particular focus
on intrinsically disordered regions (IDRs).

You make a :class:`~sparrow.protein.Protein` from a sequence and read
everything off it - sequence parameters, per-residue profiles, deep-learning
predictions, polymer dimensions and molecular grammars. For many sequences,
:func:`sparrow.batch` makes a ``Protein`` for each, with the values you name
already computed.

.. code-block:: python

   import sparrow
   from sparrow import Protein

   p = Protein("MEEEKKKKSSSTTTDDDQQQQNNNN")
   p.FCR                                # fraction of charged residues
   p.kappa                              # charge patterning
   p.predictor.disorder()               # per-residue disorder, from metapredict
   p.predictor.radius_of_gyration()     # ALBATROSS radius of gyration, in Angstroms

   proteins = sparrow.batch("idrs.fasta", ["FCR", "kappa", "predictor.radius_of_gyration"])

How these docs are organized
----------------------------

:doc:`overview/index`
   What sparrow is and what it can do, how it is organized inside, and how to
   install it.

:doc:`protein/index`
   Everything a ``Protein`` gives you, one page for each kind of thing: its
   composition, charge, hydrophobicity, patterning, profiles, domains and
   motifs, predictions, polymer models and molecular grammar.

:doc:`batch/index`
   :func:`sparrow.batch`, for fifty sequences or fifty thousand, and
   :doc:`every analysis it can be asked for <batch/analyses>`, with its
   arguments and their defaults.

:doc:`null/index`
   Null sequences to compare a sequence with, each with its exact composition:
   shuffles in blocks, shuffles that keep every dipeptide, and sequences
   rebuilt from the k-mers or Markov chain of a background such as a proteome.
   How to choose one, how each works, and how to write your own.

:doc:`power_user/index`
   The functions underneath, for when you want arrays rather than ``Protein``
   objects, or more control: properties for a whole batch at once, running the
   networks directly, NARDINI+ over a proteome, the polymer models,
   annotations, alignment, plotting and the bundled data.

:doc:`api/index`
   Every module, class and function, documented from its docstring.

:doc:`examples`
   Short walkthroughs that end in a figure or a result: a linear NCPR profile,
   NARDINI+ over a proteome, and predictions for many sequences.

:doc:`adding_a_predictor` and :doc:`writing_a_plugin`
   How to add a predictor, from a network trained with PARROT to a
   ``Protein.predictor`` method that ``sparrow.batch`` can run, and how to
   contribute an analysis as a plugin.

:doc:`citing`
   How to cite sparrow, and the original work behind each tool.

:doc:`changelog`
   Every change to sparrow.

.. toctree::
   :maxdepth: 2
   :hidden:

   overview/index
   protein/index
   batch/index
   null/index
   power_user/index
   api/index
   examples
   adding_a_predictor
   writing_a_plugin
   citing
   changelog

License
-------

sparrow is released under the MIT License. The PLAAC port and the bundled
NARDINI+ / GIN reference data are MIT-licensed by their own authors, and their
notices ship beside them. See ``LICENSE`` and the README in the
`repository <https://github.com/idptools/sparrow>`_.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
