Getting Started
===============

This page is a lightweight quickstart for installing and using sparrow.

Installation
------------

Install from PyPI
^^^^^^^^^^^^^^^^^

With ``pip``:

.. code-block:: bash

   pip install idptools-sparrow

With ``uv pip``:

.. code-block:: bash

   uv pip install idptools-sparrow

Install from GitHub
^^^^^^^^^^^^^^^^^^^

With ``pip``:

.. code-block:: bash

   pip install git+https://github.com/idptools/sparrow.git

With ``uv pip``:

.. code-block:: bash

   uv pip install git+https://github.com/idptools/sparrow.git

Quick Tutorial
--------------

Create a ``Protein`` object
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sparrow import Protein

   p = Protein("MGSQSSRSSSQQQQQQQ")

   print(p.sequence)
   print(p.FCR)
   print(p.NCPR)
   print(p.kappa)

Use a few common analyses
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sparrow import Protein

   p = Protein("QQQQQQAASSSSTTTTQQQQQ")

   print(p.amino_acid_fractions["Q"])
   print(p.FCR)
   print(p.NCPR)
   print(p.kappa)
   print(p.complexity)
   

Read proteins from a FASTA file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sparrow import read_fasta

   proteins = read_fasta("example.fasta")
   print(list(proteins.keys())[:3])

   first_header = next(iter(proteins))
   first_protein = proteins[first_header]
   print(first_header, len(first_protein), first_protein.FCR)

Notes
-----

* ``read_fasta()`` returns a dictionary mapping FASTA headers to ``Protein`` objects.
* The function forwards keyword arguments to ``protfasta.read_fasta`` for advanced parsing controls.

Next steps
----------

* For the full API by workflow, see :doc:`api`.
* For predictor-specific guidance, see :doc:`predictors`.
