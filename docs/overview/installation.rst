Installation
============

sparrow runs on **Python 3.10+** and is **pure Python** - there is nothing to
compile, so no C compiler is required. The sections below cover setting up an
isolated environment and installing sparrow directly from GitHub with either
``pip`` or ``uv``.

.. note::
   Releases before this one shipped Cython extensions for the patterning
   metrics and needed a working compiler to install. Those were replaced by
   vectorized NumPy implementations - see :mod:`sparrow.properties.patterning`
   - so installation is now a plain pure-Python wheel.

.. contents:: On this page
   :local:
   :depth: 2

1. Create a virtual environment (recommended)
---------------------------------------------

Installing into an isolated environment keeps sparrow and its dependencies from
interfering with other projects. Use either the built-in ``venv`` (with
``pip``) or `uv <https://docs.astral.sh/uv/>`_ (a fast drop-in replacement).

With ``venv`` + ``pip``
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # create the environment
   python -m venv sparrow-env

   # activate it
   source sparrow-env/bin/activate        # macOS / Linux
   # sparrow-env\Scripts\activate         # Windows (PowerShell/cmd)

   # make sure pip is current
   python -m pip install --upgrade pip

With ``uv``
^^^^^^^^^^^

.. code-block:: bash

   # create and activate
   uv venv sparrow-env
   source sparrow-env/bin/activate        # macOS / Linux
   # sparrow-env\Scripts\activate         # Windows

``uv`` can also manage the environment for you transparently; in that case use
``uv pip install ...`` in place of ``pip install ...`` below.

2. Install sparrow
------------------

Install from GitHub
^^^^^^^^^^^^^^^^^^^

sparrow is currently installed directly from the repository:

.. code-block:: bash

   # with pip
   pip install git+https://github.com/idptools/sparrow.git

   # with uv
   uv pip install git+https://github.com/idptools/sparrow.git

To install a specific release instead, append its tag, as listed on the
`tags page <https://github.com/idptools/sparrow/tags>`_:

.. code-block:: bash

   pip install git+https://github.com/idptools/sparrow.git@<tag>

sparrow takes its version from these tags when it is built, so
``sparrow.__version__`` tells you which one you have; between releases it also
names the commit, for example ``0.2.2+97.gc0df3a3``. This documentation was built
from sparrow |release|.

Install for development
^^^^^^^^^^^^^^^^^^^^^^^

If you want to work on sparrow itself, clone it and install in editable mode:

.. code-block:: bash

   git clone https://github.com/idptools/sparrow.git
   cd sparrow
   pip install -e ".[test]"       # or ".[dev]" to add pre-commit as well

   # run the test suite (works from the repository root)
   pytest sparrow/tests

There is no build step - sparrow is pure Python.

.. warning::
   If you are upgrading a checkout that predates the removal of
   ``sparrow.patterning``, delete that directory first::

       rm -r sparrow/patterning

   A compiled ``kappa.cpython-*.so`` left in it would still be importable, so
   code that imports ``sparrow.patterning.kappa`` would silently keep running
   the old compiled version rather than failing.

3. Verify the installation
--------------------------

.. code-block:: bash

   python -c "import sparrow; print(sparrow.__version__)"

.. code-block:: python

   from sparrow import Protein

   p = Protein("MEEEKKKKSSSTTTDDD")
   print(p.FCR, p.NCPR, p.kappa)

If those run without error, sparrow is installed correctly.

Dependencies
------------

Core sequence analysis works out of the box. The deep-learning predictors
(reached via ``Protein.predictor``) rely on PyTorch through the ``parrot``
dependency, which is installed automatically; the first call to a given
predictor loads its network lazily.

Multiple sequence alignment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~sparrow.sequence_analysis.alignment.SequenceAlignment` aligns
sequences with `pyFAMSA <https://github.com/althonos/pyfamsa>`_, which sparrow
installs only if you ask for its ``alignment`` extra:

.. code-block:: bash

   pip install "idptools-sparrow[alignment] @ git+https://github.com/idptools/sparrow.git"

   # or, in a clone
   pip install -e ".[alignment]"

Nothing else in sparrow needs pyFAMSA. Without it, importing
``sparrow.sequence_analysis.alignment`` raises an error saying how to install
it. The ``test`` extra includes it, so the test suite covers alignment too.

Next steps
----------

* New to sparrow? See :doc:`../examples` for worked, runnable
  walkthroughs.
* Looking for a specific call? Everything you can do with a protein is in
  :doc:`../protein/index`, and with many in :doc:`../batch/index`.
