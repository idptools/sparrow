Installation
============

sparrow runs on **Python 3.7+** and includes compiled (Cython) extensions. The
sections below cover setting up an isolated environment and installing sparrow
with either ``pip`` or ``uv``, from PyPI or directly from GitHub.

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

Install from PyPI (stable)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # with pip
   pip install idptools-sparrow

   # with uv
   uv pip install idptools-sparrow

Install from GitHub (latest)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To get the most recent development version straight from the repository:

.. code-block:: bash

   # with pip
   pip install git+https://github.com/idptools/sparrow.git

   # with uv
   uv pip install git+https://github.com/idptools/sparrow.git

.. note::
   Installing from GitHub builds the Cython extensions locally, so you need a
   working C compiler (Xcode command-line tools on macOS, ``build-essential`` on
   Debian/Ubuntu, or MSVC build tools on Windows). NumPy is required at build
   time and is installed automatically.

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

Next steps
----------

* New to sparrow? See :doc:`examples` for worked, runnable walkthroughs.
* Looking for a specific call? Everything you can do with a protein is in
  :doc:`api_guides/protein`.
