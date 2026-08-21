Installation
============

sparrow runs on **Python 3.10+** and is **pure Python** - there is nothing to
compile, so no C compiler is required. The sections below cover setting up an
isolated environment and installing sparrow directly from GitHub with either
``pip`` or ``uv``.

.. note::
   Releases before this one shipped Cython extensions for the patterning
   metrics and needed a working compiler to install. Those were replaced by
   vectorized NumPy implementations - see
   :mod:`sparrow.patterning.vectorized` - so installation is now a plain
   pure-Python wheel.

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

To install a specific tagged release, append ``@<tag>``:

.. code-block:: bash

   pip install git+https://github.com/idptools/sparrow.git@1.0.0

Install for development
^^^^^^^^^^^^^^^^^^^^^^^

If you want to work on sparrow itself, clone it and install in editable mode:

.. code-block:: bash

   git clone https://github.com/idptools/sparrow.git
   cd sparrow
   pip install -e .

   # run the test suite (works from the repository root)
   pytest sparrow/tests

There is no build step - sparrow is pure Python.

.. warning::
   If you are upgrading a checkout that predates the removal of the Cython
   extensions, delete any stale compiled artifacts first::

       rm -f sparrow/patterning/*.so sparrow/patterning/*.c

   Python looks for extension modules before source modules, so a leftover
   ``kappa.cpython-*.so`` would shadow the new ``kappa.py`` and you would
   silently keep running the old compiled code.

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
