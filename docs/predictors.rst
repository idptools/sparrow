Predictors
=================

.. note::
   This page is a focused guide for **adding** a predictor. To **use** the
   existing predictors from a ``Protein`` object, see
   :doc:`The Protein Object <api_guides/protein>`.

sparrow implements a set of different sequence-based predictors in a modular, extendable way that enables additional predictors to be easily added. 


Creating new predictors with PARROT
--------------------------------------
The guide below assumes you have cloned the git repository of sparrow, created a new branch to add your new predictor to, and have switched into that branch to begin work. As a reminder, when adding new features in Git, the general workflow is:

1. Clone the current up-to-date version
2. Create a new branch (this is a seperate version where you can work in peace, but if new features are added to the main branch you can update your branch as you go)
3. Add in your snazzy new feature
4. Once complete, make a pull request to merge your branch back into the main branch.

This guide assumes these ideas are clear, and specifically provides insight into the details of implementing a new predictor in sparrow, focussing here on using PARROT to train that predictor. 


**Step 1: Train a predictor with PARROT 2**

The first step in adding a new PARROT based predictor is to use PARROT 2 to train your model. PARROT 2 has not been released yet, so install it from its branch (sparrow itself only needs PARROT 1, which is what ``pip install idptools-parrot`` gives you)::

  pip install "git+https://github.com/idptools/parrot.git@parrot2"

The details of training go beyond the scope of this documentation (``parrot-train --show-examples`` is a good place to start), but as a concrete example::

  parrot-train brnn --input-file data.tsv --output-network my_property_network_v1.pt --split kmer -d residues --regression

Once trained you're left with a Torch file (a ``.pt`` file), which is the file we're going to use with SPARROW to add our custom predictor, and a ``my_property_network_v1_info.txt`` describing how it was trained, which is worth keeping with your notes.

You can train the network in ``residues`` mode (one value per residue, like the disorder or transmembrane predictors) or in ``sequence`` mode (one value per sequence, like the ALBATROSS radius of gyration), for regression or classification, with any of PARROT's architectures and encodings. A PARROT 2 network file records all of this, so there is nothing to declare about it in sparrow.

Note that anyone using your predictor will need PARROT 2 installed too, until it is released; without it, sparrow raises an error saying how to install it.


**Step 2: Copy the parameter file into SPARROW**

Next we take ``my_property_network_v1.pt`` and we're going to copy it into sparrow. Specifically, this trained network should be placed in its own directory under ``sparrow/data/networks/``::

  sparrow/data/networks/<predictor_name>/

and MUST follow the naming convention ``<file_stem>_network_v<X>.pt``. Note there that:

* ``<predictor_name>`` should be a single word or word connected by underscores, all lower case, that we will use as the function name to call the predictor. For example, *disorder*, *dssp* or *transmembrane* are good examples. Keep this simple but it should be clear and unambigious.
* ``<file_stem>`` is usually just ``<predictor_name>``, though several of the existing networks append ``_predictor`` (see the examples below). Whatever you choose must match the ``NETWORK_PATH`` template you set in step 3.
* ``<X>`` here is the specific version of this network. It is possible that your network may be retrained later, and as such we want to enable future sparrow users to select specific network versions, althogh of the course the predictors should default to the most recent version. This ability to select specific network versions is built into the standard predictor template code.

As examples, the transmembrane and radius-of-gyration networks are stored as::

  sparrow/data/networks/transmembrane/transmembrane_predictor_network_v4.pt
  sparrow/data/networks/rg/rg_network_v2.pt


**Step 3: Build a predictor class which performs the prediction**

The next step is to build a stand-alone predictor class which reads in this network file and enables the return of the per-residue prediction implemented therein. This file should be created in a new package (i.e. a directory with a ``__init__.py``) in the::

  sparrow/predictors

and this file should be called ``<relevant_name>_predictor.py``.

As a specific example, our transmembrane predictor is implemented in::

  sparrow/predictors/transmembrane

and within this directory there are two files::

  __init__.py # this is needed so we can import the predictor
  transmembrane_predictor.py # this is where the predictor is implemented

The reason to make a separate package (directory) for every predictor is that if someone has a non-parrot-based predictor they want to incoporate into sparrow (1) this is absolutely welcome and (2) we want to provide a consistent file ecosystem where they have a directory to implement as much/little additional code as they want. As such, the ``__init__.py`` and ``<predictor_name>_predictor.py`` are the **minimum** files needed, but you are free to add anything else as well.

``__init__.py`` should probably just be empty - it's what tells Python that this directory is a package. 

``<relevant_name>_predictor.py`` should NOT be empty, but it is short: copy the template ``sparrow/predictors/predictor_template.py`` into it and fill it in. All network loading is centralized, so the predictor is a thin subclass of :class:`sparrow.predictors.base.BaseNetworkPredictor`, which reads the ``.pt`` file via the shared :func:`sparrow.predictors.network_loader.load_parrot_network` and runs the network. You only need to:

* rename the class (``MyPropertyPredictor``) and its method (``predict_my_property``);
* set ``NETWORK_PATH`` -- a ``{version}`` template for the weights file under ``sparrow/data/`` (here ``"networks/my_property/my_property_network_v{version}.pt"``) -- and ``DEFAULT_VERSION``, the version loaded when none is asked for;
* reduce PARROT's prediction to what the predictor should return, in ``_from_raw``. For a regression network, PARROT's prediction is one value per residue, or a one-element array for a ``sequence`` network; for a classification network it is the probability of each class (per residue, or for the sequence).

The whole template is:

.. literalinclude:: ../sparrow/predictors/predictor_template.py
   :language: python

``sparrow/tests/test_predictors_parrot2.py`` runs this template, with only its weights path changed, on per-residue and per-sequence networks saved by PARROT 2, so it is known to work as written.

.. note::
   Every network sparrow currently ships was trained with PARROT 1, which saved only a BRNN's weights. Those predictors (for example ``sparrow/predictors/rg/radius_of_gyration_predictor.py``) also declare ``ARCHITECTURE`` -- ``"MtO"`` for one value per sequence or ``"MtM"`` for one per residue -- and their ``_from_raw`` turns the network's raw output into a result with a helper from :mod:`sparrow.predictors.outputs`. A network saved by PARROT 2 records its architecture, so new predictors need neither.

The predictor's ``_from_raw`` is the one place its prediction is made, from the network's output: ``predict(seq)`` calls it for one sequence, and ``predict_many(sequences)`` for many, run through the network in batches - many times faster than one at a time. A network saved by PARROT 2 is run in batches by PARROT's own ``Predictor.predict_batch``, and one saved by PARROT 1 by sparrow's batch harness, :func:`~sparrow.predictors.batch_predictor.run_network`; either way the predictor is called the same way, so nobody using it needs to know which PARROT trained it. Batching needs a PARROT 2 recent enough to have ``predict_batch``, and sparrow refuses to load a network saved by PARROT 2 with an earlier development version, saying how to update it.

Once this is done, it's worth seeing if you can import and run predictions using this class as a stand-alone predictor i.e. you should be able to do::

    from sparrow.predictors.my_property.my_property_predictor import MyPropertyPredictor

    sequence = 'MSAAVTAGKLARAPADPGKAGVPGVAAPGAPAAAPPAKEIPEVLVDPRSRRRYVRGRFLG'
    P = MyPropertyPredictor()
    P.predict_my_property(sequence)

and it return a set of values.


**Step 4: Integrate the predictor in the sparrow.predictors.Predictor class**

At this stage we have a working predictor - the last step is to connect this predictor to the sparrow Protein object in a way that inccurs minimal computational overhead if not used, is syntactically simple, and offers functionality like other Protein analysis functions and properties.

This is achieved by adding a function into the :class:`sparrow.predictors.Predictor` class, a class implemented in ``sparrow/predictors/__init__.py``.

This class generates an object which is accessible in Protein object under the ``.predictor`` dot operator. As such, functions defined in the ``sparrow.predictors.Predictor`` class are then accessible as::

  seq = 'MSAAVTAGKLARAPADPGKAGVPGVAAPGAPAAAPPAKEIPE'
  p = Protein(seq)

  p.predictor.<predictor function>


As such, to finally make a new predictor accessible, three small edits to ``sparrow/predictors/__init__.py`` are needed:

1. Register the network in ``NETWORK_PREDICTORS``, under a short name, with the module and class of your predictor: ``"my_property": ("sparrow.predictors.my_property.my_property_predictor", "MyPropertyPredictor")``. The module is imported only when the network is first used, so an unused predictor costs nothing.
2. Add a method to the ``Predictor`` class, named for the prediction (e.g. ``my_property``), that reads the network's prediction with ``self._prediction("my_property", recompute)`` and returns it - wrapped in ``copy_of``, so that a caller who changes the result cannot change the cache - or a value made from it. ``_prediction`` runs the network on first use and caches its prediction, so the network runs once for a sequence, whatever its methods are asked.
3. Add the method to ``METHOD_NETWORKS``, with the network's name. That is what lets :func:`sparrow.batch` predict it for a whole batch at once, and ``batch_predict(sequences, network="my_property")`` runs it from the registration alone.

``transmembrane_regions()`` is the simplest example, and ``dssp_helicity()`` one that makes its result from the prediction with options of its own.

One important thing is to document these predictors clearly. In particular:

* Give every option a numpydoc ``Parameters`` entry, including its default. Options that exist in the signature but are absent from the docstring are effectively invisible to users.
* If a predictor memoizes its result, make the memoization key include **every** argument that can change the returned value. A selector that ignores an argument will silently hand back a result computed with different settings - see ``binary_disorder()`` for the pattern to follow.
* State the return type and units explicitly (for example, Rg and Re are returned in Angstroms).

Reference: the shared predictor machinery
-----------------------------------------

.. automodule:: sparrow.predictors.base

.. automodule:: sparrow.predictors.network_loader
   :exclude-members: number_of_classes, input_size, number_of_layers, hidden_vector_size, device,
                     saved_by_parrot2, datatype

.. automodule:: sparrow.predictors.outputs
