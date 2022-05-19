Predictors
=================

sparrow implements a set of different sequence-based predictors in a modular, extendable way that enables additional predictors to be easily added. 


Creating new predictors with PARROT
--------------------------------------
The guide below assumes you have cloned the git repository of sparrow, created a new branch to add your new predictor to, and have switched into that branch to begin work. As a reminder, when adding new features in Git, the general workflow is:

1. Clone the current up-to-date version
2. Create a new branch (this is a seperate version where you can work in peace, but if new features are added to the main branch you can update your branch as you go)
3. Add in your snazzy new feature
4. Once complete, make a pull request to merge your branch back into the main branch.

This guide assumes these ideas are clear, and specifically provides insight into the details of implementing a new predictor in sparrow, focussing here on using PARROT to train that predictor. 


**Step 1: Train a predictor with PARROT**

The first step in adding a new PARROT based predictor is to use PARROT to train your model. The details of how one does this go beyond the scope of this documentation, but once trained you should be left with a Torch parameter file (a ``.pt`` file). This is the file we're going to use with SPARROW to add our custom predictor. Lets call this parameter file ``new_predictor.pt`` to make this concrete.

Note that the PARROT predictor should be predicted in ``residues`` mode - i.e. we need to recieve one value per residue


**Step 2: Copy the parameter file into SPARROW**

Next we take ``new_predictor.pt`` and we're going to copy it into sparrow. Specifically, this trained network should be placed under::

  sparrow/data/networks/predictor

and MUST follow the naming convention ``<predictor_name>_network_v<X>.pt``. Note there that:

* ``<predictor_name>`` should be a single word or word connected by underscores, all lower case, that we will use as the function name to call the predictor. For example, *disorder*, *dssp* or *transmembrane* are good examples. Keep this simple but it should be clear and unambigious.
* ``<X>`` here is the specific version of this network. It is possible that your network may be retrained later, and as such we want to enable future sparrow users to select specific network versions, althogh of the course the predictors should default to the most recent version. This ability to select specific network versions is built into the standard predictor template code.

As an example, our transmembrane predictor has the format::

  transmembrane_predictor_network_v4.pt


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

``__init__.py``` should probably just be empty - it's what tells Python that this directory is a package. 

``<relevant_name>_predictor.py`` should NOT be empty, but should be based on the template file found under ``sparrow/predictors/predictor_template.py``. The template is REALTIVELY simple, but provides code for reading in a PARROT-trained network and performing a prediction. You could re-implement this yourself if you really wanted, but, assuming you're using one-hot encoding on the trained network, this code should work out of the box. The template itself walks through the various small configuration tweaks needed to make this work with your specific network of interest. Note that for classification vs. regression there are some small difference, but the template file provides code for both, so just delete/comment out the irrelevant lines (these are clearly marked).

Once this is done, it's worth seeing if you can import and run predictions using this class/function as a stand-alone predictor i.e. you should be able to do::


    from sparrow.predictor.<predictor_package>.<predictor_module> import <RelevantName>Predictor
    
    sequence = 'MSAAVTAGKLARAPADPGKAGVPGVAAPGAPAAAPPAKEIPEVLVDPRSRRRYVRGRFLG'
    P = <RelevantName>Predictor()
    P.predict_<SOMETHING_RELEVANT>(sequence)


and it return a set of values.


**Step 4: Integrate the predictior in the sparrow.Predictor class**

At this stage we have a working predictor - the last step is to connect this predictor to the sparrow Protein object in a way that inccurs minimal computational overhead if not used, is syntactically simple, and offers functionality like other Protein analysis functions and properties.

This is achieved by adding a function into the ``sparrow.predictor.Predictor`` class, a class implemented in the ``sparrow.predictor.__init__.py``.

This class generates an object which is accessible in Protein object under the ``.predictor`` dot operator. As such, functions defined in the `sparrow.predictor.Predictor`` class are then accessible as::

  seq = 'MSAAVTAGKLARAPADPGKAGVPGVAAPGAPAAAPPAKEIPE'
  p = Protein(seq)

  p.predictor.<predictor function>


As such, to finally make a new predictor accessible, ``sparrow.predictor.Predictor`` class should be edited to add a new function which is simply the name of the prediction (e.g. ``dssp``, ``transmembrane`` etc). This function should do three things:

1. It should UPON BEING CALLED import the predictor package you just created.
2. It should then perform the prediction on the underlying protein sequence
3. It should (ideally) memoize the outcome into a local dictionary that means if the same prediction is called again it is simply referenced rather than recomputed.

Rather than going into the details here, the underlying code and example should make this clear. Noteably, see ``dssp()`` and ``transmembrane_regions()`` for good examples of PARROT-based predictors. One important thing is to document these predictors clearly








