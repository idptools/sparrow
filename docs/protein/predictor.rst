Predictions (``protein.predictor``)
===================================

``protein.predictor`` gives sparrow's deep-learning predictions for the
sequence. Each prediction's network is loaded the first time it is needed, and
each prediction is made once and cached, so repeated calls are free and a
predictor you never use costs nothing.

.. code-block:: python

   from sparrow import Protein

   p = Protein("MKYLAAYLLLNAAGNTPDATKIKAILESVGIEIEDEKVSSVLSALEGKSVDELITEGNEKLAAVPAAGPASAGGAAAASGDAAAEEEKEEEAAEESDDDMGFGLFD")

   p.predictor.disorder()                 # per-residue disorder, from metapredict
   p.predictor.dssp_helicity()            # 1 for each residue predicted helical
   p.predictor.radius_of_gyration()       # ALBATROSS Rg, in Angstroms
   p.predictor.serine_phosphorylation()   # 1 for each predicted phosphoserine

Every prediction method but the three DSSP methods takes ``recompute=True``,
to make the prediction again rather than return the cached one. For many sequences, :func:`sparrow.batch`
makes these predictions for the whole batch at once, which is many times
faster (:doc:`../batch/index`); :doc:`../power_user/networks` covers running
the networks directly, on the CPU or a GPU, and choosing a network version.

Disorder and secondary structure
--------------------------------

.. code-block:: python

   p.predictor.disorder()                 # per-residue disorder score, 0-1
   p.predictor.disorder_domains()         # 1 for each residue in an IDR
   p.predictor.disorder_domains(return_boundaries=True)   # [[start, end], ...]
   p.predictor.binary_disorder(disorder_threshold=0.6)    # the IDRs, with metapredict's options
   p.predictor.pLDDT()                    # predicted AlphaFold2 confidence, 0-1

   p.predictor.dssp_helicity()            # 1/0 for each residue, from DSSP-trained networks
   p.predictor.dssp_extended(mode="probability")
   p.predictor.dssp_coil(mode="both")     # (classes, probabilities)

Disorder and pLDDT come from `metapredict <https://github.com/idptools/metapredict>`_
(Emenecker *et al.*), with its default networks. ``disorder_domains`` and
``binary_disorder`` both use metapredict's IDR-definition algorithm rather than
a threshold on each residue's score; ``binary_disorder`` exposes its options.

The three DSSP methods read one network, which predicts the probability of
helix, extended and coil at each residue, so asking for all three runs it once.
Each turns its column into classes with a ``threshold`` and a minimum region
length, or gives the probabilities (``mode="probability"``), or both.

Polymer dimensions
------------------

The ALBATROSS networks (Lotthammer *et al.*, 2024) predict the dimensions of a
disordered chain from its sequence, as single values:

.. code-block:: python

   p.predictor.radius_of_gyration()                   # Rg, in Angstroms
   p.predictor.radius_of_gyration(use_scaled=False)   # the network trained on Rg itself
   p.predictor.end_to_end_distance()                  # Re, in Angstroms
   p.predictor.scaling_exponent()                     # nu
   p.predictor.asphericity()
   p.predictor.prefactor()                            # A0, in Angstroms

The ``Rg`` and ``Re`` predictors take a ``use_scaled`` flag selecting between
the networks trained on ``Rg/sqrt(N)`` and ``Re/sqrt(N)`` (the default, and
recommended) and those trained on the raw values. For sequences shorter than
35 residues (``sparrow.data.configs.MIN_LENGTH_ALBATROSS_RE_RG``) the scaled
network is used regardless, unless you pass ``safe=False``.

``protein.polymeric`` reads these predictions too, and adds polymer-model
distributions built from them; see :doc:`polymeric`.

Modification, localization and phase behavior
---------------------------------------------

.. code-block:: python

   p.predictor.serine_phosphorylation()                    # 1/0 mask of predicted sites
   p.predictor.threonine_phosphorylation(raw_values=True)  # per-residue probabilities
   p.predictor.tyrosine_phosphorylation(return_sites_only=True)   # 0-indexed positions

   p.predictor.nuclear_import_signal()      # per-residue probability of an NLS
   p.predictor.nuclear_export_signal()      # ... of an NES
   p.predictor.transactivation_domains()    # ... of a transactivation domain
   p.predictor.transmembrane_regions()      # 1 for each transmembrane residue
   p.predictor.mitochondrial_targeting_sequence()   # 1 for each residue of an N-terminal MTS
   p.predictor.pscore()                     # phase-separation propensity, rescaled to 0-1

Reference
---------

.. autoclass:: sparrow.predictors.Predictor
   :no-members:
   :no-inherited-members:
   :no-undoc-members:

.. autoproperty:: sparrow.protein.Protein.predictor

Disorder and secondary structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   sparrow.predictors.Predictor.disorder
   sparrow.predictors.Predictor.disorder_domains
   sparrow.predictors.Predictor.binary_disorder
   sparrow.predictors.Predictor.pLDDT
   sparrow.predictors.Predictor.dssp_helicity
   sparrow.predictors.Predictor.dssp_extended
   sparrow.predictors.Predictor.dssp_coil

.. automethod:: sparrow.predictors.Predictor.disorder

.. automethod:: sparrow.predictors.Predictor.disorder_domains

.. automethod:: sparrow.predictors.Predictor.binary_disorder

.. automethod:: sparrow.predictors.Predictor.pLDDT

.. automethod:: sparrow.predictors.Predictor.dssp_helicity

.. automethod:: sparrow.predictors.Predictor.dssp_extended

.. automethod:: sparrow.predictors.Predictor.dssp_coil

Polymer dimensions
^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   sparrow.predictors.Predictor.radius_of_gyration
   sparrow.predictors.Predictor.end_to_end_distance
   sparrow.predictors.Predictor.scaling_exponent
   sparrow.predictors.Predictor.asphericity
   sparrow.predictors.Predictor.prefactor

.. automethod:: sparrow.predictors.Predictor.radius_of_gyration

.. automethod:: sparrow.predictors.Predictor.end_to_end_distance

.. automethod:: sparrow.predictors.Predictor.scaling_exponent

.. automethod:: sparrow.predictors.Predictor.asphericity

.. automethod:: sparrow.predictors.Predictor.prefactor

Modification, localization and phase behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   sparrow.predictors.Predictor.serine_phosphorylation
   sparrow.predictors.Predictor.threonine_phosphorylation
   sparrow.predictors.Predictor.tyrosine_phosphorylation
   sparrow.predictors.Predictor.nuclear_import_signal
   sparrow.predictors.Predictor.nuclear_export_signal
   sparrow.predictors.Predictor.transactivation_domains
   sparrow.predictors.Predictor.transmembrane_regions
   sparrow.predictors.Predictor.mitochondrial_targeting_sequence
   sparrow.predictors.Predictor.pscore

.. automethod:: sparrow.predictors.Predictor.serine_phosphorylation

.. automethod:: sparrow.predictors.Predictor.threonine_phosphorylation

.. automethod:: sparrow.predictors.Predictor.tyrosine_phosphorylation

.. automethod:: sparrow.predictors.Predictor.nuclear_import_signal

.. automethod:: sparrow.predictors.Predictor.nuclear_export_signal

.. automethod:: sparrow.predictors.Predictor.transactivation_domains

.. automethod:: sparrow.predictors.Predictor.transmembrane_regions

.. automethod:: sparrow.predictors.Predictor.mitochondrial_targeting_sequence

.. automethod:: sparrow.predictors.Predictor.pscore
