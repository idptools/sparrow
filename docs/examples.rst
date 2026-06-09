Worked Examples
===============

Short, copy-pasteable walkthroughs. Each one starts from a ``Protein`` object
and builds toward a figure or a result you can use.

.. contents:: On this page
   :local:
   :depth: 1

Plotting a linear NCPR profile
------------------------------

The **net charge per residue (NCPR)** averaged in a sliding window is one of the
most useful ways to *see* how charge is distributed along a disordered protein.
sparrow computes it with
:meth:`~sparrow.protein.Protein.linear_sequence_profile`; here we plot it with
matplotlib.

Step 1 -- create a protein
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sparrow import Protein

   # a sequence with distinct acidic and basic blocks, so the profile is interesting
   seq = (
       "MASNDDDDEEEDDEEGGSGSGSGSGSGSKKKRKKRKRKKGGSGSGSGSGS"
       "DDEEDDEEDDEEGGSGSGSGSKKRKKRKKRKKRGGSGSGSGSGSGSGSGS"
   )
   p = Protein(seq)

Step 2 -- compute the linear NCPR profile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``linear_sequence_profile`` returns one value per residue (the window-averaged
NCPR centred on that position). A window of 5-9 residues is typical.

.. code-block:: python

   ncpr_profile = p.linear_sequence_profile(mode="NCPR", window_size=7)

   # the profile has the same length as the sequence
   assert len(ncpr_profile) == len(p)

Step 3 -- plot it
^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   positions = np.arange(1, len(p) + 1)      # 1-indexed residue positions
   ncpr = np.asarray(ncpr_profile)

   fig, ax = plt.subplots(figsize=(10, 3))

   # shade positive (basic) and negative (acidic) regions
   ax.fill_between(positions, ncpr, 0, where=ncpr >= 0, color="#2900f5", alpha=0.6, label="net positive")
   ax.fill_between(positions, ncpr, 0, where=ncpr < 0, color="#ff0d0d", alpha=0.6, label="net negative")

   ax.axhline(0, color="black", linewidth=0.8)
   ax.set_xlabel("Residue position")
   ax.set_ylabel("NCPR (window = 7)")
   ax.set_title("Linear NCPR profile")
   ax.set_xlim(1, len(p))
   ax.legend(loc="upper right", frameon=False)
   fig.tight_layout()

   fig.savefig("ncpr_profile.png", dpi=150)
   plt.show()

The resulting figure shows blue (net-positive) and red (net-negative) blocks
along the sequence, making charge segregation immediately visible -- the same
blockiness that the scalar :attr:`~sparrow.protein.Protein.kappa` summarises in a
single number.

Variations
^^^^^^^^^^

* **Smoother profile:** increase ``window_size`` (e.g. ``window_size=11``).
* **Other linear properties:** swap ``mode`` for ``"FCR"``, ``"hydrophobicity"``,
  ``"aromatic"``, etc. -- see
  :meth:`~sparrow.protein.Protein.linear_sequence_profile`.
* **Published amino-acid scales:** use
  :meth:`~sparrow.protein.Protein.linear_property_profile` with an AAindex index,
  e.g. ``p.linear_property_profile("hydropathy-kyte-1982", window_size=9)``.

Overlaying a hydropathy profile
-------------------------------

You can plot any number of profiles on shared axes. Here we add a Kyte-Doolittle
hydropathy track (from the :doc:`AAindex property database <api_guides/properties>`)
beneath the NCPR profile.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from sparrow import Protein

   p = Protein("MEEEKKKKSSSTTTDDDQQQQNNNNGGGGSSSSLLLVVVAAAFFFWWW")
   positions = np.arange(1, len(p) + 1)

   ncpr = np.asarray(p.linear_sequence_profile(mode="NCPR", window_size=7))
   hydro = np.asarray(p.linear_property_profile("hydropathy-kyte-1982", window_size=7))

   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
   ax1.plot(positions, ncpr, color="#2900f5")
   ax1.axhline(0, color="black", lw=0.8)
   ax1.set_ylabel("NCPR")

   ax2.plot(positions, hydro, color="#04700d")
   ax2.set_ylabel("Hydropathy\n(Kyte-Doolittle)")
   ax2.set_xlabel("Residue position")
   fig.tight_layout()
   plt.show()

Predicting properties for many sequences
----------------------------------------

To compute one ALBATROSS prediction (here radius of gyration) over a whole FASTA
file efficiently, use :func:`~sparrow.predictors.batch_predict.batch_predict`:

.. code-block:: python

   from sparrow import read_fasta
   from sparrow.predictors.batch_predict import batch_predict

   proteins = read_fasta("example.fasta")          # {header: Protein}
   rg_by_header = batch_predict(proteins, network="rg", return_seq2prediction=False)

   for header, (sequence, rg) in rg_by_header.items():
       print(header, round(rg, 2))

See :doc:`api_guides/batch` for all batch options.
