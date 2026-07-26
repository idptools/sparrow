Worked Examples
===============

Short, copy-pasteable walkthroughs. Each one starts from a ``Protein`` object
and builds toward a figure or a result you can use.

.. tip::
   Two longer, fully executed Jupyter notebooks live in ``examples/`` in the
   repository:

   * ``nardini_from_protein_objects.ipynb`` -- NARDINI+ for a single sequence via
     ``Protein.nardini``, worked through on a real human IDR.
   * ``nardini_module.ipynb`` -- the batch interface: building a compositional
     background, scoring a set of sequences, and GIN cluster assignment.

   Both have their outputs saved, so you can read them without running anything.

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

NARDINI+ vectors for a whole proteome
-------------------------------------

The realistic molecular-grammar workflow: read IDRs from a FASTA, compute every
NARDINI+ vector in parallel, and get back ``Protein`` objects with the results
attached so nothing is recomputed afterwards. See :doc:`api_guides/nardini` for the
full guide.

Step 1 -- read the sequences
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from sparrow import read_fasta

   idrs = read_fasta("my_idrs.fasta")        # {header: Protein}

Step 2 -- compute every vector in parallel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:func:`~sparrow.nardini.batch.nardini_proteins` accepts sequences or ``Protein``
objects, as a dict or a list, and returns the same shape back. The expensive part
is the scramble-based patterning null, which is why this is worth parallelizing:
on 160 IDRs at 1,000 scrambles it takes 18.7 s on one core and 2.7 s on eight.

.. code-block:: python

   from sparrow.nardini import nardini_proteins

   proteins = nardini_proteins(
       idrs,
       num_scrambles=10000,     # the published protocol uses 100,000
       seed=42,                 # pass one, or the numbers are not reproducible
       show_progress=True,
       on_error="skip",         # real proteomes contain unscoreable entries
   )

Step 3 -- read the results, free of charge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The vectors are cached on each ``Protein``, so these calls do no work -- as long as
the options match the ones you computed with.

.. code-block:: python

   import numpy as np

   protein = proteins["IDR1"]

   vector = protein.nardini.vector(seed=42, num_scrambles=10000)
   vector["Frac K+R"]                 # one feature by name
   vector.values                      # the underlying (90,) array

   # nearest published human GIN grammar, plus whether to believe it
   record = protein.nardini.gin_cluster(seed=42, num_scrambles=10000)
   print(record["description"], record["is_typical"])

   # stack the whole set into one matrix for clustering or a model
   matrix = np.vstack([
       p.nardini.vector(seed=42, num_scrambles=10000).values
       for p in proteins.values()
   ])

.. important::
   On Windows -- or any time you pass ``start_method='spawn'`` -- call this from
   inside an ``if __name__ == "__main__":`` block. sparrow defaults to ``fork``
   where it exists, which needs no guard and works in notebooks; see
   :ref:`nardini-proteome`.

Variations
^^^^^^^^^^

* **A non-human proteome:** build your own compositional background rather than
  using the bundled human one, with
  :func:`~sparrow.nardini.background.build_background`, and pass it as
  ``background=``.
* **Your own grammars:** rather than assigning to the 30 published human clusters,
  cluster your own vectors with
  :func:`~sparrow.nardini.clustering.build_gin_reference`.
* **Patterning only:** pass ``background=None`` for the 36 composition-independent
  patterning features, which need no reference set at all.

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
