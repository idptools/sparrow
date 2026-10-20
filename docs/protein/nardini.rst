.. _nardini-protein:

Molecular Grammar (``protein.nardini``)
=======================================

`NARDINI <https://www.sciencedirect.com/science/article/pii/S0022283621006100>`_
asks whether the residues of an intrinsically disordered region are arranged
non-randomly. It coarse-grains the amino acids into a handful of biophysical
types and, for each type and each pair of types, compares the sequence against
scrambles of itself; since a scramble keeps the composition, the result
isolates *patterning* from *composition*. NARDINI+ adds a block of
compositional features, scored against a reference set of IDRs, and the GIN
resource groups the human IDRome into 30 clusters of shared molecular grammar.

``protein.nardini`` gives all of this for one sequence:

.. code-block:: python

   from sparrow import Protein

   p = Protein("MENDPRVRKFASDGAQWAIKWQKKSSGSPRNTTASSGEKSSDRPRSTSSGRRSSASS")

   v = p.nardini.vector(seed=42)        # the 90-feature NARDINI+ vector
   p.nardini.patterning(seed=42)        # just the 36 patterning z-scores
   p.nardini.composition()              # just the 54 compositional z-scores
   p.nardini.gin_cluster(seed=42)       # the nearest human GIN grammar
   print(p.nardini.gin_report(seed=42)) # the assignment, and how much to trust it

   v["Frac K+R"]                        # one feature, by name
   v.values                             # the underlying array
   v.to_dict()

Results are labelled NumPy containers
(:class:`~sparrow.nardini.table.FeatureVector`), not DataFrames. Each is cached
under its options, so asking again with the same ones is free.

Seeds and scrambles
-------------------

The patterning z-scores come from scrambles, so pass a ``seed`` to make them
reproducible; without one, they differ slightly each time they are computed
(though a ``Protein`` still caches what it computed). ``num_scrambles`` sets how
many scrambles are drawn (10,000 by default; the published protocol uses
100,000).

Background
----------

Compositional z-scores need a reference set of IDRs, and for one protein there
is nothing to build one from, so they default to the **published human IDRome
background** bundled with sparrow - which is also the space the GIN centroids
live in. Pass ``background="yeast"`` (or another of the bundled species), a
saved ``.npz`` path, or a
:class:`~sparrow.nardini.background.NardiniBackground` you have built. The
patterning half never uses it, so it is meaningful for any sequence.

Trusting a GIN label
--------------------

GIN labels only mean something for sequences that resemble human IDRs, and
nearest-centroid assignment never abstains. Check ``is_typical`` in the record
``gin_cluster`` returns, or read ``gin_report()``, before trusting one.

For many proteins at once, :func:`sparrow.batch` and
:func:`~sparrow.nardini.batch.nardini_proteins` compute the vectors in parallel
and attach them to the ``Protein`` objects, after which the calls above cost
nothing. :doc:`../power_user/nardini` covers NARDINI+ in full: building
backgrounds, the patterning backends, GIN assignment and its diagnostics,
clustering your own grammars, and running over a proteome.

Reference
---------

.. autoproperty:: sparrow.protein.Protein.nardini

.. autoclass:: sparrow.nardini.protein_interface.ProteinNardini
   :no-members:
   :no-inherited-members:
   :no-undoc-members:

.. autosummary::
   :nosignatures:

   sparrow.nardini.protein_interface.ProteinNardini.vector
   sparrow.nardini.protein_interface.ProteinNardini.patterning
   sparrow.nardini.protein_interface.ProteinNardini.composition
   sparrow.nardini.protein_interface.ProteinNardini.gin_cluster
   sparrow.nardini.protein_interface.ProteinNardini.gin_report

.. automethod:: sparrow.nardini.protein_interface.ProteinNardini.vector

.. automethod:: sparrow.nardini.protein_interface.ProteinNardini.patterning

.. automethod:: sparrow.nardini.protein_interface.ProteinNardini.composition

.. automethod:: sparrow.nardini.protein_interface.ProteinNardini.gin_cluster

.. automethod:: sparrow.nardini.protein_interface.ProteinNardini.gin_report
