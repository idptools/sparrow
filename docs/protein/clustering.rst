Complexity, Clustering and Patches
==================================

How varied a sequence's composition is, and how closely its residues of a given
kind cluster together.

.. code-block:: python

   from sparrow import Protein

   p = Protein("MEEEKKKKSSSTTTDDDQQQQNNNNGRGRGGYGGRGG")

   p.complexity                                 # Wootton-Federhen (SEG) complexity

   p.compute_iwd(["D", "E"])                    # clustering of acidic residues
   p.compute_iwd("ILVAM")                       # a string works too
   p.compute_iwd_charged_weighted("-")          # negative residues, weighted by local charge
   p.compute_bivariate_iwd_charged_weighted()   # how positive and negative residues mix

   p.compute_patch_fraction("Q")                # fraction of the sequence in Q-rich patches

   # NARDINI+'s 'RG Frac': R/G patches holding at least two adjacent RG pairs
   p.compute_patch_fraction("RG", min_target_count=None,
                            adjacent_pair_pattern="RG", min_adjacent_pair_count=2)

Complexity
----------

``complexity`` is the compositional complexity of the SEG algorithm (Wootton &
Federhen): a Shannon-style measure that is higher the more varied the
composition. It is 0 for a homopolymer.

Clustering (IWD)
----------------

The inverse weighted distance is built from the inverse of the separations
between the target residues, so it is high when they sit close together and low
when they are spread out. ``compute_iwd`` takes any residues. The charge-weighted
versions use the local net charge per residue, over a window of 8:
``compute_iwd_charged_weighted`` weights each residue of one sign by the
magnitude of the local NCPR at its position, and a residue whose local NCPR has
the opposite sign contributes nothing; ``compute_bivariate_iwd_charged_weighted``
weights each pair of oppositely charged residues by the difference in their
local NCPR, to measure how intermixed the two signs are.

Patches
-------

A patch is a stretch rich in target residues, bridging up to ``interruption``
other residues (2 by default) and holding at least ``min_target_count`` target
residues (4 by default). ``compute_patch_fraction`` is the fraction of the
sequence that patches cover. ``adjacent_pair_pattern`` and
``min_adjacent_pair_count`` further require a motif inside each patch, which is
how NARDINI+ defines its ``'RG Frac'`` feature (as above).

For the low-complexity *domains* that a residue set defines, see
:doc:`annotations`.

Reference
---------

.. autosummary::
   :nosignatures:

   sparrow.protein.Protein.complexity
   sparrow.protein.Protein.compute_iwd
   sparrow.protein.Protein.compute_iwd_charged_weighted
   sparrow.protein.Protein.compute_bivariate_iwd_charged_weighted
   sparrow.protein.Protein.compute_patch_fraction

.. autoproperty:: sparrow.protein.Protein.complexity

.. automethod:: sparrow.protein.Protein.compute_iwd

.. automethod:: sparrow.protein.Protein.compute_iwd_charged_weighted

.. automethod:: sparrow.protein.Protein.compute_bivariate_iwd_charged_weighted

.. automethod:: sparrow.protein.Protein.compute_patch_fraction
