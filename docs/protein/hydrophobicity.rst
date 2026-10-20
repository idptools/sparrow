Hydrophobicity
==============

How hydrophobic a sequence is on average, and how its hydrophobic residues are
arranged.

.. code-block:: python

   from sparrow import Protein
   from sparrow.data.amino_acids import AA_hydro_KD_normalized

   p = Protein("MEEEKKKKSSSTTTDDDLLLVVVAAAFFF")

   p.hydrophobicity                              # mean Kyte-Doolittle hydrophobicity
   p.SHD                                         # sequence hydropathy decoration, HPS scale
   p.compute_SHD_custom(AA_hydro_KD_normalized)  # SHD on any scale

``hydrophobicity`` is the **mean** over the sequence, on the Kyte-Doolittle
scale (:data:`~sparrow.data.amino_acids.AA_hydro_KD`). For a per-residue track,
use ``p.linear_sequence_profile(mode="hydrophobicity")``, or
``p.linear_property_profile`` with any published hydrophobicity scale; see
:doc:`profiles`.

``SHD`` (Zheng *et al.*, 2020) is the hydropathy counterpart of SCD: a sum over
every pair of residues of their summed hydropathy, divided by their separation,
so it is larger when hydrophobic residues are clustered together.
It uses the HPS model's hydropathy scale
(:data:`~sparrow.data.amino_acids.AA_hydro_HPS`), which SHD was defined with;
``compute_SHD_custom`` takes any scale that covers the residues in the
sequence.

Reference
---------

.. autosummary::
   :nosignatures:

   sparrow.protein.Protein.hydrophobicity
   sparrow.protein.Protein.SHD
   sparrow.protein.Protein.compute_SHD_custom

.. autoproperty:: sparrow.protein.Protein.hydrophobicity

.. autoproperty:: sparrow.protein.Protein.SHD

.. automethod:: sparrow.protein.Protein.compute_SHD_custom
