Sequence and Composition
========================

What a sequence is made of: the sequence itself, its molecular weight, the
fraction of each amino acid, and the fraction of any set of residues.

.. code-block:: python

   from sparrow import Protein

   p = Protein("MGSQSSRSSSQQQQQQQ")

   p.sequence                              # 'MGSQSSRSSSQQQQQQQ'
   len(p)                                  # 17
   p.molecular_weight                      # in Daltons
   p.amino_acid_fractions["Q"]             # fraction of each of the 20 amino acids
   p.compute_residue_fractions(["S", "T"]) # fraction that is S or T

   p.fraction_aromatic                     # F, W, Y
   p.fraction_aliphatic                    # A, I, L, M, V
   p.fraction_polar                        # G, H, N, Q, S, T
   p.fraction_proline

The residue sets are constants in :mod:`sparrow.data.amino_acids`
(:data:`~sparrow.data.amino_acids.ARO`, :data:`~sparrow.data.amino_acids.ALI`,
:data:`~sparrow.data.amino_acids.POLAR`). NARDINI+ counts cysteine as polar as
well; see :ref:`protein-shared-with-nardini`. The fractions of charged residues
are on :doc:`charge`, and those of chain-expanding and disorder-promoting
residues on :doc:`conformational`.

For a whole list of sequences at once, these come from
:mod:`sparrow.properties.composition` (:doc:`../power_user/properties`), or by
name from :func:`sparrow.properties.compute`.

Reference
---------

.. autosummary::
   :nosignatures:

   sparrow.protein.Protein.sequence
   sparrow.protein.Protein.molecular_weight
   sparrow.protein.Protein.amino_acid_fractions
   sparrow.protein.Protein.compute_residue_fractions
   sparrow.protein.Protein.fraction_aromatic
   sparrow.protein.Protein.fraction_aliphatic
   sparrow.protein.Protein.fraction_polar
   sparrow.protein.Protein.fraction_proline

.. autoproperty:: sparrow.protein.Protein.sequence

.. autoproperty:: sparrow.protein.Protein.molecular_weight

.. autoproperty:: sparrow.protein.Protein.amino_acid_fractions

.. automethod:: sparrow.protein.Protein.compute_residue_fractions

.. autoproperty:: sparrow.protein.Protein.fraction_aromatic

.. autoproperty:: sparrow.protein.Protein.fraction_aliphatic

.. autoproperty:: sparrow.protein.Protein.fraction_polar

.. autoproperty:: sparrow.protein.Protein.fraction_proline
