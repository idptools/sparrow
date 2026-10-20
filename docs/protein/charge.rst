Charge and Charge Patterning
============================

How much charge a sequence carries, of which sign, and how it is arranged.

.. code-block:: python

   from sparrow import Protein

   p = Protein("MEEEKKKKSSSTTTDDDQQQQNNNN")

   p.FCR, p.NCPR                             # fraction of charged residues, net charge per residue
   p.fraction_positive, p.fraction_negative  # K and R; D and E

   p.kappa                                   # charge segregation (0-1, or -1)
   p.compute_kappa_x("ED", "KR", window_size=6)
   p.SCD                                     # sequence charge decoration
   p.compute_SCD_x(["E", "D"], ["R", "K"])   # SCD between any two residue groups

   p.isoelectric_point                       # pI, from side-chain pKa values
   p.charge_at_pH(5.5)                       # mean titratable charge at any pH

Formal charge and titratable charge
-----------------------------------

``FCR`` and ``NCPR`` count *formal* charges - D and E negative, K and R
positive - and assume full ionization. ``charge_at_pH`` instead applies
Henderson-Hasselbalch with side-chain pKa values
(:data:`~sparrow.data.amino_acids.AA_PKA`: the EMBOSS values, except lysine at
10.0 rather than 10.8), so it also sees H, Y and C, and it is the quantity
``isoelectric_point`` bisects to zero. Termini are not included in either,
matching localCIDER, whose pKa values these are.

Patterning
----------

``kappa`` (Das & Pappu, 2013) measures how segregated the positive and negative
residues are, from 0 (well mixed) to 1 (fully segregated), using windows of 5
and 6 residues. ``compute_kappa_x`` generalizes it to any two residue groups,
or to one group against everything else, at any window size. Both return
``-1`` where kappa is undefined: a sequence shorter than the window, or one
missing a group.

``SCD`` (Sawle & Ghosh, 2015) is a sum over every pair of charged residues,
weighted by the square root of their separation, so it grows more negative as
opposite charges are segregated into blocks. ``compute_SCD_x`` computes it for
any two residue groups.

How these statistics are computed, and how they relate to NARDINI's patterning
features, is on :doc:`../power_user/patterning`.

Reference
---------

.. autosummary::
   :nosignatures:

   sparrow.protein.Protein.FCR
   sparrow.protein.Protein.NCPR
   sparrow.protein.Protein.fraction_positive
   sparrow.protein.Protein.fraction_negative
   sparrow.protein.Protein.kappa
   sparrow.protein.Protein.compute_kappa_x
   sparrow.protein.Protein.SCD
   sparrow.protein.Protein.compute_SCD_x
   sparrow.protein.Protein.isoelectric_point
   sparrow.protein.Protein.charge_at_pH

.. autoproperty:: sparrow.protein.Protein.FCR

.. autoproperty:: sparrow.protein.Protein.NCPR

.. autoproperty:: sparrow.protein.Protein.fraction_positive

.. autoproperty:: sparrow.protein.Protein.fraction_negative

.. autoproperty:: sparrow.protein.Protein.kappa

.. automethod:: sparrow.protein.Protein.compute_kappa_x

.. autoproperty:: sparrow.protein.Protein.SCD

.. automethod:: sparrow.protein.Protein.compute_SCD_x

.. autoproperty:: sparrow.protein.Protein.isoelectric_point

.. automethod:: sparrow.protein.Protein.charge_at_pH
