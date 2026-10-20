Conformational Propensity
=========================

Compositional summaries that speak to how expanded a disordered chain is likely
to be.

.. code-block:: python

   from sparrow import Protein

   p = Protein("MEEEKKKKSSSTTTDDDPPPRRRGGG")

   p.fraction_chain_expanding      # fraction that is E, D, R, K or P
   p.fraction_disorder_promoting   # fraction in the disorder-promoting set
   p.PPII_propensity               # mean polyproline II propensity (Hilser scale)

   p.RK_ratio                      # log10 (R+1)/(K+1); positive means R-rich
   p.ED_ratio                      # log10 (E+1)/(D+1); positive means E-rich

Chain-expanding residues drive a disordered chain towards more expanded
conformations, through charge repulsion and backbone stiffening. The
disorder-promoting set is T, A, G, R, D, H, Q, K, S, E and P
(:data:`~sparrow.data.amino_acids.DISORDER_PROMOTING`). Polyproline II is an
extended left-handed helix that disordered regions sample readily, and a high
mean propensity (:data:`~sparrow.data.amino_acids.AA_PPII_HILSER`) goes with a
more expanded ensemble.

``RK_ratio`` and ``ED_ratio`` split residue pairs that share a formal charge but
behave differently - arginine supports cation-pi and stacking interactions that
lysine does not - so they say something net charge cannot. The ``+1``
pseudo-count keeps them finite when a residue is absent.

For the ALBATROSS predictions of a chain's dimensions themselves, see
:doc:`predictor` and :doc:`polymeric`.

Reference
---------

.. autosummary::
   :nosignatures:

   sparrow.protein.Protein.fraction_chain_expanding
   sparrow.protein.Protein.fraction_disorder_promoting
   sparrow.protein.Protein.PPII_propensity
   sparrow.protein.Protein.RK_ratio
   sparrow.protein.Protein.ED_ratio

.. autoproperty:: sparrow.protein.Protein.fraction_chain_expanding

.. autoproperty:: sparrow.protein.Protein.fraction_disorder_promoting

.. autoproperty:: sparrow.protein.Protein.PPII_propensity

.. autoproperty:: sparrow.protein.Protein.RK_ratio

.. autoproperty:: sparrow.protein.Protein.ED_ratio
