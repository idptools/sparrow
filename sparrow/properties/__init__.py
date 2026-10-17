"""Sequence properties, computed in batches.

This package is the one place in sparrow where sequence-derived properties are
actually calculated. :class:`~sparrow.protein.Protein` and :mod:`sparrow.nardini`
both call into it rather than carrying arithmetic of their own, so a property
and its NARDINI+ counterpart cannot drift apart.

Every function here works on a batch: it takes many sequences (or many
arrangements of one sequence) and returns an array. A single sequence is simply
a batch of one, which is how ``Protein`` uses it.

Modules
-------
:mod:`~sparrow.properties.encoding`
    Validation, and turning sequences into integer arrays.
:mod:`~sparrow.properties.composition`
    Everything that depends only on composition: fractions, residue sets,
    ratios, mean scales, molecular weight, SEG complexity.
:mod:`~sparrow.properties.charge`
    pH-dependent charge and the isoelectric point.
:mod:`~sparrow.properties.patches`
    Patch fractions.
:mod:`~sparrow.properties.windows`
    Sliding-window sums and counts.
:mod:`~sparrow.properties.patterning`
    How residues are arranged: kappa, delta, the patterning percentile, IWD,
    SCD and SHD.
:mod:`~sparrow.properties.profiles`
    Linear profiles: any of these properties along a sequence, one window at
    a time.
"""
