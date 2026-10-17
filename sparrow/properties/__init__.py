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
"""
