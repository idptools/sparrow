Molecular Grammar (``nardini.``)
================================

The methods of ``protein.nardini`` (:doc:`../protein/nardini`). The four that
scramble - ``vector``, ``patterning``, ``gin_cluster`` and ``gin_report`` - are
computed for the whole batch at once, in ``n_processes`` processes, once for
each set of options; the GIN methods share the scrambles of the vector with the
same options. Pass a ``seed`` to make the patterning z-scores reproducible.

.. include:: _analyses_nardini.rst
