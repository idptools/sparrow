``sparrow.tools``
=================

.. py:module:: sparrow.tools

Reading and fetching sequences, and the small utilities the rest of sparrow
uses. :doc:`../power_user/utilities` shows the first in use.

``sparrow.tools.io``
--------------------

.. automodule:: sparrow.tools.io
   :members: uniprot_fetch

``read_fasta`` is :func:`sparrow.read_fasta`, documented in :doc:`sparrow`.

``sparrow.tools.general_tools``
-------------------------------

.. automodule:: sparrow.tools.general_tools

``sparrow.tools.utilities``
---------------------------

The helpers ``Protein`` and its accessors use to cache values, hand back
copies, check options and warn about renamed functions.

.. automodule:: sparrow.tools.utilities
   :members: cache_key, copy_of, validate_keyword_option, warn_renamed
