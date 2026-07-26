Amino-Acid Property Profiles (AAindex)
======================================

sparrow ships the `AAindex1 <https://www.genome.jp/aaindex/>`_ database: 500+
published indices that each assign a numerical value to every amino acid
(hydropathy, volume, secondary-structure propensity, flexibility, and many
more). :meth:`sparrow.protein.Protein.linear_property_profile` maps a sequence
onto any of these indices and returns a smoothed, windowed profile -- the
property-based analogue of :meth:`~sparrow.protein.Protein.linear_sequence_profile`.

Quick start
-----------

.. code-block:: python

   from sparrow import Protein

   p = Protein("MEEEKKKKSSSTTTDDDQQQQNNNNGGGGSSSS")

   # window-averaged Kyte-Doolittle hydropathy along the sequence
   hydropathy = p.linear_property_profile("hydropathy-kyte-1982", window_size=9)

   # any index works; e.g. a volume or flexibility scale
   volume = p.linear_property_profile("KYTJ820101")  # accession also accepted

``linear_property_profile`` takes the **same arguments** as
``linear_sequence_profile`` (``window_size``, ``end_mode``, ``smooth``); only
``mode`` differs -- here it selects an AAindex property.

Choosing an index (identifiers)
-------------------------------

Each index is addressable two ways:

* a readable **slug** of the form ``<meaning>-<first-author>-<year>``, where
  ``<meaning>`` is the first informative word of the index description
  (for example ``hydropathy-kyte-1982``); and
* the canonical **AAindex accession** (for example ``KYTJ820101``).

When several indices share the same meaning, author, and year, a ``-N``
increment is appended to every member of that group (ordered by accession), so
identifiers are always unique -- for example ``positional-aurora-1998-1``
through ``positional-aurora-1998-20``.

Discovering indices programmatically:

.. code-block:: python

   from sparrow.data import aaindex

   # list every (identifier, accession, description)
   all_indices = aaindex.list_property_indices()

   # metadata for one index
   aaindex.get_property_metadata("hydropathy-kyte-1982")

   # the raw amino-acid -> value mapping
   aaindex.get_property_values("hydropathy-kyte-1982")

Notes
-----

* A few indices have no value for one or more residues in the source database.
  Requesting such an index for a sequence that contains an affected residue
  raises a ``ProteinException`` rather than returning a misleading number.
* The profile is the mean index value within each ``window_size`` window;
  ``end_mode`` and ``smooth`` behave exactly as in ``linear_sequence_profile``.

Module reference
----------------

.. automodule:: sparrow.data.aaindex
   :members:
   :no-index:

Property Index Reference
------------------------

Every index sparrow ships is listed below -- all **569** of them. The table is
regenerated from the bundled database on every documentation build, so it cannot
fall out of step with what is actually installed.

* **Identifier** -- the readable slug. Pass this (or the accession) as the ``mode``
  argument to :meth:`~sparrow.protein.Protein.linear_property_profile`.
* **Accession** -- the canonical AAindex1 accession, if you want to cross-reference
  against the `AAindex database <https://www.genome.jp/aaindex/>`_ directly.
* **Description** -- the original AAindex summary of what the score measures.
* **Source** -- the publication the scale comes from, linked to PubMed where the
  database records a PMID (453 of the 569 do). Worth checking: many descriptions
  are terse, and several scales with near-identical descriptions differ in how they
  were derived.

The table is long, so use your browser's find (:kbd:`Ctrl+F` / :kbd:`Cmd+F`) to
search it, or filter programmatically:

.. code-block:: python

   from sparrow.data import aaindex

   # every (identifier, accession, description)
   everything = aaindex.list_property_indices()

   # e.g. all the hydrophobicity-related scales
   hydro = [row for row in everything if "hydroph" in row[2].lower()]

   # full metadata, including authors, title, journal and PMID
   aaindex.get_property_metadata("hydropathy-kyte-1982")

.. include:: _property_index_table.rst
