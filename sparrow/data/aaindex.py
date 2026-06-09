"""Access to the AAindex1 amino-acid property database.

AAindex1 is a collection of 500+ published indices that each map the 20 standard
amino acids to a numerical value (hydropathy, volume, secondary-structure
propensity, etc.). The raw database ships with sparrow as
``sparrow/data/properties/aaindex1.json``; this module loads it and exposes each
index under a readable, unique identifier.

Identifier scheme
-----------------
Every index is addressable by a slug of the form::

    <meaning>-<first-author>-<year>

where ``<meaning>`` is the first informative word of the index description (for
example ``hydropathy``). When several indices share the same meaning, author,
and year, a ``-N`` increment is appended to every member of that group (ordered
by accession), so identifiers are always unique, e.g. ``positional-aurora-1998-1``
through ``positional-aurora-1998-20``. The canonical AAindex accession (e.g.
``KYTJ820101``) is also always accepted.

Use :func:`list_property_indices` to enumerate all available identifiers.
"""

import functools
import json
import re

from sparrow.sparrow_exceptions import SparrowException

__all__ = [
    "list_property_indices",
    "resolve_identifier",
    "get_property_values",
    "get_property_metadata",
]

# the 20 standard amino acids every AAindex entry provides a value for
VALID_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# generic words skipped when picking the single "meaning" word of a description
_STOP_WORDS = {
    "index", "indices", "normalized", "frequency", "frequencies", "of", "the",
    "a", "an", "for", "in", "on", "to", "and", "or", "value", "values", "scale",
    "propensity", "propensities", "parameter", "parameters", "average",
    "relative", "number", "mean", "content", "residue", "residues", "amino",
    "acid", "acids", "protein", "proteins", "side", "chain", "based", "using",
    "derived", "from", "with", "at", "by", "information", "measure", "measures",
    "net",
}

_DATA_RELATIVE_PATH = "properties/aaindex1.json"


@functools.lru_cache(maxsize=1)
def _load_raw():
    """Load and cache the raw AAindex1 JSON (keyed by accession)."""
    # imported lazily so importing this module does not trigger the full
    # sparrow package import chain
    import sparrow

    with open(sparrow.get_data(_DATA_RELATIVE_PATH)) as handle:
        return json.load(handle)


def _meaning_word(description):
    """Return the first informative word of an index description."""
    # drop the trailing "(Author et al., year)" citation
    base = re.sub(r"\([^()]*\)\s*$", "", description).strip()
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]*", base)
    for token in tokens:
        if token.lower() not in _STOP_WORDS and len(token) > 1:
            return token.lower()
    return tokens[0].lower() if tokens else "property"


def _first_author(authors):
    """Return the lower-cased surname of the first author."""
    surname = (authors or "").split(",")[0].strip()
    surname = re.sub(r"[^A-Za-z]", "", surname).lower()
    return surname or "unknown"


def _year(description, accession):
    """Return a 4-digit year string, preferring the description citation."""
    match = re.search(r"\b(1[89]\d{2}|20\d{2})\b", description or "")
    if match:
        return match.group(1)
    # fall back to the two-digit year embedded in the accession
    year_digits = accession[4:6]
    if year_digits.isdigit():
        return ("20" if int(year_digits) < 30 else "19") + year_digits
    return "na"


@functools.lru_cache(maxsize=1)
def _registry():
    """Build and cache the identifier <-> accession mappings.

    Returns
    -------
    tuple[dict, dict]
        ``(slug_to_accession, accession_to_slug)``.
    """
    raw = _load_raw()

    # group accessions by their base slug (meaning-author-year)
    base_groups = {}
    for accession in sorted(raw):
        entry = raw[accession]
        slug = "{meaning}-{author}-{year}".format(
            meaning=_meaning_word(entry.get("description", "")),
            author=_first_author(entry.get("authors", "")),
            year=_year(entry.get("description", ""), accession),
        )
        base_groups.setdefault(slug, []).append(accession)

    slug_to_accession = {}
    accession_to_slug = {}
    for slug, accessions in base_groups.items():
        if len(accessions) == 1:
            identifier = slug
            slug_to_accession[identifier] = accessions[0]
            accession_to_slug[accessions[0]] = identifier
        else:
            # numbered -1..-N for every member of a colliding group
            for number, accession in enumerate(accessions, start=1):
                identifier = f"{slug}-{number}"
                slug_to_accession[identifier] = accession
                accession_to_slug[accession] = identifier

    return slug_to_accession, accession_to_slug


def resolve_identifier(identifier):
    """Resolve a property identifier or accession to its AAindex accession.

    Parameters
    ----------
    identifier : str
        A ``<meaning>-<author>-<year>[-N]`` slug or a raw AAindex accession.

    Returns
    -------
    str
        The canonical AAindex accession.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If the identifier matches neither a known slug nor an accession.
    """
    slug_to_accession, _ = _registry()
    raw = _load_raw()

    if identifier in slug_to_accession:
        return slug_to_accession[identifier]
    if identifier in raw:
        return identifier
    if isinstance(identifier, str) and identifier.upper() in raw:
        return identifier.upper()

    raise SparrowException(
        f"Unknown property index '{identifier}'. Use a slug like "
        "'hydropathy-kyte-1982' or an AAindex accession like 'KYTJ820101'. "
        "Call sparrow.data.aaindex.list_property_indices() to see all options."
    )


def get_property_values(identifier):
    """Return the amino-acid -> value mapping for a property index.

    Parameters
    ----------
    identifier : str
        Property slug or AAindex accession.

    Returns
    -------
    dict[str, float or None]
        Mapping from each of the 20 standard amino acids to its value. Some
        indices have ``None`` for one or more residues (missing in the source
        database).
    """
    accession = resolve_identifier(identifier)
    return dict(_load_raw()[accession]["values"])


def get_property_metadata(identifier):
    """Return descriptive metadata for a property index.

    Parameters
    ----------
    identifier : str
        Property slug or AAindex accession.

    Returns
    -------
    dict
        Keys: ``identifier`` (canonical slug), ``accession``, ``description``,
        ``authors``, ``reference``, ``title``, ``journal``.
    """
    accession = resolve_identifier(identifier)
    _, accession_to_slug = _registry()
    entry = _load_raw()[accession]
    return {
        "identifier": accession_to_slug[accession],
        "accession": accession,
        "description": entry.get("description", ""),
        "authors": entry.get("authors", ""),
        "reference": entry.get("reference", ""),
        "title": entry.get("title", ""),
        "journal": entry.get("journal", ""),
    }


def list_property_indices():
    """List every available property index.

    Returns
    -------
    list[tuple[str, str, str]]
        Sorted list of ``(identifier, accession, description)`` tuples.
    """
    _, accession_to_slug = _registry()
    raw = _load_raw()
    return [
        (accession_to_slug[accession], accession, raw[accession].get("description", ""))
        for accession in sorted(raw)
    ]
