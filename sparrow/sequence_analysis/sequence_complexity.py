"""Low-complexity domains (LCDs) enriched in a chosen set of residues.

An LCD here is a region rich in residues from a chosen set
(``residue_selector``, such as ``'Q'`` or ``'ED'``), the chosen residues. A
region is a candidate if it starts and ends at a chosen residue, and it passes
if it is at least ``minimum_length`` long and the chosen residues make up more
than ``fractional_threshold`` of its length. The domains are found by a search
along the sequence:

1. From the first chosen residue, the candidates are the regions from it to
   each later chosen residue.
2. The longest candidate that passes is a domain, and the search starts again
   from the first chosen residue after it. If no candidate passes, the search
   starts again from the next chosen residue.

So a domain holds at least two chosen residues, and neither a sparse start nor
a domain found earlier stops the search finding one further on. The approach
follows Gutierrez et al. from the Holt lab (hence the names): Gutierrez JI, et
al. SWI/SNF senses carbon starvation with a pH-sensitive low-complexity
sequence. eLife. 2022;11:e70344.

The two variants differ in where the search runs:

* :func:`low_complexity_domains_holt` splits the sequence into stretches at
  every run of more than ``max_interruption`` other residues (Gutierrez et al.
  allowed up to 17; the default here is 5) and searches each stretch on its
  own, so no domain holds such a run.
* :func:`low_complexity_domains_holt_permissive` has no ``max_interruption``
  and searches the whole sequence at once, so a run of other residues of any
  length can sit inside a domain, and ``fractional_threshold`` alone decides
  how far a domain extends.

From a :class:`~sparrow.protein.Protein`,
``Protein.low_complexity_domains(mode='holt', residue_selector=...)`` (or
``mode='holt-permissive'``) calls these, and ``Protein.low_complexity_mask``
gives the same domains as a 1 or 0 for each residue.
"""

import numpy as np

from sparrow.sparrow_exceptions import SparrowException


def low_complexity_domains_holt_permissive(
    sequence, residue_selector, minimum_length=15, max_interruption=None, fractional_threshold=0.25
):
    """
    Low-complexity domains with no limit on interruptions, bounded by the fraction alone.

    This is the "permissive" variant of :func:`low_complexity_domains_holt`. It
    has no ``max_interruption``, so a run of other residues of any length can
    sit inside a domain, and ``fractional_threshold`` alone decides how far a
    domain extends. As in :func:`low_complexity_domains_holt`, a region is a
    candidate if it starts and ends at a residue in ``residue_selector`` (a
    chosen residue), and it passes if it is at least ``minimum_length`` long
    and its chosen residues make up more than ``fractional_threshold`` of its
    length. The same search runs, over the whole sequence at once:

    1. From the first chosen residue, the candidates are the regions that end
       at each later chosen residue.
    2. The longest candidate that passes is a domain, and the search starts
       again from the first chosen residue after it. If no candidate passes,
       the search starts again from the next chosen residue.

    Parameters
    ----------
    sequence : str
        Amino acid sequence

    residue_selector : list or str
        One or more one-letter amino acid codes defining the residues to search
        for (e.g. ``'Q'`` or ``'ED'``)

    minimum_length : int
        Minimum length of a low complexity domain. Default is 15.

    max_interruption : None
        Not used, since this variant has no limit on interruptions: leave it
        out. It is kept in the place :func:`low_complexity_domains_holt` has it
        so that a call giving it, by name or by position, fails rather than
        having the value taken as ``fractional_threshold``.

    fractional_threshold : float
        The fraction (0-1) of a domain's length that its chosen residues must
        exceed. Default is 0.25.

    Returns
    -------
    list
        List of lists. Each list contains the low complexity domain sequence,
        the start index (0-indexed) and the end index (exclusive), such that
        ``sequence[start:end]`` is the domain.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``max_interruption`` is given.

    Examples
    --------
    Eight As split two runs of eight Qs for :func:`low_complexity_domains_holt`,
    but not here, where the whole region is 16/24 = 67% Q:

    >>> sequence = "QQQQQQQQ" + "AAAAAAAA" + "QQQQQQQQ"
    >>> low_complexity_domains_holt(sequence, "Q", minimum_length=5)
    [['QQQQQQQQ', 0, 8], ['QQQQQQQQ', 16, 24]]
    >>> low_complexity_domains_holt_permissive(sequence, "Q", minimum_length=5)
    [['QQQQQQQQAAAAAAAAQQQQQQQQ', 0, 24]]

    A threshold of 0.7 keeps them apart:

    >>> low_complexity_domains_holt_permissive(sequence, "Q", minimum_length=5, fractional_threshold=0.7)
    [['QQQQQQQQ', 0, 8], ['QQQQQQQQ', 16, 24]]
    """
    if max_interruption is not None:
        raise SparrowException(
            f"low_complexity_domains_holt_permissive has no max_interruption (got {max_interruption}): how far a "
            "domain extends is decided by fractional_threshold alone. Leave max_interruption out, or use "
            "low_complexity_domains_holt (mode='holt'), which applies it."
        )

    chosen_positions = _chosen_positions(sequence, residue_selector)
    return _search(sequence, chosen_positions, minimum_length, fractional_threshold)


def low_complexity_domains_holt(
    sequence, residue_selector, minimum_length=15, max_interruption=5, fractional_threshold=0.25
):
    """
    Low-complexity domains, interrupted by runs of at most ``max_interruption`` other residues.

    A run of more than ``max_interruption`` residues not in ``residue_selector``
    splits the sequence into stretches, and each stretch is searched on its own.
    A region is a candidate if it starts and ends at a chosen residue, and it
    passes if it is at least ``minimum_length`` long and its chosen residues
    make up more than ``fractional_threshold`` of its length. In each stretch:

    1. From the stretch's first chosen residue, the candidates are the regions
       that end at each later chosen residue in the stretch.
    2. The longest candidate that passes is a domain, and the search starts
       again from the first chosen residue after it. If no candidate passes,
       the search starts again from the next chosen residue.

    See :func:`low_complexity_domains_holt_permissive` for the variant with no
    limit on interruptions.

    Parameters
    ----------
    sequence : str
        Amino acid sequence

    residue_selector : list or str
        One or more one-letter amino acid codes defining the residues to search
        for (e.g. ``'Q'`` or ``'ED'``)

    minimum_length : int
        Minimum length of a low complexity domain. Default is 15.

    max_interruption : int
        Maximum number of consecutive residues NOT in residue_selector that are
        permitted between two selector residues inside a domain; 0 or more.
        Default is 5 (Gutierrez et al. used 17).

    fractional_threshold : float
        Minimum fraction (0-1) of residues in a low complexity domain that must
        be in the residue_selector list. Note this is a strict threshold: the
        fraction must exceed it. Default is 0.25.

    Returns
    -------
    list
        List of lists. Each list contains the low complexity domain sequence,
        the start index (0-indexed) and the end index (exclusive), such that
        ``sequence[start:end]`` is the domain.

    Raises
    ------
    sparrow.sparrow_exceptions.SparrowException
        If ``max_interruption`` is negative.

    Examples
    --------
    >>> low_complexity_domains_holt("MKKRPKSPSGEEEQQQQQQQQQQQQQQQQASDLEKRK", "Q", minimum_length=10)
    [['QQQQQQQQQQQQQQQQ', 13, 29]]

    A lone Q five residues before a run of six Qs is in the run's stretch. No
    candidate from it is more than 80% Q, so the search moves on to the run:

    >>> low_complexity_domains_holt("QAAAAAQQQQQQ", "Q", minimum_length=5, fractional_threshold=0.8)
    [['QQQQQQ', 6, 12]]
    """
    if max_interruption < 0:
        raise SparrowException(f"max_interruption must be 0 or more, got {max_interruption}.")

    chosen_positions = _chosen_positions(sequence, residue_selector)

    # a break wherever more than max_interruption other residues lie between
    # two chosen residues
    other_residues_between = np.diff(chosen_positions) - 1
    breaks = np.flatnonzero(other_residues_between > max_interruption) + 1

    domains = []
    for stretch in np.split(chosen_positions, breaks):
        domains.extend(_search(sequence, stretch, minimum_length, fractional_threshold))
    return domains


def _chosen_positions(sequence, residue_selector):
    """The position of every residue in ``residue_selector``, in sequence order.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    residue_selector : list or str
        The chosen residues, as one-letter codes.

    Returns
    -------
    numpy.ndarray
        The 0-indexed positions, as integers.
    """
    return np.array([position for position, residue in enumerate(sequence) if residue in residue_selector], dtype=int)


def _search(sequence, chosen_positions, minimum_length, fractional_threshold):
    """The domains the module's search finds among some chosen residues.

    :func:`low_complexity_domains_holt` runs this on each stretch, and
    :func:`low_complexity_domains_holt_permissive` on the whole sequence.

    Parameters
    ----------
    sequence : str
        Amino acid sequence.
    chosen_positions : numpy.ndarray
        The positions of the chosen residues to search among, in sequence
        order.
    minimum_length : int
        The shortest domain, in residues.
    fractional_threshold : float
        The fraction of a domain's length that its chosen residues must exceed.

    Returns
    -------
    list
        Each domain as ``[subsequence, start, end]``, with ``start`` 0-indexed
        and ``end`` exclusive, in sequence order.
    """
    domains = []
    first = 0
    while first < len(chosen_positions):
        last = _furthest_passing_end(chosen_positions, first, minimum_length, fractional_threshold)

        # nothing passes from this chosen residue, so try from the next one
        if last is None:
            first = first + 1
            continue

        start = int(chosen_positions[first])
        end = int(chosen_positions[last]) + 1
        domains.append([sequence[start:end], start, end])

        # the next domain is looked for after this one
        first = last + 1

    return domains


def _furthest_passing_end(chosen_positions, first, minimum_length, fractional_threshold):
    """The furthest chosen residue a domain starting at ``chosen_positions[first]`` can end at.

    Parameters
    ----------
    chosen_positions : numpy.ndarray
        The positions of the chosen residues being searched, in sequence order.
    first : int
        The index, in ``chosen_positions``, of the chosen residue the domain
        starts at.
    minimum_length : int
        The shortest domain, in residues.
    fractional_threshold : float
        The fraction of a domain's length that its chosen residues must exceed.

    Returns
    -------
    int or None
        The index, in ``chosen_positions``, of the chosen residue that ends the
        longest candidate from ``first`` that passes, or None if none does.
    """
    start = chosen_positions[first]

    # each later chosen residue as the candidate's last: the candidate's length,
    # and how many chosen residues it holds (its first, its last and every one
    # between)
    lengths = chosen_positions[first + 1 :] - start + 1
    counts = np.arange(2, len(lengths) + 2)

    passing = np.flatnonzero((lengths >= minimum_length) & (counts / lengths > fractional_threshold))
    if len(passing) == 0:
        return None
    return first + 1 + int(passing[-1])
