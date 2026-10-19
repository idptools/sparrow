"""Low-complexity domains (LCDs) enriched in a chosen set of residues.

An LCD here is a stretch of sequence made of residues from a chosen set
(``residue_selector``, such as ``'Q'`` or ``'ED'``) interrupted by runs of no
more than ``max_interruption`` other residues, which is at least
``minimum_length`` long and in which the chosen residues make up more than
``fractional_threshold`` of its length. A run of more than ``max_interruption``
other residues ends a stretch; within each stretch the longest region that starts
at its first chosen residue, ends at a chosen residue and passes both thresholds
is reported. The approach follows Gutierrez et al. from the Holt lab (hence the
names): Gutierrez JI, et al. SWI/SNF senses carbon starvation with a
pH-sensitive low-complexity sequence. eLife. 2022;11:e70344. They allowed
interruptions of up to 17 residues, where the default here is 5.

The two variants differ only in what counts towards the fraction:
:func:`low_complexity_domains_holt` counts only the chosen residues, while
:func:`low_complexity_domains_holt_permissive` also counts the bridged
interruptions, so every domain's fraction is 1, ``fractional_threshold`` has no
effect below 1, and each stretch at least ``minimum_length`` long is reported
whole.

From a :class:`~sparrow.protein.Protein`,
``Protein.low_complexity_domains(mode='holt', residue_selector=...)`` (or
``mode='holt-permissive'``) calls these, and ``Protein.low_complexity_mask``
gives the same domains as a 1 or 0 for each residue.
"""


def low_complexity_domains_holt_permissive(
    sequence, residue_selector, minimum_length=15, max_interruption=5, fractional_threshold=0.25
):
    """
    Low-complexity domains, counting bridged interruptions towards the fraction.

    This is the "permissive" variant of :func:`low_complexity_domains_holt`: the
    bridged interruption residues also count towards the domain's residue count,
    so every residue of a domain counts and its fraction is always 1. Below 1,
    ``fractional_threshold`` therefore has no effect, and each stretch at least
    ``minimum_length`` long is reported whole.

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
        permitted between two selector residues inside a domain. Default is 5.

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

    Examples
    --------
    One Q in every three residues is 38% Q, so it fails a threshold of 0.5 unless
    the bridged residues count:

    >>> low_complexity_domains_holt_permissive("QAAQAAQAAQAAQAAQ", "Q", minimum_length=10, fractional_threshold=0.5)
    [['QAAQAAQAAQAAQAAQ', 0, 16]]
    >>> low_complexity_domains_holt("QAAQAAQAAQAAQAAQ", "Q", minimum_length=10, fractional_threshold=0.5)
    []
    """
    return __low_complexity_domains_holt_internal(
        sequence=sequence,
        residue_selector=residue_selector,
        minimum_length=minimum_length,
        max_interruption=max_interruption,
        fractional_threshold=fractional_threshold,
        permissive=True,
    )


def low_complexity_domains_holt(
    sequence, residue_selector, minimum_length=15, max_interruption=5, fractional_threshold=0.25
):
    """
    Low-complexity domains, counting only the chosen residues towards the fraction.

    Only residues drawn from ``residue_selector`` count towards a domain's
    residue count. See :func:`low_complexity_domains_holt_permissive` for the
    variant in which bridged interruption residues count too.

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
        permitted between two selector residues inside a domain. Default is 5
        (Gutierrez et al. used 17).

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

    Examples
    --------
    >>> low_complexity_domains_holt("MKKRPKSPSGEEEQQQQQQQQQQQQQQQQASDLEKRK", "Q", minimum_length=10)
    [['QQQQQQQQQQQQQQQQ', 13, 29]]
    """
    return __low_complexity_domains_holt_internal(
        sequence=sequence,
        residue_selector=residue_selector,
        minimum_length=minimum_length,
        max_interruption=max_interruption,
        fractional_threshold=fractional_threshold,
        permissive=False,
    )


def __low_complexity_domains_holt_internal(
    sequence, residue_selector, minimum_length=10, max_interruption=2, fractional_threshold=0.25, permissive=False
):
    """
    Function to identify low complexity domains enriched in a specific residues

    Parameters
    ----------
    sequence : str
        Amino acid sequence

    residue_selector : list or str
        List of residues to search through

    minimum_length : int
        Minimum length of a low complexity domain

    max_interruption : int
        Maximum number of residues that can be between two residues in the
        residue_selector list

    fractional_threshold : float
        Minimum fraction of residues in a low complexity domain that must be
        in the residue_selector list

    permissive : bool
        Controls what counts towards a domain's residue count. If False
        (default), only residues in residue_selector are counted. If True, the
        bridged interruption residues are counted as well, which makes the
        fractional_threshold easier to clear. Note that max_interruption is
        honoured in BOTH modes - it always determines where a domain ends.

    Returns
    -------
    list
        List of lists. Each list contains the low complexity domain sequence,
        the start index (0-indexed) and the end index (exclusive), such that
        ``sequence[start:end]`` is the domain.

    """

    # build binary sequence
    new_seq = ""
    seqlen = len(sequence)
    for i in sequence:
        if i in residue_selector:
            new_seq = new_seq + "1"

        else:
            new_seq = new_seq + "0"

    return_lists = []

    # FIRST ask if first residue is a hit

    idx = new_seq.find("1")

    # exit if no hits
    if idx == -1:
        return return_lists

    # if we get here we know there's at least one residue
    # in the sequence matching the residue selector
    start = idx

    # initialize count
    count = 1

    # increment idx so idx is the "next" position
    idx = idx + 1

    # initialize current region list
    current_region_list = []

    # loop over the rest of the sequence
    while idx < seqlen:
        # if next position is a hit
        if new_seq[idx] == "1":
            count = count + 1
            idx = idx + 1

            # note this means the region is start-idx exclusive (i.e.
            # we incremented idx here, so seq[start:idx] is the region).
            current_region_list.append([count, start, idx])

        # else next position was not a hit
        else:
            # if we're here we know idx-1 was a hit

            #  5 6 7 8 9
            # [0,0,1,1,1]
            # jump to the next hit... Note that we KNOW gap will be -1 (no more hits) OR 1 or
            # greater because we know new_seq[idx] != 0 - i.e. gap cannot == 0
            gap = new_seq[idx:].find("1")

            # print('gap: %i'%(gap))

            # if gap is -1 then we know there are no more hits
            if gap == -1:
                break

            # if the gap is bigger than the tollerated interuption then this
            # position ends the current LCD. Gap must be between 1 and 'big'
            if gap > max_interruption:
                # get the biggest valid region
                tmp = _return_best_hit(current_region_list, minimum_length, fractional_threshold)

                # if we found 1 or more entries that were valid
                if len(tmp) > 0:
                    # last valid entry will be biggest
                    entry = tmp[-1]
                    local_start = entry[1]
                    local_end = entry[2]
                    return_lists.append([sequence[local_start:local_end], local_start, local_end])

                # reset IDX and count. Note count = 1 because we know idx + gap + 1
                # is a hit, so therefore we iterate to the NEXT idx (idx + gap + 2)

                count = 1

                # reset the empty current region list
                current_region_list = []

                # watch out that we set start 1st...
                start = idx + gap  # we know idx + gap +1 is a hit!
                idx = idx + gap + 1

            # if not bigger than max_interuption then we extend the current region
            else:
                # if permissive is True then we increment count by both 1 and
                # the gap size. If permissive is False then we only increment
                # by 1
                if permissive:
                    count = count + gap + 1
                else:
                    count = count + 1

                # note we increment with +gap + 1 to increment to the
                # next position after the gap
                idx = idx + gap + 1
                current_region_list.append([count, start, idx])

    tmp = _return_best_hit(current_region_list, minimum_length, fractional_threshold)

    # if we found 1 or more entries that were valid
    if len(tmp) > 0:
        # last valid entry will be biggest
        entry = tmp[-1]
        local_start = entry[1]
        local_end = entry[2]
        return_lists.append([sequence[local_start:local_end], local_start, local_end])

    return return_lists


def _return_best_hit(current_region_list, minimum_length, fractional_threshold):
    """
    Function to return the best hit from a list of hits

    Parameters
    ----------
    current_region_list : list
        List of lists. Each list contains the low complexity domain sequence,
        the start index, and the end index

    minimum_length : int
        Minimum length of a low complexity domain

    fractional_threshold : float
        Minimum fraction of residues in a low complexity domain that must be
        in the residue_selector list

    Returns
    -------
    list
        List of lists. Each list contains the low complexity domain sequence,
        the start index, and the end index

    """

    tmp = []

    for entry in current_region_list:
        region_size = entry[2] - entry[1]
        region_count = entry[0]

        # if this entry is big enough...
        if region_size >= minimum_length:
            # if entry has a high enough fraction...
            if region_count / region_size > fractional_threshold:
                tmp.append(entry)

    return tmp
