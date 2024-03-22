def low_complexity_domains_holt_permissive(sequence, residue_selector, minimum_length=15, max_interruption=5, fractional_threshold=0.25):
    """
    Function to identify low complexity domains enriched in a specific residues.

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

    Returns
    -------
    list
        List of lists. Each list contains the low complexity domain sequence,
        the start index, and the end index

    """
    return __low_complexity_domains_holt_internal(sequence=sequence, 
                                                  residue_selector=residue_selector, 
                                                  minimum_length=minimum_length, 
                                                  max_interruption=max_interruption, 
                                                  permissive=True)


def low_complexity_domains_holt(sequence, residue_selector, minimum_length=15, max_interruption=5, fractional_threshold=0.25):

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

    fractional_threshold : float
        Minimum fraction of residues in a low complexity domain that must be

    Returns
    -------
    list
        List of lists. Each list contains the low complexity domain sequence,
        the start index, and the end index

    """
    return __low_complexity_domains_holt_internal(sequence=sequence, 
                                                  residue_selector=residue_selector, 
                                                  minimum_length=minimum_length, 
                                                  max_interruption=max_interruption, 
                                                  fractional_threshold=fractional_threshold, 
                                                  permissive=False)

def __low_complexity_domains_holt_internal(sequence, residue_selector, minimum_length=10, max_interruption=2, fractional_threshold=0.25, permissive=False):
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
        If True, then the max_interruption is ignored and the domain is
        extended as far as possible

    Returns
    -------
    list
        List of lists. Each list contains the low complexity domain sequence,
        the start index, and the end index
    
    """

    # build binary sequence
    new_seq = ''
    seqlen = len(sequence)
    for i in sequence:
        if i in residue_selector:
            new_seq = new_seq + '1'
            
        else:
            new_seq = new_seq + '0'


    return_lists = []

    # FIRST ask if first residue is a hit
    
    idx = new_seq.find('1')

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
        if new_seq[idx]  == '1':
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
            gap = new_seq[idx:].find('1')
            
            #print('gap: %i'%(gap))

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
            if region_count/region_size > fractional_threshold:
                tmp.append(entry)

    return tmp

