from sparrow.data import amino_acids



def low_complexity_domains_holt_permissive(sequence, residue_selector, minimum_length=15, max_interruption=5, fractional_threshold=0.25):
    """
    Function to identify low complexity domains enriched in a specific residues

    """
    return __low_complexity_domains_holt_internal(sequence=sequence, 
                                                  residue_selector=residue_selector, 
                                                  minimum_length=minimum_length, 
                                                  max_interruption=max_interruption, 
                                                  permissive=True)


def low_complexity_domains_holt(sequence, residue_selector, minimum_length=15, max_interruption=5, fractional_threshold=0.25):
    return __low_complexity_domains_holt_internal(sequence=sequence, 
                                                  residue_selector=residue_selector, 
                                                  minimum_length=minimum_length, 
                                                  max_interruption=max_interruption, 
                                                  fractional_threshold=fractional_threshold, 
                                                  permissive=False)

def __low_complexity_domains_holt_internal(sequence, residue_selector, minimum_length=10, max_interruption=2, fractional_threshold=0.25, permissive=False):

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
    start = idx
    count = 1

    idx = idx + 1

    current_region_list = []
    while idx < seqlen:
        
        if new_seq[idx]  == '1':
            count = count + 1
            idx = idx + 1
            current_region_list.append([count, start, idx])
            
            
        # else next position was not a hit
        else:

            # if we're here we know idx-1 was a hit

            # 5  6 7 8 9
            # [0,0,1,1,1]
            # jump to the next hit... Note that we KNOW gap will be 1 or
            # greater because we know new_seq[idx] != 0
            gap = new_seq[idx:].find('1')
            
            #print('gap: %i'%(gap))

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


                # reset IDX and count. Note count =1 because we know idx + gap + 1
                # is a hit, so therefore we iterate to the NEXT idx (idx + gap + 2)
                
                count = 1     

                # reset the empty current region list
                current_region_list = []

                # watch out that we set start 1st...
                start = idx + gap  # we know idx + gap +1 is a hit!
                idx = idx + gap + 1
                

            # if not bigger than threshold
            else:
                
                if permissive:
                    count = count + gap + 1
                else:
                    count = count + 1

                # note we increment with +gap + 2 because we know idx + gap + 1 is a hit
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
    tmp = []


    for entry in current_region_list:

        # if this entry is big enough...
        if entry[0] >= minimum_length:

            # if entry has a high enough fraction...
            if entry[0]/(entry[2]-entry[1]) > fractional_threshold:
                tmp.append(entry)

    return tmp
