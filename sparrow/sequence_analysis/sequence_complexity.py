def low_complexity_domains_holt(sequence, residue_selector, minimum_length=10, max_interruption=2):

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

    while idx < seqlen:
        
        if new_seq[idx]  == '1':
            count = count + 1
            idx = idx + 1
            
            
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


                # if we found an big enough LCD... note here idx is the index
                # of the last hit which means we always get a hit....hit style
                # LCD
                if count >= minimum_length:
                     return_lists.append([sequence[start:idx], start, idx])                        

                # reset IDX and count. Note count =1 because we know idx + gap + 1
                # is a hit, so therefore we iterate to the NEXT idx (idx + gap + 2)
                
                count = 1                

                # watch out that we set start 1st...
                start = idx + gap  # we know idx + gap +1 is a hit!
                idx = idx + gap + 1
                

            # if not bigger than threshold
            else:
                
                #count = count + gap + 1
                count = count + 1

                # note we increment with +gap + 2 because we know idx + gap + 1 is a hit
                idx = idx + gap + 1

    if count >= minimum_length:
        return_lists.append([sequence[start:idx], start, idx])                        

    return return_lists
        
