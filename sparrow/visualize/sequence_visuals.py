from IPython.core.display import display, HTML
from sparrow.data.amino_acids import AA_COLOR
from sparrow.sparrow_exceptions import SparrowException

def show_sequence(seq, 
                  blocksize=10, 
                  newline=50, 
                  fontsize=14, 
                  font_family='Courier', 
                  colors={},
                  header=None,
                  bold_positions=[],
                  bold_residues=[],
                  return_raw_string=False):

                  
    """
    Function that generates an HTML colored string that either renders in the browser or returns the 
    html string. Contains various customizable components.

    Parameters
    -------------

    blocksize : int
        Defines how big blocks of residues are. Blocks are equal to blocksize or the newline parameter, whicever is smaller. 
        Default=10. If set to -1 uses length of the sequence.

    newline : int
        Defines how many residues are shown before a newline is printed. Default is 50. If set to -1 uses the length of
        the sequence.

    fontsize : int
        Fontsize used. Default is 14

    font_family : str
        Which font family (from HTML fonts) is used. Using a non-monospace font makes no sense as columns will be 
        unaligned. Default is Courier. 

    colors : dict
        Dictionary that allows overiding of default color scheme. Should be of format key-value as 'residue'-'color' where 
        residue is a residue in the string and color is a valid HTML color (which can be a Hexcode, standard HTML color name). 
        Note that this also lets you define colors for non-standard amino acids should these be useful. Default is an empty 
        dictionary. Note also that the standard amino acid colorings are defined at sparrow.data.amino_acids.AA_COLOR
        

    header : str
        If provided, this is a string that provides a FASTA-style header (with a leading carrett included). Default None.

    bold_positions : list
        List of positions (indexing from 1 onwards) which will be bolded. Useful for highlighting specific regions. Note that this
        defines individual residues so (for example) to bold residues 10 to 15 would require bold_positions=[10,11,12,13,14,15]. 
        Default is an empty list.

    bold_residues : list
        List of residue types that can be bolded. Useful for highlighting specific residue groups.  Default is an empty list.

    return_raw_string : bool
        If set to true, the function returns the actual raw HTML string, as opposed to an in-notebook rendering. 
        Default is False

    Returns
    ----------
    None or str
        If return_raw_string is set to true then an HTML-compatible string is returned.

    
    Raises
    -------
    sparrow.sparrow_exceptions.SparrowException
        Raises a sparrow exception if invalid input is provided (within reason).

    """
    
    if blocksize > newline:
        newline = blocksize

    if blocksize == -1:
        blocksize = len(seq)
        newline = len(seq)


    if blocksize < 1:
        raise 


    colorString = '<p style="font-family:%s;font-size: %ipx">'%(font_family, fontsize)

    if header:
        colorString = colorString + "><b>%s</b><br>"%(str(header))
        

    count = -1
    for residue in seq:

        count = count + 1

        if count > 0:
            if count % newline == 0:
                colorString = colorString + "<br>"
            
            elif count % blocksize == 0:
                colorString = colorString + " "


        if residue not in AA_COLOR and residue not in colors:
            print('Warning: found invalid amino acid (%s and position %i'%(residue, count+1))
            colorString = colorString + '<span style="color:%s"><b>%s</b></span>' % ('black', residue)
        else:

            # override with user-suppplied pallete if present
            if residue in colors:
                c = colors[residue]

            # else fall back on the standard pallete 
            else:
                c = AA_COLOR[residue]

            # if the residue type OR residue position is to be bolded...
            if residue in bold_residues or (count+1) in bold_positions:
                colorString = colorString + '<span style="color:%s"><b>%s</b></span>' % (c, residue)
            else:
                colorString = colorString + '<span style="color:%s">%s</span>' % (c, residue)
            


    colorString = colorString +"</p>"
            
    if return_raw_string:
        return colorString
    else:
        display(HTML(colorString))
