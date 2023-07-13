import protfasta
import urllib3
from sparrow.protein import Protein
from sparrow.sparrow_exceptions import SparrowException


def uniprot_fetch(uniprot_accession):
    """
    Function that pulls down the amino acid sequence associated with a given uniprot ID. Note this
    actually queries the UniProt website via an HTTP request and SHOULD NOT be used for large scale
    downloading of sequences. You have be warned - UniProt will block your IP if you use this to fetch
    large numbers of sequences. 

    Parameters
    -------------
    uniprot_accession : str
        Valid uniprot accession (note that no validation is performed!).

    Returns
    -----------
    Str or None
        If the accession can be downloaded returns a string with the amino acid sequence. If not
        (for ANY reason) returns None.

    """

    http = urllib3.PoolManager()
    r = http.request('GET', 'https://www.uniprot.org/uniprot/%s.fasta'%(uniprot_accession))
    
    s = "".join(str(r.data).split('\\n')[1:]).replace("'","")


    if s.find('Sorry') > -1:
        return None

    return Protein(s)



def read_fasta(filename, **kwargs):
    """
    Function that reads in a FASTA file using protfasta and returns
    a dictionary of Protein objects. keywords passed as defined by protfasta
    read_fasta(), and not explicitly enumerated but can be found here:

    https://protfasta.readthedocs.io/en/latest/read_fasta.html

    Parameters
    ---------------
    filename : str
        Name of FASTA file

    This will read in the file associated with filename and return a dictionary, where the keys
    are the FASTA file headers and the values are the amino acid sequences associated with each.
    
    Note that as of python 3.7 the order in which one adds items to a dictionary is guaranteed
    to be the order in which they're retrieved, so cycling through the resulting dictionary will
    in fact allow you to cycle through in order. 

    In addition to this simple usage, there are a number of keywords which are described in depth
    below and allow additional processing to be complete. 

    There is an order of options in which sanitization occurs:
    
    1. File is read in, custom headers are parsed, and unique headers are tested (if ``expect_unique = True``)
    
    2. Check for duplicate records and respond appropriately (**optional**)
    
    3. Check for duplicate sequences and respond appropriately (**optional**)
    
    4. Invalid sequences dealt with (**optional**)
    
    5. Final set of sequences/headers written to a new FASTA file (**optional**)
    
    6. Dictionary/list returned to user.
    
    Understanding there is a specific order is important when considering what options to
    pass. If a set of options are incompatible, this will be caught before the file is read.

    Parameters
    ----------

    expect_unique_header : bool
        [**Default = True**] Should the function expect each header to be unique? In general this is true for FASTA files, 
        but this is strictly not guarenteed. If this is set to True and a duplicate header is found
        then this means an error will be thrown. If it's set to false duplicate headers are dealt with,
        although for this to work ``return_list`` must also be set to True. Note that this won't happen
        automatically to avoid the scenario where you expect a dictionary to return and actually get
        a list. 

    header_parser : function
        [**Default = None**] ``header_parser`` allows a user-defined function that will be fed the FASTA header and 
        whatever it returns will be used as the actual header as the files are parsed. This can be useful if you 
        know your FASTA header has a consistent format that you want to take advantage of. A function provided here MUST        
        **(1)** Take a single input argument (the header string) and **(2)** Return a single string.
        When parsing this function the following test is applied
            >>> return_string = header_parser('this test string should work')
        Where ``return_string`` is tested to be a string. The function will show an exception if this test fails.
        
    duplicate_record_action : ``'ignore'``, ``'fail'``, ``'remove'``
        [**Default = 'fail'**] Selector that determines how to deal with duplicate entries. Note that duplicate records refers to
        entries in the fasta file where both the sequence and the header are identical. duplicate_record_action
        is only relevant keyword when expect_unique_header is False.
        Options are as follows:        
            * ``ignore``  - duplicate entries are allowed and ignored

            * ``fail``    - duplicate entries cause parsing to fail and throw an exception
  
            * ``remove`` - duplicate entries are removed, so there's only one copy of any duplicates     
    
    duplicate_sequence_action : ``'ignore'``, ``'fail'``, ``'remove'``
        [**Default = 'ignore'**] Selector that determines how to deal with duplicate sequences. This completely ignores the header
        and simply asks is two sequences are duplicated (or not). 
            * ``ignore``  - duplicate sequences are allowed and ignored

            * ``fail``    - duplicate sequences cause parsing to fail and throw an exception
  
            * ``remove`` - duplicate sequences are removed, so there's only one copy of any duplicates (1st instance kept)     
    
    invalid_sequence_action : ``'ignore'``, ``'fail'``, ``'remove'``, ``'convert'``, ``'convert-ignore', ``'convert-remove'``
        [**Default = 'fail'**] Selector that determines how to deal with invalid sequences. If ``convert`` or ``convert-ignore`` are chosen, then conversion is completed with either the standard conversion table (shown under the ``correction_dictionary`` documentation) or with a custom conversion dictionary passed to ``correction_dictionary``. 
        Options are as follows: 
            * ``ignore``  - invalid sequences are completely ignored

            * ``fail``    - invalid sequence cause parsing to fail and throw an exception
  
            * ``remove`` - invalid sequences are removed

            * ``convert`` - invalid sequences are convert

            * ``convert-ignore`` - invalid sequences are converted to valid sequences and any remaining invalid residues are ignored

            * ``convert-remove`` - invalid sequences are converted to valid sequences where possible, and any remaining sequences with invalid residues are removed

    alignment : bool
        [**Default = False**] Flag which - if set to true - the Fasta file is treated as containing alignments (with dashes) such that '-' characters are not
        treated as invalid or converted. Works in concert with other flags.  
    
    return_list : bool 
        [**Default = False**] Flag that tells the function to return a list of 2-mer lists (where position 0 is the header
        and position 1 the sequence). If you have duplicate identical headers which you want to deal with, this is required.

    output_filename : string 
        [**Default = None**] If you are performing sanitization of the input file it is often useful to write out the 
        actual set of sequences you'll be analyzing, so you have a persistent copy of this data 
        for further analysis later on. If you provide a string to output filename it will cause
        a new FASTA file to be written with the final set of sequences returned.

    correction_dictionary : dict
        [**Default = None**] **protfasta** can automatically correct non-standard amino acids to standard amino acids using the
        ``invalid_sequence`` keyword. This is useful if downstream analysis assumes/requires fully standard amino acids. 
        This is also useful for removing '-'  from aligned sequences. The standard conversions used are:
        
            * ``B   -> N``
            * ``U   -> C``
            * ``X   -> G``
            * ``Z   -> Q``
            * ``" " -> <empty string>`` (i.e. a whitespace character)
            * ``*   -> <empty string>``
            * ``-   -> <empty string>``
        However, if alternative definitions are needed they can be passed via the ``correction_dictionary`` keyword.
        The ``correction_dictionary`` should be a dictionary that maps sequences characters to some other character (ideally
        valid amino acid characters). In principle this could be used to perform arbitrary coarse-graining if a sequence...
        
    verbose : bool 
        [**Default = False**] If set to True, **protfasta** will print out information as it works its way through reading and
        parsing FASTA files. This can be useful for diagnosis.

    Returns
    ----------        
        Return type is *list* or *dict*
        If ``return_list`` is set to ``True`` then the function returns a list of lists. In each sublist contains two elements, where the first is the FASTA record header and the second the sequence. The order of FASTA records will match the order they were read in from the FASTA file. If ``return_list`` is ``False`` then the function returns a dictionary where the keys are the FASTA record heades and the values are the sequences. NOTE the order of keys will match the order that the FASTA file was read in IF the Python version is 3.7 or higher.


    Returns
    ------------
    dict or list
        Returns either a dictionary where keys are FASTA headers and values are sparrow.protein.Protein
        objects OR a list of sparrow.protein.Protein objects (if the return_true keyword was set to True)


    """
    

    # read in file
    F = protfasta.read_fasta(filename, **kwargs)


    if 'return_list' in kwargs:
        return_list = kwargs['return_list']
    else:
        return_list = False

    # build a return dictionary of Protein objects
    if return_list:
        return_dict = []
        for i in F:
            return_dict.append(Protein(i[1]))

    else:
        return_dict = {}
        for i in F:
            return_dict[i] = Protein(F[i])

        
    return return_dict
    
    
