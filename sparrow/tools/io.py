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

    **kwargs
        Keyword arguments as defined in protfasta.read_fasta

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
    
    
def validate_keyword_option(keyword, allowed_vals, keyword_name, error_message=None):
    """
    Helper function that checks a passed keyword is only one of a set of possible
    valid keywords

    Parameters
    -----------
    keyword : str
        The actual passed keyword value

    allowed_vals : list of str
        A list of possible keywords

    keyword_name : str
        the name of the keyword as the user would select it in the function call

    error_message : str
        Allows the user to pass a custom error message


    Returns
    --------
    None

        No return value, but raises ctexceptions.CTException if ``keyword `` is not
        found in the allowed_vals list
           
    """


    if keyword not in allowed_vals:
        if error_message is None:
            raise SparrowException('Keyword %s passed value [%s], but this is not valid.\nMust be one of: %s'  % (keyword_name, keyword, ", ".join(allowed_vals)))
