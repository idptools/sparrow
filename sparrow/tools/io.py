import protfasta
from sparrow import protein

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

    """

    # read in file
    F = protfasta.read_fasta(filename, **kwargs)

    # build a return dictionary of Protein objects
    return_dict = {}
    for i in F:
        return_dict[i] = protein.Protein(F[i])

    return return_dict
    
    
