from typing import List, Union
from pyfamsa import Aligner, Sequence
from sparrow.protein import Protein
from sparrow import read_fasta
from sparrow.visualize.sequence_visuals import show_sequence

from IPython import embed


def __encode_string(string_to_encode, encoding="utf-8"):
    return string_to_encode.encode(f"{encoding}")

def multiple_sequence_alignment(protein_objs : List[Protein], 
                                threads : int = 0,
                                guide_tree : str = "upgma",
                                tree_heuristic : Union[str, None] = None,
                                medoid_threshold : int = 0,
                                n_refinements : int = 200,
                                keep_duplicates : bool = False,
                                refine : Union[bool, None]= None,
                                ) -> Aligner:
    """Construct a multiple sequence alignment with pyFAMSA

    Parameters
    ----------
    protein_objs : List[Protein]
        a list of sparrow.Protein objects to align

    threads : int
        The number of threads to use for parallel computations. 
        If none provided (the default), use os.cpu_count to spawn one thread per CPU on the host machine.

    guide_tree : str, optional
        The method for building the guide tree by default "upgma" is used.
        Supported values are: 
            sl for MST+Prim single linkage, 
            slink for SLINK single linkage,
            upgma for UPGMA, 
            nj for neighbour joining.
        
    tree_heuristic : Union[str, None], optional
        The heuristic to use for constructing the tree, by default None
            Supported values are: 
            medoid for medoid trees,
            part for part trees, 
            or None to disable heuristics.

    medoid_threshold : int, optional
        The minimum number of sequences a set must contain for medoid trees to be used,
        if enabled with tree_heuristic., by default 0
    n_refinements : int, optional
        The number of refinement iterations to run, by default 200
        
    keep_duplicates : bool, optional
        Set to True to avoid discarding duplicate sequences before building trees or alignments, by default False
    refine : Union[bool, None], optional

        Set to True to force refinement, False to disable refinement, or 
        leave as None to disable refinement automatically for sets of more than 1000 sequences., by default None

    Returns
    -------
    Aligner
        Returns the constructed MSA as a pyfamsa._famsa.Alignment.
    """
    sequences = [Sequence(__encode_string(header), __encode_string(protein_obj.sequence)) for header, protein_obj in protein_objs.items()]

    aligner = Aligner(threads=threads,
                      guide_tree=guide_tree,
                      tree_heuristic=tree_heuristic,
                      medoid_threshold=medoid_threshold,
                      n_refinements=n_refinements,
                      keep_duplicates=keep_duplicates,
                      refine=refine)
    
    msa = aligner.align(sequences)

    return msa

def print_msa(msa,ljust=10,html=False):
    for seq in msa:
        if html:
            print(seq.id.decode().ljust(ljust), end=None)
            show_sequence(seq.sequence.decode())
        else:
            print(seq.id.decode().ljust(ljust), seq.sequence.decode())
