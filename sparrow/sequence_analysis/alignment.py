from typing import Dict, List, Literal, Union, cast

from protfasta import read_fasta, write_fasta

try:
    from pyfamsa import Aligner, Alignment, Sequence
except ModuleNotFoundError as error:
    # pyfamsa is installed only with sparrow's alignment extra
    raise ModuleNotFoundError(
        "sparrow.sequence_analysis.alignment needs pyfamsa, which sparrow installs only with its alignment "
        "extra: pip install 'idptools-sparrow[alignment] @ git+https://github.com/idptools/sparrow.git', "
        "or pip install pyfamsa.",
        name="pyfamsa",
    ) from error

from sparrow import Protein
from sparrow.visualize.sequence_visuals import show_sequence


class SequenceAlignment:
    def __init__(
        self,
        input_data: Union[str, Dict[str, Protein]],
        threads: int = 0,
        scoring_matrix: str = "BLOSUM62",
        guide_tree: Literal["sl", "slink", "upgma", "nj"] = "upgma",
        tree_heuristic: Union[Literal["medoid", "parttree"], None] = None,
        medoid_threshold: Union[int, None] = None,
        n_refinements: int = 200,
        keep_duplicates: bool = False,
        refine: Union[bool, None] = None,
    ):
        """
        Set up a multiple sequence alignment of a set of proteins, made with pyFAMSA.

        Nothing is aligned until :meth:`construct_msa` or :attr:`alignment` is
        first used. The arguments after ``input_data`` are FAMSA's, passed to
        ``pyfamsa.Aligner``.

        Parameters
        ----------
        input_data : str or dict of sparrow.protein.Protein
            The path to a FASTA file, or a dict of ``Protein`` objects keyed by
            the names the alignment gives them.
        threads : int, optional
            Threads to align with. Default 0, one for each CPU.
        scoring_matrix : str, optional
            The substitution matrix, by name. Default ``'BLOSUM62'``.
        guide_tree : {'sl', 'slink', 'upgma', 'nj'}, optional
            How the guide tree is built: single linkage (by MST and Prim's
            algorithm, or by SLINK), UPGMA or neighbour joining. Default
            ``'upgma'``.
        tree_heuristic : {'medoid', 'parttree'} or None, optional
            A heuristic for building the guide tree of a large set, or None
            for none. Default None.
        medoid_threshold : int or None, optional
            Number of sequences at or above which pyFAMSA switches to medoid-seeded
            guide-tree construction. ``None`` (the default) leaves pyFAMSA's own
            default in place, currently 2000. If you do set it, pyFAMSA requires it
            to be at least ``medoid_seeds`` (100 by default) and raises otherwise.
        n_refinements : int, optional
            Refinement iterations. Default 200.
        keep_duplicates : bool, optional
            Keep duplicate sequences, rather than discarding them before the
            guide tree is built. Default False.
        refine : bool or None, optional
            True to refine the alignment, False not to, or None (the default)
            to let pyFAMSA decide: it refines sets of up to 1000 sequences.

        Notes
        -----
        ``input_data`` is read when the proteins are aligned, which raises a
        ``ValueError`` if it is neither a path nor a dict.
        """
        self.input_data = input_data
        self.threads = threads
        self.guide_tree = guide_tree
        self.tree_heuristic = tree_heuristic
        self.medoid_threshold = medoid_threshold
        self.n_refinements = n_refinements
        self.keep_duplicates = keep_duplicates
        self.refine = refine
        self.scoring_matrix = scoring_matrix
        self.aligner = self._initialize_aligner()
        self._cached_msa: Union[Alignment, None] = None  # Cache for the computed MSA

    def _initialize_aligner(self) -> Aligner:
        """
        Initialize the Aligner object with the given parameters.
        """
        return Aligner(
            threads=self.threads,
            guide_tree=self.guide_tree,
            # pyfamsa's own _famsa.pyi declares Literal['medoid', 'part'], but its
            # runtime validator accepts 'parttree' and rejects 'part'. Trust the
            # runtime: the annotation above is correct and the stub is wrong.
            tree_heuristic=self.tree_heuristic,  # ty: ignore[invalid-argument-type]
            n_refinements=self.n_refinements,
            keep_duplicates=self.keep_duplicates,
            refine=self.refine,
            scoring_matrix=self.scoring_matrix,
            # Only forwarded when set. This used to default to 0, which is not
            # "off" but "always" - and pyFAMSA rejects any threshold below its
            # medoid_seeds (100), so every SequenceAlignment(...) raised
            # ValueError on construction. Omitting the argument keeps pyFAMSA's
            # default rather than restating a number that can drift upstream.
            **({} if self.medoid_threshold is None else {"medoid_threshold": self.medoid_threshold}),
        )

    @staticmethod
    def _encode_string(string_to_encode: str, encoding: str = "utf-8") -> bytes:
        """
        Encode a string to bytes using the specified encoding.
        """
        return string_to_encode.encode(encoding)

    def _load_sequences(self) -> List[Sequence]:
        """
        Load the sequences from a FASTA file or a dict of Protein objects.

        Returns
        -------
        List[Sequence]
            A list of pyfamsa.Sequence objects for alignment.
        """
        if isinstance(self.input_data, str):
            # Assume input_data is a path to a FASTA file
            # read_fasta is annotated Union[dict, list[list[str]]] because of its
            # return_list flag; with the default it always returns the dict form.
            fasta_data = cast(Dict[str, str], read_fasta(self.input_data))
            sequences = [
                Sequence(self._encode_string(header), self._encode_string(seq)) for header, seq in fasta_data.items()
            ]
        elif isinstance(self.input_data, dict):
            # Assume input_data is a dictionary of header-sequence mappings
            sequences = [
                Sequence(self._encode_string(header), self._encode_string(seq.sequence))
                for header, seq in self.input_data.items()
            ]
        else:
            raise ValueError(
                "Invalid input_data format. Must be the path to a FASTA file, or a dict of Protein objects keyed by "
                "name."
            )

        return sequences

    def construct_msa(self) -> Alignment:
        """
        Construct a multiple sequence alignment with pyFAMSA.

        Returns
        -------
        Alignment
            The constructed MSA, as a ``pyfamsa.Alignment``.
        """
        if self._cached_msa is not None:
            # Return cached MSA if it exists
            return self._cached_msa

        sequences = self._load_sequences()
        self._cached_msa = self.aligner.align(sequences)  # Cache the computed MSA
        return self._cached_msa

    @property
    def alignment(self) -> Alignment:
        """
        Property to access the cached MSA result.

        Returns
        -------
        Alignment
            The cached MSA if available, otherwise computes it.
        """
        # construct_msa() returns the cached alignment when there is one, so this
        # both computes on first access and keeps the return type non-optional.
        return self.construct_msa()

    def save_msa(self, filename: str, linelength: int = 60, append_to_fasta: bool = False):
        """
        Save the multiple sequence alignment to a FASTA file.

        Parameters
        ----------
        filename : str
            The filename to save the MSA. Should end with .fasta or .fa.

        linelength : int, optional
            Length of lines in the output file, by default 60.

        append_to_fasta : bool, optional
            Whether to append to an existing FASTA file, by default False.
        """
        msa = self.alignment
        fasta_data = {seq.id.decode(): seq.sequence.decode() for seq in msa}
        write_fasta(fasta_data, filename, linelength=linelength, append_to_fasta=append_to_fasta)

    def display_msa(self, ljust: int = 10, html: bool = False):
        """
        Print the multiple sequence alignment using the cached MSA.

        Parameters
        ----------
        ljust : int, optional
            The number of spaces to pad the sequence ID, by default 10

        html : bool, optional
            Set to True to print the alignment in HTML format, by default False
        """
        msa = self.alignment

        for seq in msa:
            if html:
                print(seq.id.decode().ljust(ljust), end="")
                show_sequence(seq.sequence.decode())
            else:
                print(seq.id.decode().ljust(ljust), seq.sequence.decode())
