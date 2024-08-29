from typing import Dict, List, Union

from protfasta import read_fasta, write_fasta
from pyfamsa import Aligner, Sequence

from sparrow import Protein
from sparrow.visualize.sequence_visuals import show_sequence


class SequenceAlignment:
    def __init__(
        self,
        input_data: Union[str, Dict[str, Protein]],
        threads: int = 0,
        scoring_matrix: str = "BLOSUM62",
        guide_tree: str = "upgma",
        tree_heuristic: Union[str, None] = None,
        medoid_threshold: int = 0,
        n_refinements: int = 200,
        keep_duplicates: bool = False,
        refine: Union[bool, None] = None,
    ):
        """
        Initialize the SequenceAlignment object.

        Parametersip
        ----------
        input_data : Union[List[Protein], str, Dict[str, str]]
            A list of Protein objects, a path to a FASTA file, or a dictionary
            of FASTA headers to sequences.
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
        self._cached_msa = None  # Cache for the computed MSA

    def _initialize_aligner(self) -> Aligner:
        """
        Initialize the Aligner object with the given parameters.
        """
        return Aligner(
            threads=self.threads,
            guide_tree=self.guide_tree,
            tree_heuristic=self.tree_heuristic,
            medoid_threshold=self.medoid_threshold,
            n_refinements=self.n_refinements,
            keep_duplicates=self.keep_duplicates,
            refine=self.refine,
            scoring_matrix=self.scoring_matrix,
        )

    @staticmethod
    def _encode_string(string_to_encode: str, encoding: str = "utf-8") -> bytes:
        """
        Encode a string to bytes using the specified encoding.
        """
        return string_to_encode.encode(encoding)

    def _load_sequences(self) -> List[Sequence]:
        """
        Load sequences from either a list of Protein objects, a FASTA file, or
        a dictionary of header-sequence mappings.

        Returns
        -------
        List[Sequence]
            A list of pyfamsa.Sequence objects for alignment.
        """
        if isinstance(self.input_data, str):
            # Assume input_data is a path to a FASTA file
            fasta_data = read_fasta(self.input_data)
            sequences = [
                Sequence(self._encode_string(header), self._encode_string(seq))
                for header, seq in fasta_data.items()
            ]
        elif isinstance(self.input_data, dict):
            # Assume input_data is a dictionary of header-sequence mappings
            sequences = [
                Sequence(self._encode_string(header), self._encode_string(seq.sequence))
                for header, seq in self.input_data.items()
            ]
        else:
            raise ValueError(
                "Invalid input_data format. Must be either a list of Protein objects, "
                "a path to a FASTA file, or a dictionary of header-sequence mappings."
            )

        return sequences

    def construct_msa(self) -> Aligner:
        """
        Construct a multiple sequence alignment with pyFAMSA.

        Returns
        -------
        Aligner
            Returns the constructed MSA as a pyfamsa._famsa.Alignment.
        """
        if self._cached_msa is not None:
            # Return cached MSA if it exists
            return self._cached_msa

        sequences = self._load_sequences()
        self._cached_msa = self.aligner.align(sequences)  # Cache the computed MSA
        return self._cached_msa

    @property
    def alignment(self) -> Aligner:
        """
        Property to access the cached MSA result.

        Returns
        -------
        Aligner
            Returns the cached MSA if available, otherwise computes it.
        """
        if self._cached_msa is None:
            # Compute MSA if it hasn't been computed yet
            self.construct_msa()
        return self._cached_msa

    def save_msa(
        self, filename: str, linelength: int = 60, append_to_fasta: bool = False
    ):
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
        write_fasta(
            fasta_data, filename, linelength=linelength, append_to_fasta=append_to_fasta
        )

    @property
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
                print(seq.id.decode().ljust(ljust), end=None)
                show_sequence(seq.sequence.decode())
            else:
                print(seq.id.decode().ljust(ljust), seq.sequence.decode())
