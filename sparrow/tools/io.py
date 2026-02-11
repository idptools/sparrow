import protfasta
import urllib3
import numpy as np

from sparrow.protein import Protein
from sparrow.sparrow_exceptions import SparrowException


def uniprot_fetch(uniprot_accession):
    """Fetch a UniProt sequence and return a Protein object.

    This performs a lightweight HTTP request to the UniProt service. It is
    intended for interactive / small scale use only. Excessive automated
    querying may violate UniProt's usage policies and get your IP blocked.

    Parameters
    ----------
    uniprot_accession : str
        UniProt accession (no validation beyond fetch attempt).

    Returns
    -------
    Protein or None
        ``Protein`` object if retrieval succeeds, otherwise ``None`` (e.g. if
        accession not found or service error pattern detected).

    Notes
    -----
    The returned object is constructed from the raw FASTA sequence; any header
    metadata is discarded. Errors are silenced and expressed as ``None`` to
    keep this helper lightweight. For robust batch workflows prefer the UniProt
    API clients or bulk download services.
    """
    http = urllib3.PoolManager()
    r = http.request(
        "GET", f"https://www.uniprot.org/uniprot/{uniprot_accession}.fasta"
    )
    s = "".join(str(r.data).split("\n")[1:]).replace("'", "")
    if "Sorry" in s:
        return None
    return Protein(s)


def read_fasta(filename, **kwargs):
    """Read a FASTA file and return Protein objects.

    Wraps :func:`protfasta.read_fasta` adding conversion to ``Protein`` instances
    and optional list-mode return. All keyword arguments are forwarded to
    ``protfasta.read_fasta``.

    Parameters
    ----------
    filename : str
        Path to FASTA file.
    return_list : bool, optional (in **kwargs**)
        If True, return a list ``[[header, Protein], ...]``; otherwise return a
        dict ``{header: Protein}``. Default False.
    expect_unique_header : bool, optional
        Passed through; if True duplicate headers raise. Default True.
    header_parser : callable, optional
        User function transforming each header. Must accept and return str.
    duplicate_record_action : {'ignore','fail','remove'}, optional
        Strategy for exact duplicate header+sequence pairs (when
        ``expect_unique_header`` is False). Default 'fail'.
    duplicate_sequence_action : {'ignore','fail','remove'}, optional
        Strategy for duplicate sequences irrespective of header. Default 'ignore'.
    invalid_sequence_action : {'ignore','fail','remove','convert','convert-ignore','convert-remove'}, optional
        Handling for invalid sequences. Default 'fail'.
    alignment : bool, optional
        Treat sequences as alignments (keeps '-'). Default False.
    output_filename : str, optional
        If provided, sanitized output FASTA filename.
    correction_dictionary : dict, optional
        Mapping for converting non‑standard residues when using conversion modes.
    verbose : bool, optional
        Emit diagnostic output from protfasta. Default False.

    Returns
    -------
    dict[str, Protein] or list[list]
        Dictionary (default) mapping headers to ``Protein`` objects, or list of
        ``[header, Protein]`` pairs if ``return_list=True``.

    Raises
    ------
    SparrowException
        Propagated if underlying protfasta raises and is not handled by mode
        selections.

    Examples
    --------
    >>> data = read_fasta('example.fasta')  # doctest: +SKIP
    >>> isinstance(next(iter(data.values())), Protein)  # doctest: +SKIP
    True
    """
    # read in file
    F = protfasta.read_fasta(filename, **kwargs)
    return_list = kwargs.get("return_list", False)
    if return_list:
        return [[h, Protein(seq)] for h, seq in F]
    else:
        return {h: Protein(seq) for h, seq in F.items()}


def build_grammar_background_from_fasta(
    fasta_filename,
    output_filename,
    dtype=np.float32,
    compressed=True,
    **read_fasta_kwargs,
):
    """Compute and save grammar composition background stats from a FASTA file.

    This helper reads FASTA records as ``Protein`` objects, computes
    composition/patch background means and standard deviations, and writes a
    compact NumPy archive for downstream grammar z-score workflows.

    Parameters
    ----------
    fasta_filename : str
        Input FASTA path.
    output_filename : str
        Destination ``.npz`` filename.
    dtype : numpy dtype, optional
        Numeric dtype for saved mean/std arrays. Default ``np.float32``.
    compressed : bool, optional
        If True, write with ``np.savez_compressed``. Default True.
    **read_fasta_kwargs
        Forwarded to :func:`read_fasta`.

    Returns
    -------
    sparrow.sequence_analysis.grammar.GrammarCompositionStats
        Computed background statistics object.
    """
    # Local import avoids module-level circular dependency.
    from sparrow.sequence_analysis import grammar

    proteins = read_fasta(fasta_filename, **read_fasta_kwargs)
    stats = grammar.compute_composition_background_stats(
        proteins, dtype=dtype
    )
    grammar.save_composition_stats_npz(
        output_filename=output_filename,
        composition_stats=stats,
        dtype=dtype,
        compressed=compressed,
    )
    return stats
