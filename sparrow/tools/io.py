import protfasta
import urllib3

from sparrow.protein import Protein


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
    r = http.request("GET", f"https://www.uniprot.org/uniprot/{uniprot_accession}.fasta")

    # non-200 responses (e.g. unknown accession) carry no usable FASTA
    if getattr(r, "status", 200) != 200:
        return None

    # decode the raw bytes and parse the FASTA: drop header line(s) (those
    # starting with '>') and concatenate the remaining sequence lines.
    raw = r.data.decode("utf-8", errors="replace")
    if "Sorry" in raw:
        return None

    sequence = "".join(line.strip() for line in raw.splitlines() if line and not line.startswith(">"))
    if sequence == "":
        return None

    return Protein(sequence)


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
    protfasta.protfasta_exceptions.ProtfastaException
        Propagated from the underlying ``protfasta.read_fasta`` call - for
        example if the file is missing, a header is duplicated while
        ``expect_unique_header=True``, or a sequence is invalid and
        ``invalid_sequence_action='fail'``.

    Examples
    --------
    >>> data = read_fasta('example.fasta')  # doctest: +SKIP
    >>> isinstance(next(iter(data.values())), Protein)  # doctest: +SKIP
    True
    """
    # read in file
    F = protfasta.read_fasta(filename, **kwargs)

    if isinstance(F, dict):
        return {h: Protein(seq) for h, seq in F.items()}
    return [[h, Protein(seq)] for h, seq in F]
