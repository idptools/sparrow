from sparrow.data import amino_acids


def validate_protein_sequence(
    sequence,
    allow_empty=True,
    uppercase=True,
    exception_cls=ValueError,
    sequence_name="sequence",
):
    """
    Validate and optionally normalize an amino acid sequence.

    Parameters
    ----------
    sequence : str
        Sequence to validate.
    allow_empty : bool, optional
        If False, empty sequence raises. Default True.
    uppercase : bool, optional
        If True, uppercase sequence before validation. Default True.
    exception_cls : Exception subclass, optional
        Exception type raised on validation failure. Default ValueError.
    sequence_name : str, optional
        Name used in exception messages. Default "sequence".

    Returns
    -------
    str
        Validated sequence (possibly uppercased).
    """
    if not isinstance(sequence, str):
        raise exception_cls(f"{sequence_name} must be a string")

    if uppercase:
        sequence = sequence.upper()

    if (not allow_empty) and len(sequence) == 0:
        raise exception_cls(f"{sequence_name} must be non-empty")

    invalid = sorted(set(sequence) - set(amino_acids.VALID_AMINO_ACIDS))
    if len(invalid) > 0:
        raise exception_cls(f"Invalid residue(s) found in {sequence_name}: {invalid}")

    return sequence


def normalize_residue_selector(
    residue_selector,
    selector_name="residue_selector",
    exception_cls=ValueError,
    uppercase=True,
    require_nonempty=True,
    unique=False,
    sort_unique=False,
    expected_length=None,
    return_type="list",
):
    """
    Normalize and validate a residue selector.

    Parameters
    ----------
    residue_selector : str or iterable
        One-letter residues.
    selector_name : str, optional
        Name used in exception messages.
    exception_cls : Exception subclass, optional
        Exception type raised on validation failure.
    uppercase : bool, optional
        If True, uppercase residues.
    require_nonempty : bool, optional
        If True, empty selector raises.
    unique : bool, optional
        If True, remove duplicate residues.
    sort_unique : bool, optional
        If True and unique=True, sort residues after deduplication.
    expected_length : int or None, optional
        If provided, require exact selector length after normalization.
    return_type : {'list', 'str'}, optional
        Return representation type.

    Returns
    -------
    list or str
        Normalized selector.
    """
    if isinstance(residue_selector, str):
        selector = list(residue_selector)
    else:
        try:
            selector = [str(x) for x in residue_selector]
        except TypeError:
            raise exception_cls(
                f"{selector_name} must be a string or iterable of residues"
            )

    if uppercase:
        selector = [x.upper() for x in selector]

    if unique:
        if sort_unique:
            selector = sorted(set(selector))
        else:
            # preserve first-seen order
            selector = list(dict.fromkeys(selector))

    if require_nonempty and len(selector) == 0:
        raise exception_cls(f"{selector_name} must contain one or more residues")

    if expected_length is not None and len(selector) != expected_length:
        raise exception_cls(
            f"{selector_name} must contain exactly {expected_length} residues"
        )

    for residue in selector:
        if residue not in amino_acids.VALID_AMINO_ACIDS:
            raise exception_cls(f"Invalid residue in {selector_name}: {residue}")

    if return_type == "list":
        return selector
    if return_type == "str":
        return "".join(selector)

    raise exception_cls("return_type must be 'list' or 'str'")


def is_valid_protein_sequence(sequence):
    """
    Function that tests if a passed sequence contains non-standard ammino acids

    Parameters
    ----------------
    sequence : str
        Protein sequence

    Returns
    ---------------
    bool
        If sequences contains non-standard amino acids returns False, else returns
        True
    """
    try:
        validate_protein_sequence(sequence, allow_empty=True, uppercase=False)
    except Exception:
        return False

    return True


def compare_sequence(s1, s2, verbose=False, ignore_gaps=False, return_positions=False):
    """
    Function that compares two sequences of the same length and returns
    either the set of positions where the sequences are different (indxed at 0) or
    the number of differences between them, depending on the status of the flag
    return_position. This function Will also print the differences if verbose is
    set to True.

    If ignore_gaps is set to True, will ignore gaps in the comparison (i.e.
    will ignore '-' characters in either sequence). This is useful when running
    analyses for aligned sequences.

    WARNING: Sequence must have the same length - if two passed sequences are not
    identical in terms of length then this function throws a ValueError

    Parameters
    ----------------
    s1 : str
        First sequence to compare

    s2 : str
        Second sequence to compare

    verbose : bool
        If True, will print the differences between the two sequences.
        Default is False

    return_positions : bool
        If True, will return a list of positions where the two sequences
        differ. If false return the count only.

    Returns
    ---------------
    int
        Number of differences between the two sequences

    Raises
    ---------------
    ValueError
        If sequences are not the same length.

    """

    # first things first check if sequences are the same length and
    # freak out if not!
    if len(s1) != len(s2):
        raise ValueError("Sequences must have the same length")

    # define comparison function based on ignore_gaps
    if ignore_gaps:

        def _compare(p1, p2):
            if p1 == "-" or p2 == "-":
                return False
            elif p1 == p2:
                return False
            else:
                return True
    else:

        def _compare(p1, p2):
            if p1 == p2:
                return False
            else:
                return True

    # cycle through each position in the sequence
    positions = []
    for i in range(len(s1)):
        if _compare(s1[i], s2[i]):
            positions.append(i)
            if verbose:
                print(f"{i + 1}: {s1[i]} vs. {s2[i]}")

    if return_positions:
        return positions
    else:
        return len(positions)
