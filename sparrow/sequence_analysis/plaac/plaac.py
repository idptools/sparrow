"""
PLAAC: Prion-Like Amino Acid Composition
=========================================

A Python implementation of the PLAAC algorithm for identifying prion-like
domains in protein sequences.

This is a faithful port of the original Java program by Oliver King
(oliver.king@umassmed.edu).

Original program copyright 2009 Whitehead Institute for Biomedical Research;
additions copyright 2011 BBRI and copyright 2014 University of Massachusetts
Medical School.

The program combines:
  - A hidden Markov model (HMM) for identifying prion-like domains
  - FoldIndex disorder prediction (Prilusky et al, Bioinformatics, 2005)
  - PAPA prion propensity scoring (Toombs et al, MCB 2010; PNAS 2012)
  - Michelitsch-Weissman Q/N enrichment scoring (PNAS, 2000)

References
----------
- Alberti S, Halfmann R, King O, Kapila A, Lindquist S. Cell 2009.
- Toombs JA, McCarty BR, Ross ED. Mol Cell Biol 30:319-332, 2010.
- Toombs JA, Petri M, Paul KR, et al. PNAS 109:6519-6524, 2012.
- Prilusky J, et al. Bioinformatics 21:3435-3438, 2005.
- Durbin R, Eddy S, Krogh A, Mitchison G. "Biological Sequence Analysis",
  Cambridge University Press, 1998.

Usage
-----
    python plaac.py -i input.fa > output.txt
    python plaac.py --help

See LICENSE.TXT for license information.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, TextIO, Tuple

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

LOG2 = math.log(2)

#: Single-letter amino acid names indexed 0..21 (X=unknown, \*=stop)
AA_NAMES: List[str] = list("XACDEFGHIKLMNPQRSTVWY*")

#: Three-letter amino acid names indexed 0..21
AA_NAMES_3: List[str] = [
    "???", "Ala", "Cys", "Asp", "Glu", "Phe", "Gly", "His", "Ile", "Lys",
    "Leu", "Met", "Asn", "Pro", "Gln", "Arg", "Ser", "Thr", "Val", "Trp",
    "Tyr", "***",
]

#: Mapping from single-letter AA code to integer index
_AA_TO_INT: Dict[str, int] = {c: i for i, c in enumerate(AA_NAMES)}

# FoldIndex weights: fi = cc[0]*hydro + cc[1]*|charge| + cc[2]
FOLDINDEX_COEFFS: List[float] = [2.785, -1.0, -1.151]


# ═══════════════════════════════════════════════════════════════════════════════
# Amino Acid Property Tables
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AminoAcidProperties:
    """
    Physical / statistical properties for each of the 22 amino acid slots.

    Each attribute is a list of length 22 indexed by the integer AA code
    (0=X, 1=A, ..., 20=Y, 21=*).
    """

    #: Formal charge used by FoldIndex (D+1, E+1, K-1, R-1; rest 0)
    charge: Tuple[float, ...] = (
        0, 0, 0, 1, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0,
    )

    #: Kyte-Doolittle hydropathicity (J Mol Biol 157:105-132, 1982)
    hydro: Tuple[float, ...] = (
        0.0, 1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9,
        3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3, 0.0,
    )

    #: PAPA foreground frequencies - set 1 (Toombs et al MCB 2010)
    fg_papa1: Tuple[float, ...] = (
        0.0, 0.042, 0.033, 0.014, 0.009, 0.075, 0.038, 0.059, 0.102, 0.009,
        0.059, 0.038, 0.096, 0.038, 0.024, 0.054, 0.125, 0.069, 0.102, 0.024,
        0.054, 0.0,
    )

    #: PAPA background frequencies - set 1 (Toombs et al MCB 2010)
    bg_papa1: Tuple[float, ...] = (
        0.0, 0.072, 0.022, 0.051, 0.017, 0.032, 0.040, 0.078, 0.045, 0.045,
        0.061, 0.020, 0.089, 0.127, 0.022, 0.081, 0.109, 0.078, 0.045, 0.012,
        0.025, 0.0,
    )

    #: PAPA odds-ratios - set 1 (Toombs et al MCB 2010)
    od_papa1: Tuple[float, ...] = (
        0.0,
        0.67267686, 1.5146198, 0.27887323, 0.5460614, 2.313433,
        0.96153843, 0.75686276, 2.2562358, 0.20664589, 0.9607843,
        1.9615384, 1.0836071, 0.30196398, 1.0716166, 0.6664044,
        1.1432927, 0.8917492, 2.2562358, 1.9478673, 2.1785367,
        0.0,
    )

    #: S. cerevisiae global AA frequencies (Alberti et al, Cell 2009)
    bg_freq_scer: Tuple[float, ...] = (
        0, 0.0550, 0.0126, 0.0586, 0.0655, 0.0441, 0.0498, 0.0217, 0.0655,
        0.0735, 0.0950, 0.0207, 0.0615, 0.0438, 0.0396, 0.0444, 0.0899,
        0.0592, 0.0556, 0.0104, 0.0337, 0,
    )

    #: Prion-domain frequencies from 4 S. cer. prions (Alberti et al 2009)
    prd_freq_scer_04: Tuple[float, ...] = (
        0, 0.0488, 0.0032, 0.0202, 0.0234, 0.0276, 0.1157, 0.0149, 0.0191,
        0.0329, 0.0456, 0.0149, 0.1444, 0.0308, 0.2208, 0.0202, 0.1008,
        0.0297, 0.0234, 0.0064, 0.0573, 0,
    )

    #: Prion-domain frequencies from 28 S. cer. domains (Alberti et al 2009)
    prd_freq_scer_28: Tuple[float, ...] = (
        0, 0.04865, 0.00219, 0.01638, 0.00783, 0.02537, 0.07603, 0.0181,
        0.02018, 0.01641, 0.02639, 0.02975, 0.25885, 0.05126, 0.15178,
        0.025, 0.10988, 0.03841, 0.01972, 0.00157, 0.05624, 0,
    )

    @property
    def hydro_scaled(self) -> List[float]:
        """Scaled & shifted hydropathicity used by FoldIndex: H/9 + 0.5."""
        return [h / 9.0 + 0.5 for h in self.hydro]

    @property
    def lod_papa1(self) -> List[float]:
        """Log-odds of PAPA set-1 odds-ratios."""
        out = [0.0] * 22
        for k in range(1, 21):
            out[k] = math.log(self.od_papa1[k]) if self.od_papa1[k] > 0 else float("-inf")
        return out


#: Module-level singleton of amino acid properties.
AA = AminoAcidProperties()


# ═══════════════════════════════════════════════════════════════════════════════
# Look-up table for log-sum-exp  (used by the HMM in log-space)
# ═══════════════════════════════════════════════════════════════════════════════

_LOGLUT_LEN = 4000
_LOGLUT: List[float] = [
    math.log(1.0 + math.exp(-i / 100.0)) for i in range(_LOGLUT_LEN + 1)
]


def _log_add_exp(a: float, b: float) -> float:
    """
    Numerically stable computation of ``log(exp(a) + exp(b))``.

    Uses a pre-computed look-up table with linear interpolation for speed.
    Correctly handles *-inf* arguments.
    """
    if a > b:
        c = a - b
        if not (c < 40):
            return a          # b is negligible
        idx = int(c * 100)
        frac = c * 100 - idx
        return a + ((frac) * _LOGLUT[idx + 1] + (1 - frac) * _LOGLUT[idx])
    elif b > a:
        c = b - a
        if not (c < 40):
            return b          # a is negligible
        idx = int(c * 100)
        frac = c * 100 - idx
        return b + ((frac) * _LOGLUT[idx + 1] + (1 - frac) * _LOGLUT[idx])
    else:
        return a + LOG2       # handles a == b == -inf


# ═══════════════════════════════════════════════════════════════════════════════
# Small numeric helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize(arr: Sequence[float]) -> List[float]:
    """
    Return a copy of *arr* scaled so the elements sum to 1.

    If all elements are zero (or near-zero), returns all zeros rather than
    producing NaN.
    """
    s = sum(arr)
    if s < 1e-12:
        s = 1.0
    return [x / s for x in arr]


def _inf_to_nan(x: float) -> float:
    """Replace +/-inf with NaN (friendlier for downstream tools like Excel)."""
    return float("nan") if math.isinf(x) else x


def _mapseq(aa_seq: Sequence[int], weights: Sequence[float]) -> List[float]:
    """Map an integer AA sequence through a weight table."""
    return [weights[a] for a in aa_seq]


def _longest_run(bitvec: Sequence[int]) -> int:
    """Return the length of the longest contiguous run of 1s."""
    best = 0
    cur = 0
    for v in bitvec:
        if v > 0:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _extract_prd_runs(vp: Sequence[int]) -> List[tuple]:
    """
    Extract all contiguous runs of PrD-like states (value 1) from a
    Viterbi path.

    Returns
    -------
    list of (start, end)
        Each tuple gives the 0-based inclusive start and end indices of a
        contiguous run of 1s.
    """
    runs: List[tuple] = []
    i = 0
    n = len(vp)
    while i < n:
        if vp[i] == 1:
            start = i
            while i < n and vp[i] == 1:
                i += 1
            runs.append((start, i - 1))
        else:
            i += 1
    return runs


def _subvec(vec: Sequence, lo: int, hi: int) -> list:
    """
    Extract ``vec[lo : hi+1]`` with bounds clamping.

    Parameters
    ----------
    vec : sequence
    lo, hi : int
        Inclusive bounds, clamped to ``[0, len(vec)-1]``.
    """
    n = len(vec)
    lo = max(0, min(lo, n - 1))
    hi = max(lo, min(hi, n - 1))
    return list(vec[lo: hi + 1])


# ═══════════════════════════════════════════════════════════════════════════════
# AA <-> integer conversions
# ═══════════════════════════════════════════════════════════════════════════════

def aa_to_int(c: str) -> int:
    """Convert a single amino acid character to its integer index (0-21)."""
    return _AA_TO_INT.get(c.upper(), 0)


def seq_to_ints(seq: str) -> List[int]:
    """Convert a protein sequence string to a list of integer AA indices."""
    return [aa_to_int(c) for c in seq]


def ints_to_seq(aa: Sequence[int]) -> str:
    """Convert a list of integer AA indices back to a one-letter string."""
    return "".join(AA_NAMES[a] for a in aa)


def is_valid_protein(aa: Sequence[int]) -> bool:
    """
    Return *True* if *aa* has no premature stops or unknown residues.

    A terminal stop codon (index 21) is tolerated in the last position.
    """
    if not aa:
        return False
    for a in aa[:-1]:
        if a == 0 or a == 21:
            return False
    return aa[-1] != 0


# ═══════════════════════════════════════════════════════════════════════════════
# FASTA reader
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FastaRecord:
    """A single FASTA record with a header name and sequence string."""
    name: str
    sequence: str


def read_fasta(path: str | Path) -> Iterator[FastaRecord]:
    """
    Lazily iterate over FASTA records in *path*.

    Yields
    ------
    FastaRecord
        One record per ``>header`` block in the file.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    current_name: Optional[str] = None
    parts: List[str] = []

    with open(path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    yield FastaRecord(name=current_name, sequence="".join(parts))
                current_name = line[1:].strip()
                parts = []
            else:
                parts.append(line)

        # last record
        if current_name is not None:
            yield FastaRecord(name=current_name, sequence="".join(parts))


# ═══════════════════════════════════════════════════════════════════════════════
# AA frequency I/O
# ═══════════════════════════════════════════════════════════════════════════════

def compute_aa_freq(path: str | Path) -> List[int]:
    """
    Compute singleton amino acid counts across all sequences in a FASTA file.

    Returns
    -------
    list of int
        Length-22 count vector indexed by integer AA code.
    """
    counts = [0] * 22
    for rec in read_fasta(path):
        aa = seq_to_ints(rec.sequence)
        if is_valid_protein(aa):
            for a in aa:
                counts[a] += 1
    return counts


def read_aa_params(path: str | Path) -> List[float]:
    """
    Read a 22-element AA parameter file (one value per line).

    Expected format per line::

        <float_value> # <AA_letter>

    The ``# <AA_letter>`` part is optional but will be sanity-checked if
    present.
    """
    vec = [0.0] * 22
    with open(path) as fh:
        for i in range(22):
            tokens = fh.readline().strip().split()
            vec[i] = float(tokens[0])
            if len(tokens) > 2 and tokens[2][0] != AA_NAMES[i]:
                print(
                    f"# warning: {path} does not have expected name in line {i + 1}",
                    file=sys.stderr,
                )
    return vec


def format_aa_params(params: Sequence[float]) -> str:
    """Format 22 AA parameters as a compact ``A=0.05000;C=0.01000;...`` string."""
    return "".join(f"{AA_NAMES[i]}={params[i]:.5f};" for i in range(22))


def write_aa_params(params: Sequence[float], out: TextIO = sys.stdout) -> None:
    """Write 22 AA parameters (one per line) to *out*."""
    for i in range(22):
        out.write(f"{params[i]:.6f} # {AA_NAMES[i]}\n")


def read_gene_list(path: str | Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Read a tab-delimited gene list for the ``-p`` plot option.

    Returns
    -------
    aliases : dict
        name -> synonym (for plot titles).
    ordering : dict
        name -> sequential integer string (for ordering plots).
    """
    aliases: Dict[str, str] = {}
    ordering: Dict[str, str] = {}
    with open(path) as fh:
        for i, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            key = parts[0]
            aliases[key] = parts[1] if len(parts) > 1 else key
            ordering[key] = str(i)
    return aliases, ordering


# ═══════════════════════════════════════════════════════════════════════════════
# Sliding-window averages
# ═══════════════════════════════════════════════════════════════════════════════

def sliding_average(
    arr: Sequence[float],
    window: int,
    *,
    shrink: bool = True,
    weighted: bool = False,
) -> List[float]:
    """
    Compute a sliding-window average.

    Parameters
    ----------
    arr : sequence of float
        Input values (one per residue).
    window : int
        Window size (should be odd).
    shrink : bool, default True
        If *True*, reduce window size at sequence boundaries.  If *False*,
        positions where the full window does not fit are set to NaN.
    weighted : bool, default False
        If *True*, weight each element by its effective window size (the
        "average of averages" weighting used by PAPA).
    """
    n = len(arr)
    if n == 0:
        return []
    w = window // 2
    if w >= n:
        w = n - 1
    sa = [0.0] * n

    if shrink:
        lo, hi = 0, n - 1
    else:
        lo, hi = w, n - w - 1
        for i in range(lo):
            sa[i] = float("nan")
        for i in range(hi + 1, n):
            sa[i] = float("nan")

    for i in range(lo, hi + 1):
        score = 0.0
        denom = 0.0
        for j in range(-w, w + 1):
            idx = i + j
            if 0 <= idx < n:
                wt = 1.0
                if weighted:
                    wt = 1.0 + min(idx, w) + min(n - idx - 1, w)
                denom += wt
                score += wt * arr[idx]
        sa[i] = score / denom if denom else 0.0
    return sa


def sliding_average_papa(
    arr: Sequence[float],
    window: int,
    seq: Sequence[int],
    proline_idx: int = 13,
) -> List[float]:
    """
    Sliding average with PAPA proline-adjustment.

    In each window, only the *first* proline in a PP or PXP motif is scored
    (subsequent prolines are skipped), following Toombs et al. (2010).

    Parameters
    ----------
    arr : sequence of float
        Per-residue PAPA propensity values.
    window : int
        Window size (should be odd).
    seq : sequence of int
        Integer-coded amino acid sequence (proline = *proline_idx*).
    proline_idx : int, default 13
        Integer code for proline.
    """
    n = len(arr)
    if n == 0:
        return []
    w = window // 2
    if w >= n:
        w = n - 1
    sa = [0.0] * n

    for i in range(n):
        score = 0.0
        denom = 0.0
        for j in range(-w, w + 1):
            idx = i + j
            if 0 <= idx < n:
                denom += 1.0
                # Skip if this proline follows another proline at idx-1 or idx-2
                if seq[idx] == proline_idx:
                    if idx >= 1 and seq[idx - 1] == proline_idx:
                        continue
                    if idx >= 2 and seq[idx - 2] == proline_idx:
                        continue
                score += arr[idx]
        sa[i] = score / denom if denom else 0.0
    return sa


# ═══════════════════════════════════════════════════════════════════════════════
# Highest-Scoring Subsequence (HSS)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HSSResult:
    """Result of a highest-scoring subsequence search."""
    start: int     #: Start index (0-based) of the best segment.
    end: int       #: End index (0-based, inclusive) of the best segment.
    score: float   #: Sum of values in the best segment.

    @property
    def length(self) -> int:
        """Length of the segment (end - start + 1)."""
        return self.end - self.start + 1

    @staticmethod
    def empty() -> HSSResult:
        """Return a sentinel result indicating no valid segment was found."""
        return HSSResult(start=-1, end=-2, score=float("nan"))


def highest_scoring_subseq(
    seq: Sequence[float],
    min_length: int,
    max_length: int,
) -> HSSResult:
    """
    Find the contiguous subsequence with maximum sum, subject to length
    constraints.

    Uses a prefix-sum approach to efficiently search over all valid windows.

    Parameters
    ----------
    seq : sequence of float
        Per-position scores.
    min_length, max_length : int
        Inclusive bounds on allowed segment lengths.

    Returns
    -------
    HSSResult
        Start/end indices and the score.  If no valid segment exists
        (e.g. the sequence is shorter than *min_length*), the returned
        result is ``HSSResult.empty()``.
    """
    n = len(seq)
    if min_length > n or min_length > max_length:
        return HSSResult.empty()
    if max_length > n:
        max_length = n

    # Prefix sums
    psum = [0.0] * (n + 1)
    for i in range(n):
        psum[i + 1] = psum[i] + seq[i]

    best_start = 0
    best_stop = min_length - 1
    cur_start = 0
    best_score = psum[min_length]

    for i in range(min_length, n):
        if (i - cur_start) >= max_length:
            cur_start += 1
        d = psum[i + 1] - psum[cur_start]
        new_start = cur_start
        for j in range(cur_start + 1, i - min_length):
            if psum[i + 1] - psum[j] >= d:
                d = psum[i + 1] - psum[j]
                new_start = j
            cur_start = new_start
        if d > best_score:
            best_score = d
            best_stop = i
            best_start = cur_start

    return HSSResult(start=best_start, end=best_stop, score=best_score)


# ═══════════════════════════════════════════════════════════════════════════════
# Hidden Markov Model
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_log(x: float) -> float:
    """Log that returns -inf for zero."""
    return math.log(x) if x > 0 else float("-inf")


def _collapse_posteriors(
    pp: List[List[float]], classes: List[int], num_classes: int
) -> List[List[float]]:
    """Collapse per-state posteriors into per-class posteriors by summation."""
    n = len(pp[0])
    npp = [[0.0] * n for _ in range(num_classes)]
    for i in range(len(pp)):
        for j in range(n):
            npp[classes[i]][j] += pp[i][j]
    return npp


class HMM:
    """
    Discrete-output Hidden Markov Model with Viterbi and MAP decoding.

    The algorithms follow Chapter 3 of Durbin et al. (1998).  All probability
    computations are done in log-space to prevent numerical underflow.

    Parameters
    ----------
    trans : list[list[float]]
        Transition probability matrix (*ns* x *ns*).
    emis : list[list[float]]
        Emission probability matrix (*ns* x *no*).
    init : list[float]
        Initial state distribution (length *ns*).

    Attributes
    ----------
    ns : int
        Number of states.
    no : int
        Number of output symbols.
    viterbi_path : list[int]
        Most-probable state path after calling :meth:`decode`.
    map_path : list[int]
        Per-position MAP states after calling :meth:`decode`.
    posterior : list[list[float]] or None
        Posterior probability matrix after calling :meth:`decode`.
    log_viterbi_prob : float
        Log P(seq, best path) after Viterbi decoding.
    log_marginal_prob : float
        Log P(seq) after forward-backward.
    """

    def __init__(
        self,
        trans: List[List[float]],
        emis: List[List[float]],
        init: List[float],
    ) -> None:
        self.trans = trans
        self.emis = emis
        self.init = init

        self.ns: int = len(init)
        self.no: int = len(emis[0])

        # Log-space versions
        self._ltrans = [[_safe_log(v) for v in row] for row in trans]
        self._lemis = [[_safe_log(v) for v in row] for row in emis]
        self._linit = [_safe_log(v) for v in init]

        # Termination probabilities (free-end model if rows already sum to 1)
        rs = [sum(row) for row in trans]
        fprob = [max(0.0, 1.0 - r) for r in rs]
        if all(f < 1e-4 for f in fprob):
            fprob = [1.0] * self.ns
        self._lfprob = [_safe_log(v) for v in fprob]

        # State metadata
        self.subtrellis: List[int] = [1] * self.ns
        self.state_chars: List[str] = [chr(i) for i in range(self.ns)]
        self.names: List[str] = [f"s{i}" for i in range(self.ns)]

        # Class collapsing (multiple states -> one output class)
        self.classes: List[int] = list(range(self.ns))
        self.num_classes: int = self.ns
        self.class_names: List[str] = self.names[:]

        # Results from most recent decoding
        self.viterbi_path: List[int] = []
        self.map_path: List[int] = []
        self.posterior: Optional[List[List[float]]] = None
        self.log_viterbi_prob: float = float("-inf")
        self.log_marginal_prob: float = float("-inf")

    # ── Configuration helpers ──────────────────────────────────────────────

    def set_names(self, names: List[str]) -> None:
        """Set human-readable state names."""
        self.names = names
        self.class_names = names[:]

    def set_classes(self, classes: List[int]) -> None:
        """
        Group states into classes for collapsed posterior output.

        Parameters
        ----------
        classes : list of int
            ``classes[i]`` gives the output-class index for state *i*.
        """
        self.classes = classes
        self.num_classes = max(classes) + 1
        self.class_names = [None] * self.num_classes  # type: ignore[list-item]
        for i in range(len(classes) - 1, -1, -1):
            self.class_names[classes[i]] = self.names[i]

    # ── Decoding ───────────────────────────────────────────────────────────

    def decode(self, seq: Sequence[int]) -> None:
        """
        Run full decoding on *seq*: Viterbi + forward-backward (MAP).

        After calling, the following attributes are populated:

        - ``viterbi_path`` - most-probable state path
        - ``map_path``     - per-position maximum-a-posteriori states
        - ``posterior``     - posterior probability matrix
        - ``log_viterbi_prob``  - log P(seq, best path)
        - ``log_marginal_prob`` - log P(seq)
        """
        self._viterbi_log(seq)
        self._map_decode_log(seq)

    def _viterbi_log(self, seq: Sequence[int]) -> None:
        """Viterbi decoding in log-space."""
        n = len(seq)
        ns = self.ns
        s = [[0.0] * n for _ in range(ns)]
        tb = [[0] * n for _ in range(ns)]

        # Initialisation
        for i in range(ns):
            s[i][0] = self._linit[i] + self._lemis[i][seq[0]]

        # Recursion
        for t in range(1, n):
            for i in range(ns):
                best_k = 0
                best_sc = self._ltrans[0][i] + s[0][t - 1]
                for k in range(1, ns):
                    sc = self._ltrans[k][i] + s[k][t - 1]
                    if sc > best_sc:
                        best_sc = sc
                        best_k = k
                s[i][t] = best_sc + self._lemis[i][seq[t]]
                tb[i][t] = best_k

        # Termination
        best_k = 0
        best_sc = s[0][n - 1] + self._lfprob[0]
        for k in range(1, ns):
            sc = s[k][n - 1] + self._lfprob[k]
            if sc > best_sc:
                best_sc = sc
                best_k = k

        # Traceback
        vit = [0] * n
        vit[n - 1] = best_k
        for t in range(n - 2, -1, -1):
            vit[t] = tb[vit[t + 1]][t + 1]

        self.log_viterbi_prob = best_sc
        if self.num_classes < ns:
            vit = [self.classes[v] for v in vit]
        self.viterbi_path = vit

    def _forward_backward_log(self, seq: Sequence[int]) -> List[List[float]]:
        """
        Forward-backward algorithm in log-space.

        Returns
        -------
        list[list[float]]
            Posterior probability matrix (num_classes x seq_length).
        """
        n = len(seq)
        ns = self.ns
        NEG_INF = float("-inf")

        # Forward pass
        a = [[NEG_INF] * n for _ in range(ns)]
        for i in range(ns):
            a[i][0] = self._linit[i] + self._lemis[i][seq[0]]

        for t in range(1, n):
            for i in range(ns):
                sc = NEG_INF
                for k in range(ns):
                    sc = _log_add_exp(sc, self._ltrans[k][i] + a[k][t - 1])
                a[i][t] = sc + self._lemis[i][seq[t]]

        ltot = NEG_INF
        for i in range(ns):
            ltot = _log_add_exp(ltot, a[i][n - 1] + self._lfprob[i])
        self.log_marginal_prob = ltot

        # Backward pass
        b = [[0.0] * n for _ in range(ns)]
        for i in range(ns):
            b[i][n - 1] = self._lfprob[i]

        for t in range(n - 2, -1, -1):
            for i in range(ns):
                sc = NEG_INF
                for k in range(ns):
                    sc = _log_add_exp(
                        sc,
                        self._ltrans[i][k] + b[k][t + 1] + self._lemis[k][seq[t + 1]],
                    )
                b[i][t] = sc

        # Posterior probabilities
        lpseq = NEG_INF
        for i in range(ns):
            lpseq = _log_add_exp(lpseq, a[i][0] + b[i][0])

        pp = [[0.0] * n for _ in range(ns)]
        for t in range(n):
            for i in range(ns):
                pp[i][t] = math.exp((a[i][t] + b[i][t]) - lpseq)

        if self.num_classes < ns:
            pp = _collapse_posteriors(pp, self.classes, self.num_classes)
        return pp

    def _map_decode_log(self, seq: Sequence[int]) -> None:
        """MAP decoding via forward-backward posteriors."""
        pp = self._forward_backward_log(seq)
        self.posterior = pp
        n = len(seq)
        mp = [0] * n
        for t in range(n):
            for j in range(len(pp)):
                if pp[j][t] > pp[mp[t]][t]:
                    mp[t] = j
        self.map_path = mp

    # ── Subtrellis probability ─────────────────────────────────────────────

    def log_prob_subtrellis(self, seq: Sequence[int]) -> float:
        """
        Compute P(seq stays on subtrellis) / P(seq) in log-space.

        Uses a scaled forward algorithm on the full trellis and again
        restricted to the subtrellis, then divides.
        """
        n = len(seq)
        ns = self.ns
        st = self.subtrellis

        def _scaled_forward(mask_all: bool) -> float:
            a = [[0.0] * n for _ in range(ns)]
            sc = [1.0] * n
            for i in range(ns):
                if mask_all or st[i] == 1:
                    a[i][0] = self.init[i] * self.emis[i][seq[0]]
            for t in range(1, n):
                sf = 0.0
                for i in range(ns):
                    if mask_all or st[i] == 1:
                        s = sum(self.trans[k][i] * a[k][t - 1] for k in range(ns))
                        a[i][t] = s * self.emis[i][seq[t]]
                        sf += a[i][t]
                if sf > 0:
                    sc[t] = 1.0 / sf
                for i in range(ns):
                    a[i][t] *= sc[t]
            return -sum(math.log(c) if c > 0 else 0 for c in sc)

        lpd = _scaled_forward(mask_all=True)
        lpdas = _scaled_forward(mask_all=False)
        return lpdas - lpd

    # ── Graphviz export ────────────────────────────────────────────────────

    def write_dot(self, path: str | Path, show_emissions: bool = True) -> None:
        """
        Write HMM structure in Graphviz DOT format.

        Parameters
        ----------
        path : str or Path
            Output file path.
        show_emissions : bool, default True
            Whether to include emission probability tables.
        """
        with open(path, "w") as out:
            out.write("Digraph G {\n")
            out.write(
                "edge [fontname=Courier, fontsize=8];\n"
                " node [fontname=Courier, fontsize=10]\n"
                " start [label=start, shape=circle, height=0.25, "
                "style=filled, color=grey];\n"
            )
            for i in range(self.ns):
                out.write(
                    f'  n{i} [label="{self.names[i]}", shape=circle, height=1.2];\n'
                    f'  start -> n{i} [label="{self.init[i]:.3f}", color=gray];\n'
                )
            for i in range(self.ns):
                for j in range(self.ns):
                    if self.trans[i][j] > 0:
                        lab = f"{self.trans[i][j]:.3f}"
                        if i == j:
                            port = ":w" if i % 2 == 0 else ":e"
                            out.write(
                                f'  n{i}{port} -> n{j}{port} [label="{lab}", color=gray];\n'
                            )
                        else:
                            out.write(
                                f'  n{i} -> n{j} [label="{lab}", color=gray, constraint=false];\n'
                            )
            if show_emissions:
                for i in range(self.ns):
                    fields_aa = "|".join(AA_NAMES[j] for j in range(1, self.no - 1))
                    fields_pr = "|".join(
                        f"{self.emis[i][j]:.4f}" for j in range(1, self.no - 1)
                    )
                    out.write(
                        f'rec{i} [shape=record, label="'
                        f"{{ <fs> AA|{fields_aa}}}|"
                        f'{{ <f{i}> prob|{fields_pr}}}"];\n'
                    )
                    out.write(f"  n{i} -> rec{i} [style=dashed];\n")
            out.write("}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# HMM construction
# ═══════════════════════════════════════════════════════════════════════════════

def build_prion_hmm(fg: Sequence[float], bg: Sequence[float]) -> HMM:
    """
    Build the two-state prion HMM: background (state 0) <-> PrD-like (state 1).

    Parameters
    ----------
    fg : sequence of float
        Normalised foreground (prion-like) AA frequency vector (length 22).
    bg : sequence of float
        Normalised background AA frequency vector (length 22).

    Returns
    -------
    HMM
        Configured two-state hidden Markov model.
    """
    tmat = [[99.9 / 100, 0.1 / 100], [2.0 / 100, 98.0 / 100]]
    imat = [0.9524, 0.0476]  # stationary distribution
    emat = [_normalize(bg), _normalize(fg)]
    h = HMM(tmat, emat, imat)
    h.subtrellis = [1, 0]
    h.state_chars = ["-", "+"]
    h.set_names(["background", "PrD-like"])
    return h


def build_background_hmm(bg: Sequence[float]) -> HMM:
    """
    Build a one-state (background-only) HMM for likelihood-ratio scoring.

    Technically has two states but both emit from *bg*, and the transition
    matrix keeps the model in state 0 forever.

    Parameters
    ----------
    bg : sequence of float
        Normalised background AA frequency vector (length 22).

    Returns
    -------
    HMM
        Background-only hidden Markov model.
    """
    tmat = [[1, 0], [0, 1]]
    imat = [1, 0]
    emat = [_normalize(bg), _normalize(bg)]
    h = HMM(tmat, emat, imat)
    h.subtrellis = [1, 0]
    h.state_chars = ["-", "+"]
    h.set_names(["background", "also.background"])
    return h


# ═══════════════════════════════════════════════════════════════════════════════
# Disorder Report (FoldIndex + PAPA + PLAAC LLR)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DisorderReport:
    """
    Per-residue FoldIndex, PAPA, and PLAAC log-likelihood-ratio profiles.

    This combines:
    - **FoldIndex** disorder prediction
      (Prilusky et al, Bioinformatics, 2005)
    - **PAPA** prion propensity
      (Toombs et al, MCB 2010 and PNAS 2012)
    - Smoothed PLAAC log-likelihood-ratio profiles

    Attributes
    ----------
    charge, hydro, fi : list[float]
        Smoothed charge, hydropathicity, and FoldIndex profiles (length *n*).
    plaac_llr, papa : list[float]
        Single-smoothed PLAAC LLR and PAPA propensity.
    fix2, plaacllr_x2, papa_x2 : list[float]
        Twice-smoothed (average-of-averages) versions.
    mean_charge, mean_hydro, mean_fi : float
        Whole-protein averages.
    num_disordered_strict : int
        Number of residues predicted disordered (strict definition).
    max_disorder_run : int
        Length of longest disordered run (min 5 residues).
    papa_max_score : float
        Maximum PAPA combo score (distance to decision surface).
    papa_max_prop : float
        Maximum PAPA propensity in a disordered region.
    papa_max_fi, papa_max_llr, papa_max_llr2 : float
        FI and LLR values at the PAPA-optimal centre.
    papa_max_center : int
        0-based index of the centre of the PAPA-optimal window.
    """

    charge: List[float]
    hydro: List[float]
    fi: List[float]
    plaac_llr: List[float]
    papa: List[float]
    fix2: List[float]
    plaacllr_x2: List[float]
    papa_x2: List[float]
    mean_charge: float
    mean_hydro: float
    mean_fi: float
    num_disordered_strict: int
    max_disorder_run: int
    papa_max_score: float
    papa_max_prop: float
    papa_max_fi: float
    papa_max_llr: float
    papa_max_llr2: float
    papa_max_center: int


def compute_disorder(
    aa: Sequence[int],
    ww_fi: int,
    ww_papa: int,
    ww_llr: int,
    llr_weights: Sequence[float],
    papa_weights: Sequence[float],
    *,
    adjust_prolines: bool = True,
    fi_coeffs: Sequence[float] = FOLDINDEX_COEFFS,
) -> DisorderReport:
    """
    Compute FoldIndex, PAPA, and PLAAC LLR profiles for a protein.

    Parameters
    ----------
    aa : sequence of int
        Integer-coded amino acid sequence.
    ww_fi : int
        Window size for FoldIndex smoothing.
    ww_papa : int
        Window size for PAPA smoothing.
    ww_llr : int
        Window size for PLAAC LLR smoothing.
    llr_weights : sequence of float
        Per-AA log-likelihood ratios (length 22).
    papa_weights : sequence of float
        Per-AA PAPA log-odds (length 22).
    adjust_prolines : bool, default True
        Apply PAPA proline adjustment (skip PP / PXP repeats).
    fi_coeffs : sequence of float
        FoldIndex linear-model coefficients ``[hydro_coeff, charge_coeff, intercept]``.

    Returns
    -------
    DisorderReport
        All per-residue profiles and summary statistics.
    """
    n = len(aa)
    props = AA

    # Hydropathicity & charge (single-smoothed)
    hydro_raw = _mapseq(aa, props.hydro_scaled)
    mean_hydro = sum(hydro_raw) / n if n else 0.0
    hydro = sliding_average(hydro_raw, ww_fi, shrink=True)

    charge_raw = _mapseq(aa, props.charge)
    mean_charge = sum(charge_raw) / n if n else 0.0
    charge = sliding_average(charge_raw, ww_fi, shrink=True)

    # FoldIndex
    a_h, a_c, intercept = fi_coeffs
    mean_fi = intercept + a_c * abs(mean_charge) + a_h * mean_hydro
    fi = [a_h * hydro[i] + a_c * abs(charge[i]) + intercept for i in range(n)]

    # PLAAC LLR (single-smoothed)
    llr_raw = _mapseq(aa, llr_weights)
    plaac_llr = sliding_average(llr_raw, ww_llr, shrink=True)

    # PAPA propensity (single-smoothed, with optional proline adjustment)
    papa_raw = _mapseq(aa, papa_weights)
    if adjust_prolines:
        papa_single = sliding_average_papa(papa_raw, ww_papa, aa)
    else:
        papa_single = sliding_average(papa_raw, ww_papa, shrink=True)

    # Twice-smoothed (average of averages)
    papa_x2 = sliding_average(papa_single, ww_papa, shrink=False, weighted=True)
    plaacllr_x2 = sliding_average(plaac_llr, ww_llr, shrink=False, weighted=True)
    fix2 = sliding_average(fi, ww_fi, shrink=False, weighted=True)

    # Count disordered residues (strict: min-run >= 5)
    min_len = 5
    halfw = (ww_fi - 1) // 2
    if halfw > n // 2:
        halfw = n // 2
    num_dis = 0
    max_run = 0
    i = halfw
    while i < n - halfw:
        if fi[i] < 0:
            start = i
            i += 1
            while i < n - halfw and fi[i] < 0:
                i += 1
            stop = i - 1
            seg = stop - start + 1
            if start == halfw:
                start = 0
            if stop == n - halfw - 1:
                stop = n - 1
            seg = stop - start + 1
            if seg >= min_len:
                num_dis += seg
                if seg > max_run:
                    max_run = seg
        else:
            i += 1

    # PAPA scoring - find window with max PAPA propensity in disordered region
    hw2 = (ww_papa - 1) // 2
    best_score = float("-inf")
    best_center = -1
    for k in range(hw2, n - hw2):
        val = papa_x2[k]
        if not math.isnan(val) and val > best_score:
            if not math.isnan(fix2[k]) and fix2[k] < 0:
                best_center = k
                best_score = val

    if best_center >= 0:
        pmax_prop = papa_x2[best_center]
        pmax_fi = fix2[best_center]
        pmax_llr2 = plaacllr_x2[best_center]
        pmax_llr = plaac_llr[best_center]
    else:
        pmax_prop = pmax_fi = pmax_llr = pmax_llr2 = float("nan")

    return DisorderReport(
        charge=charge,
        hydro=hydro,
        fi=fi,
        plaac_llr=plaac_llr,
        papa=papa_single,
        fix2=fix2,
        plaacllr_x2=plaacllr_x2,
        papa_x2=papa_x2,
        mean_charge=mean_charge,
        mean_hydro=mean_hydro,
        mean_fi=mean_fi,
        num_disordered_strict=num_dis,
        max_disorder_run=max_run,
        papa_max_score=best_score,
        papa_max_prop=pmax_prop,
        papa_max_fi=pmax_fi,
        papa_max_llr=pmax_llr,
        papa_max_llr2=pmax_llr2,
        papa_max_center=best_center,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Run configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PLAACConfig:
    """
    All user-configurable parameters for a PLAAC run.

    These are populated from command-line arguments (see :func:`build_parser`).

    Attributes
    ----------
    input_file : str
        Path to the input FASTA file.
    bg_fasta : str
        Path to a FASTA file for computing background AA frequencies.
    bg_freq_file : str
        Path to a pre-computed background AA frequency file.
    fg_freq_file : str
        Path to a foreground AA frequency file.
    plot_list : str
        Path to a gene list for per-residue plot output, or 'all'.
    hmm_dot_file : str
        Path for Graphviz DOT export of the HMM.
    core_length : int
        Minimum contiguous prion-like domain length.
    window_fi : int
        Window size for FoldIndex smoothing.
    window_papa : int
        Window size for PAPA smoothing.
    alpha : float
        Mixing weight for S. cerevisiae vs. input background frequencies.
    print_headers : bool
        Whether to print column documentation.
    skip_parameters : bool
        Whether to skip printing run-time parameters.
    adjust_prolines : bool
        Whether to apply PAPA proline adjustment.
    """

    input_file: str = ""
    bg_fasta: str = ""
    bg_freq_file: str = ""
    fg_freq_file: str = ""
    plot_list: str = ""
    hmm_dot_file: str = ""
    core_length: int = 60
    window_fi: int = 41
    window_papa: int = 41
    alpha: float = 1.0
    print_headers: bool = False
    skip_parameters: bool = False
    adjust_prolines: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring result (public API data structure)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PRDRegion:
    """
    A single prion-like domain (PRD) region extracted from the Viterbi path.

    Each contiguous run of PrD-like states in the Viterbi path that is
    long enough to contain a CORE (≥ *core_length*) produces one
    ``PRDRegion``.  Regions are ranked by their ``core_score``.

    All start/end positions are **1-based** to match the CLI output.

    Attributes
    ----------
    prd_start, prd_end : int
        1-based start/end of the full Viterbi PrD run.
    prd_length : int
        Length of the PrD run.
    prd_seq : str
        Amino acid sequence of the full PrD run.
    prd_score : float
        Sum of LLR values over the full PrD run.
    core_start, core_end : int
        1-based start/end of the best-scoring CORE window within this run.
    core_length : int
        Length of the CORE window.
    core_seq : str
        Amino acid sequence of the CORE window.
    core_score : float
        Sum of LLR values in the CORE window.  NaN if the run is shorter
        than the requested CORE length.
    """

    prd_start: int
    prd_end: int
    prd_length: int
    prd_seq: str
    prd_score: float

    core_start: int
    core_end: int
    core_length: int
    core_seq: str
    core_score: float


@dataclass
class PLAACResult:
    """
    Complete PLAAC scoring result for a single protein sequence.

    Contains all the same fields produced by the CLI's tab-delimited output,
    organised into logical groups.  Returned by :func:`score_sequence`.

    All start/end positions are **1-based** to match the CLI output.

    Attributes
    ----------
    sequence : str
        The input amino acid sequence (with any trailing ``*`` stripped).
    protein_length : int
        Length of the sequence.

    mw_score : int
        Michelitsch-Weissman Q/N enrichment score (max N+Q in ≤80-AA window).
    mw_start, mw_end : int
        1-based start/end of the MW window.
    mw_length : int
        Length of the MW window.

    llr : float
        Maximum sum of PLAAC log-likelihood ratios in a window of size
        *core_length*.  NaN if the protein is shorter than *core_length*.
    llr_start, llr_end : int
        1-based start/end of the LLR window.
    llr_length : int
        Length of the LLR window.
    nllr : float
        Normalised LLR (= llr / llr_length).

    viterbi_max_run : int
        Longest contiguous run of PrD-like states in the Viterbi path.
    core_score : float
        Maximum LLR sum in a window of size *core_length* within the Viterbi
        PrD region.  NaN if viterbi_max_run < core_length.
    core_start, core_end : int
        1-based start/end of the CORE region.
    core_length : int
        Length of the CORE region.
    core_seq : str
        Amino acid sequence of the CORE region (``"-"`` if none found).

    prd_score : float
        LLR sum over the full Viterbi PrD region containing the CORE.
    prd_start, prd_end : int
        1-based start/end of the full PRD region.
    prd_length : int
        Length of the full PRD region.
    prd_seq : str
        Amino acid sequence of the full PRD region (``"-"`` if none found).
    prd_start_seq : str
        First 15 AA of the PRD region.
    prd_end_seq : str
        Last 15 AA of the PRD region.

    hmm_all : float
        Log-likelihood ratio for full HMM vs. background HMM.
    hmm_vit : float
        Log-likelihood ratio for Viterbi path vs. background HMM.

    fi_num_disordered : int
        Number of AAs predicted disordered by FoldIndex (runs ≥ 5).
    fi_mean_hydro : float
        Mean hydropathicity ⟨H⟩.
    fi_mean_charge : float
        Mean absolute charge ⟨R⟩.
    fi_mean_combo : float
        FoldIndex whole-protein disorder score.
    fi_max_run : int
        Longest disordered run.

    papa_combo : float
        Signed distance to the PAPA decision surface.
    papa_prop : float
        Maximum PAPA propensity in a disordered region.
    papa_fi : float
        FI score (avg-of-avg) at PAPA centre.
    papa_llr : float
        PLAAC LLR (avg) at PAPA centre.
    papa_llr2 : float
        PLAAC LLR (avg-of-avg) at PAPA centre.
    papa_center : int
        1-based index of PAPA centre.
    papa_seq : str
        Amino acid sequence of the PAPA window.

    disorder : DisorderReport
        Full per-residue disorder profiles (for downstream analysis / plotting).
    viterbi_path : list[int]
        Per-residue Viterbi state assignments (0 = background, 1 = PrD-like).
    map_path : list[int]
        Per-residue MAP state assignments.
    posterior : list[list[float]]
        Per-class posterior probabilities (num_classes × protein_length).
    prd_regions : list[PRDRegion]
        All PRD regions found in the Viterbi path, sorted by descending
        ``core_score``.  The first element (if any) corresponds to the
        same region reported by the top-level ``prd_*`` / ``core_*`` fields.
        Empty list if no PrD-like runs were found.
    """

    # Input
    sequence: str
    protein_length: int

    # Michelitsch-Weissman
    mw_score: int
    mw_start: int
    mw_end: int
    mw_length: int

    # LLR
    llr: float
    llr_start: int
    llr_end: int
    llr_length: int
    nllr: float

    # Viterbi / CORE
    viterbi_max_run: int
    core_score: float
    core_start: int
    core_end: int
    core_length: int
    core_seq: str

    # PRD
    prd_score: float
    prd_start: int
    prd_end: int
    prd_length: int
    prd_seq: str
    prd_start_seq: str
    prd_end_seq: str

    # HMM
    hmm_all: float
    hmm_vit: float

    # FoldIndex
    fi_num_disordered: int
    fi_mean_hydro: float
    fi_mean_charge: float
    fi_mean_combo: float
    fi_max_run: int

    # PAPA
    papa_combo: float
    papa_prop: float
    papa_fi: float
    papa_llr: float
    papa_llr2: float
    papa_center: int
    papa_seq: str

    # Full per-residue data
    disorder: DisorderReport
    viterbi_path: List[int]
    map_path: List[int]
    posterior: List[List[float]]

    # All PRD regions
    prd_regions: List["PRDRegion"]


# ═══════════════════════════════════════════════════════════════════════════════
# Internal model builder (shared by CLI and API)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class _PLAACModel:
    """Internal bundle of the fully resolved model components."""
    fg: List[float]
    bg: List[float]
    llr: List[float]
    hmm1: HMM
    hmm0: HMM
    alpha: float
    core_length: int
    window_fi: int
    window_papa: int
    adjust_prolines: bool


def _build_model(
    *,
    alpha: float = 1.0,
    core_length: int = 60,
    window_fi: int = 41,
    window_papa: int = 41,
    adjust_prolines: bool = True,
    bg_freqs: Optional[Sequence[float]] = None,
    fg_freqs: Optional[Sequence[float]] = None,
) -> _PLAACModel:
    """
    Build the fully resolved PLAAC model (HMMs + LLRs) from parameters.

    Parameters
    ----------
    alpha : float
        Mixing weight for S. cerevisiae background (1.0 = pure S. cer.).
    core_length : int
        Minimum contiguous prion-like domain length.
    window_fi : int
        FoldIndex window size.
    window_papa : int
        PAPA window size.
    adjust_prolines : bool
        Apply PAPA proline adjustment.
    bg_freqs : sequence of float, optional
        22-element background AA frequency vector.  If *None*, defaults to
        S. cerevisiae frequencies (``alpha=1.0``).
    fg_freqs : sequence of float, optional
        22-element foreground (prion-like) AA frequency vector.  Defaults to
        the 28-domain S. cerevisiae frequencies.

    Returns
    -------
    _PLAACModel
        Bundle of normalised fg/bg vectors, LLRs, and constructed HMMs.
    """
    props = AA
    fg_freq: List[float] = list(fg_freqs) if fg_freqs is not None else list(props.prd_freq_scer_28)
    bg_scer: List[float] = _normalize(list(props.bg_freq_scer))

    bg_raw: List[float]
    if bg_freqs is not None:
        bg_raw = list(bg_freqs)
    else:
        bg_raw = list(props.bg_freq_scer)

    if not 0 <= alpha <= 1:
        alpha = 1.0

    fg_freq[0] = fg_freq[21] = 0.0
    fg_freq = _normalize(fg_freq)

    bg_raw[0] = bg_raw[21] = 0.0
    bg_this = _normalize(bg_raw)

    bg_combo = _normalize(
        [alpha * bg_scer[i] + (1 - alpha) * bg_this[i] for i in range(22)]
    )

    eps = 1e-5
    fg_freq[0] = fg_freq[21] = eps
    bg_combo[0] = bg_combo[21] = eps
    fg = _normalize(fg_freq)
    bg = _normalize(bg_combo)

    llr_vec = [0.0] * 22
    for j in range(1, 21):
        llr_vec[j] = math.log(fg[j] / bg[j])

    return _PLAACModel(
        fg=fg,
        bg=bg,
        llr=llr_vec,
        hmm1=build_prion_hmm(fg, bg),
        hmm0=build_background_hmm(bg),
        alpha=alpha,
        core_length=core_length,
        window_fi=window_fi,
        window_papa=window_papa,
        adjust_prolines=adjust_prolines,
    )


def _score_single_sequence(
    aa: List[int],
    seq_str: str,
    model: _PLAACModel,
) -> PLAACResult:
    """
    Score a single integer-coded sequence and return a :class:`PLAACResult`.

    This is the shared core used by both the CLI (:func:`score_all_fastas`)
    and the public API (:func:`score_sequence`).
    """
    llr = model.llr
    hmm1 = model.hmm1
    hmm0 = model.hmm0
    core_len = model.core_length

    # N+Q mask for Michelitsh-Weissman score
    qn_mask = [0.0] * 22
    qn_mask[12] = 1.0  # N
    qn_mask[14] = 1.0  # Q

    lod_papa1 = AA.lod_papa1
    ww_llr = model.window_papa

    # -- MW score --
    mw_size = min(80, len(aa))
    mw = highest_scoring_subseq(_mapseq(aa, qn_mask), mw_size, mw_size)

    # -- LLR score --
    llr_mapped = _mapseq(aa, llr)
    hs_llr = highest_scoring_subseq(llr_mapped, core_len, core_len)

    # -- HMM --
    hmm1.decode(aa)
    hmm0.decode(aa)
    hmm_all = hmm1.log_marginal_prob - hmm0.log_marginal_prob
    hmm_vit = hmm1.log_viterbi_prob - hmm0.log_viterbi_prob

    # -- Disorder --
    dr = compute_disorder(
        aa, model.window_fi, model.window_papa, ww_llr,
        llr, lod_papa1, adjust_prolines=model.adjust_prolines,
    )

    # -- CORE / PRD from Viterbi path --
    vp = hmm1.viterbi_path
    longest_prd = _longest_run(vp)

    big_neg = -1_000_000.0
    masked = [llr[aa[i]] if vp[i] == 1 else big_neg for i in range(len(aa))]
    hs_core = highest_scoring_subseq(masked, core_len, core_len)

    core_start = int(hs_core.start)
    core_stop = int(hs_core.end)
    aa_start = core_start
    aa_stop = core_stop
    prd_ints: List[int] = []
    prd_score = 0.0

    if hs_core.score > big_neg / 2:
        while aa_start >= 0 and vp[aa_start] == 1:
            aa_start -= 1
        aa_start += 1
        while aa_stop < len(vp) and vp[aa_stop] == 1:
            aa_stop += 1
        aa_stop -= 1
        prd_ints = list(aa[aa_start: aa_stop + 1])
        prd_score = sum(llr[a] for a in prd_ints)
    else:
        hs_core = HSSResult(start=-1, end=-2, score=float("nan"))
        aa_start = -1
        aa_stop = -2
        core_start = -1
        core_stop = -2

    # Derived sequences
    has_prd = (aa_stop - aa_start + 1) >= core_len
    core_seq_str = ints_to_seq(aa[core_start: core_stop + 1]) if has_prd else "-"
    prd_start_seq = ints_to_seq(aa[aa_start: aa_start + 15]) if has_prd else "-"
    prd_end_seq = ints_to_seq(aa[max(0, aa_stop - 14): aa_stop + 1]) if has_prd else "-"
    prd_seq_str = ints_to_seq(prd_ints) if has_prd else "-"

    # -- Extract ALL PRD regions from the Viterbi path --
    prd_regions: List[PRDRegion] = []
    for run_start, run_end in _extract_prd_runs(vp):
        run_len = run_end - run_start + 1
        run_ints = list(aa[run_start: run_end + 1])
        run_score = sum(llr[a] for a in run_ints)
        run_seq = ints_to_seq(run_ints)

        # Find the best CORE within this run
        run_masked = [llr[aa[j]] for j in range(run_start, run_end + 1)]
        if run_len >= core_len:
            run_hs = highest_scoring_subseq(run_masked, core_len, core_len)
            r_core_start = run_start + run_hs.start
            r_core_end = run_start + run_hs.end
            r_core_score = _inf_to_nan(run_hs.score)
            r_core_seq = ints_to_seq(aa[r_core_start: r_core_end + 1])
            r_core_len = r_core_end - r_core_start + 1
        else:
            r_core_start = run_start
            r_core_end = run_end
            r_core_score = float("nan")
            r_core_seq = "-"
            r_core_len = 0

        prd_regions.append(PRDRegion(
            prd_start=run_start + 1,
            prd_end=run_end + 1,
            prd_length=run_len,
            prd_seq=run_seq,
            prd_score=run_score,
            core_start=r_core_start + 1,
            core_end=r_core_end + 1,
            core_length=r_core_len,
            core_seq=r_core_seq,
            core_score=r_core_score,
        ))

    # Sort by core_score descending (NaN-safe: NaN sorts to end)
    prd_regions.sort(
        key=lambda r: (not math.isnan(r.core_score), r.core_score),
        reverse=True,
    )

    # PAPA window sequence
    papa_aa = ints_to_seq(_subvec(
        aa, dr.papa_max_center - model.window_papa // 2,
        dr.papa_max_center + model.window_papa // 2,
    ))

    mw_sc = _inf_to_nan(mw.score)
    llr_sc = _inf_to_nan(hs_llr.score)
    nllr = llr_sc / hs_llr.length if hs_llr.length > 0 and not math.isnan(llr_sc) else float("nan")
    core_sc = _inf_to_nan(hs_core.score)

    return PLAACResult(
        sequence=seq_str,
        protein_length=len(aa),
        # MW
        mw_score=int(mw_sc) if not math.isnan(mw_sc) else 0,
        mw_start=int(mw.start + 1),
        mw_end=int(mw.end + 1),
        mw_length=mw.length,
        # LLR
        llr=llr_sc,
        llr_start=int(hs_llr.start + 1),
        llr_end=int(hs_llr.end + 1),
        llr_length=hs_llr.length,
        nllr=nllr,
        # CORE
        viterbi_max_run=longest_prd,
        core_score=core_sc,
        core_start=core_start + 1,
        core_end=core_stop + 1,
        core_length=core_stop - core_start + 1,
        core_seq=core_seq_str,
        # PRD
        prd_score=prd_score,
        prd_start=aa_start + 1,
        prd_end=aa_stop + 1,
        prd_length=aa_stop - aa_start + 1,
        prd_seq=prd_seq_str,
        prd_start_seq=prd_start_seq,
        prd_end_seq=prd_end_seq,
        # HMM
        hmm_all=hmm_all,
        hmm_vit=hmm_vit,
        # FoldIndex
        fi_num_disordered=dr.num_disordered_strict,
        fi_mean_hydro=dr.mean_hydro,
        fi_mean_charge=dr.mean_charge,
        fi_mean_combo=dr.mean_fi,
        fi_max_run=dr.max_disorder_run,
        # PAPA
        papa_combo=_inf_to_nan(dr.papa_max_score),
        papa_prop=dr.papa_max_prop,
        papa_fi=dr.papa_max_fi,
        papa_llr=dr.papa_max_llr,
        papa_llr2=dr.papa_max_llr2,
        papa_center=dr.papa_max_center + 1,
        papa_seq=papa_aa,
        # Full per-residue data
        disorder=dr,
        viterbi_path=list(vp),
        map_path=list(hmm1.map_path),
        posterior=[list(row) for row in hmm1.posterior] if hmm1.posterior else [],
        # All PRD regions
        prd_regions=prd_regions,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def _freqs_dict_to_list(freqs: Dict[str, float]) -> List[float]:
    """
    Convert an amino-acid frequency dictionary to the 22-element list
    expected by :func:`_build_model`.

    Keys are one-letter amino-acid codes (e.g. ``"A"``, ``"N"``).
    Any of the 20 standard residues may be provided; missing residues
    default to ``0.0``.  The special positions ``X`` (index 0, unknown)
    and ``*`` (index 21, stop) are always set to ``0.0``.

    Raises
    ------
    ValueError
        If a key is not a recognised one-letter amino acid code.
    """
    vec = [0.0] * 22
    for aa_letter, freq in freqs.items():
        aa_upper = aa_letter.upper()
        if aa_upper not in _AA_TO_INT:
            raise ValueError(
                f"Unrecognised amino acid code {aa_letter!r}. "
                f"Valid one-letter codes: {', '.join(AA_NAMES[1:21])}"
            )
        idx = _AA_TO_INT[aa_upper]
        if idx == 0 or idx == 21:
            # Silently ignore X and * entries
            continue
        vec[idx] = float(freq)
    return vec


def score_sequence(
    sequence: str,
    *,
    alpha: float = 1.0,
    core_length: int = 60,
    window_fi: int = 41,
    window_papa: int = 41,
    adjust_prolines: bool = True,
    bg_freqs: Optional[Dict[str, float]] = None,
    fg_freqs: Optional[Dict[str, float]] = None,
) -> PLAACResult:
    """
    Score a single amino acid sequence for prion-like composition.

    This is the main **programmatic API** for PLAAC.  It accepts a raw
    amino acid string and returns a :class:`PLAACResult` containing every
    score and annotation that the CLI produces.

    Parameters
    ----------
    sequence : str
        One-letter amino acid sequence (e.g. ``"MFKSG..."``).  A trailing
        ``*`` stop codon is automatically stripped.
    alpha : float, default 1.0
        Mixing weight for S. cerevisiae vs. custom background frequencies.
        ``1.0`` uses pure S. cerevisiae background; ``0.0`` uses only the
        frequencies supplied via *bg_freqs*.
    core_length : int, default 60
        Minimum contiguous prion-like domain length.
    window_fi : int, default 41
        Window size for FoldIndex disorder smoothing.
    window_papa : int, default 41
        Window size for PAPA propensity smoothing.
    adjust_prolines : bool, default True
        Apply PAPA proline-adjustment (skip PP / PXP repeats).
    bg_freqs : dict[str, float], optional
        Background amino-acid frequency dictionary. Keys are one-letter
        amino acid codes (e.g. ``"A"``, ``"N"``, ``"Q"``), values are
        the corresponding frequencies.  Only the 20 standard residues
        (``A C D E F G H I K L M N P Q R S T V W Y``) should be
        provided; any missing residues default to 0.0.

        Example::

            bg_freqs={"A": 0.05, "N": 0.04, "Q": 0.04, ...}

        Defaults to S. cerevisiae proteome frequencies when ``None``.
    fg_freqs : dict[str, float], optional
        Foreground (prion-like) amino-acid frequency dictionary, same
        format as *bg_freqs*.  Defaults to the Alberti et al. 28-domain
        S. cerevisiae prion frequencies when ``None``.

    Returns
    -------
    PLAACResult
        Dataclass with all PLAAC scores, HMM paths, disorder profiles,
        and extracted sequence motifs.

    Examples
    --------
    >>> import plaac
    >>> result = plaac.score_sequence("MFKSGNQNN" * 20)
    >>> result.protein_length
    180
    >>> result.core_score  # NaN or float
    ...
    >>> result.fi_mean_hydro
    ...
    >>> # Full per-residue disorder profile:
    >>> len(result.disorder.fi) == result.protein_length
    True
    """
    seq_str = sequence.rstrip("*")
    aa = seq_to_ints(seq_str)
    if len(aa) == 0:
        raise ValueError("Empty sequence after stripping stop codons.")

    # parse bg/fg frequencies 
    bg_list = _freqs_dict_to_list(bg_freqs) if bg_freqs is not None else None
    fg_list = _freqs_dict_to_list(fg_freqs) if fg_freqs is not None else None

    # default behavior over-rides non-standard background frequencies if alpha=1.0,
    # so warn user if they try this
    if bg_list is not None and alpha == 1.0:
        raise ValueError("alpha=1.0 means S. cerevisiae background is 100% weighted")

    model = _build_model(
        alpha=alpha,
        core_length=core_length,
        window_fi=window_fi,
        window_papa=window_papa,
        adjust_prolines=adjust_prolines,
        bg_freqs=bg_list,
        fg_freqs=fg_list,
    )
    return _score_single_sequence(aa, seq_str, model)


def score_sequences(
    sequences: Dict[str, str],
    *,
    alpha: float = 1.0,
    core_length: int = 60,
    window_fi: int = 41,
    window_papa: int = 41,
    adjust_prolines: bool = True,
    bg_freqs: Optional[Dict[str, float]] = None,
    fg_freqs: Optional[Dict[str, float]] = None,
) -> Dict[str, PLAACResult]:
    """
    Score multiple sequences, sharing a single HMM build.

    This is more efficient than calling :func:`score_sequence` in a loop
    because the HMMs and frequency tables are constructed only once.

    Parameters
    ----------
    sequences : dict[str, str]
        Mapping of ``{name: amino_acid_sequence}``.
    alpha, core_length, window_fi, window_papa, adjust_prolines,
    bg_freqs, fg_freqs
        Same as :func:`score_sequence`.

    Returns
    -------
    dict[str, PLAACResult]
        One :class:`PLAACResult` per input name, keyed by the same names.

    Examples
    --------
    >>> results = plaac.score_sequences({
    ...     "protein_A": "MFKSGNQNN" * 20,
    ...     "protein_B": "ACDEFGHIKLMNPQRSTVWY" * 10,
    ... })
    >>> results["protein_A"].protein_length
    180
    """
    bg_list = _freqs_dict_to_list(bg_freqs) if bg_freqs is not None else None
    fg_list = _freqs_dict_to_list(fg_freqs) if fg_freqs is not None else None

    model = _build_model(
        alpha=alpha,
        core_length=core_length,
        window_fi=window_fi,
        window_papa=window_papa,
        adjust_prolines=adjust_prolines,
        bg_freqs=bg_list,
        fg_freqs=fg_list,
    )
    results: Dict[str, PLAACResult] = {}
    for name, seq in sequences.items():
        seq_str = seq.rstrip("*")
        aa = seq_to_ints(seq_str)
        if len(aa) == 0:
            continue
        results[name] = _score_single_sequence(aa, seq_str, model)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Output routines
# ═══════════════════════════════════════════════════════════════════════════════

_SUMMARY_HEADER = (
    "SEQid\tMW\tMWstart\tMWend\tMWlen\t"
    "LLR\tLLRstart\tLLRend\tLLRlen\tNLLR\t"
    "VITmaxrun\tCOREscore\tCOREstart\tCOREend\tCORElen\t"
    "PRDscore\tPRDstart\tPRDend\tPRDlen\tPROTlen\t"
    "HMMall\tHMMvit\t"
    "COREaa\tSTARTaa\tENDaa\tPRDaa\t"
    "FInumaa\tFImeanhydro\tFImeancharge\tFImeancombo\tFImaxrun\t"
    "PAPAcombo\tPAPAprop\tPAPAfi\tPAPAllr\tPAPAllr2\tPAPAcen\tPAPAaa"
)


def _print_column_docs() -> None:
    """Print documentation for each output column to stdout."""
    docs = [
        ("SEQid", "sequence name from fasta file"),
        ("MW", "Michelitsh-Weissman score - max N+Q count in window of <=80 AA"),
        ("MWstart", "1-based start of MW window"),
        ("MWend", "1-based end of MW window"),
        ("MWlen", "length of MW window"),
        ("LLR", "max sum of PLAAC LLRs in window of size c [NaN if PROTlen<c]"),
        ("LLRstart", "1-based start of LLR window"),
        ("LLRend", "1-based end of LLR window"),
        ("LLRlen", "length of LLR window"),
        ("NLLR", "normalised LLR = LLR / LLRlen"),
        ("VITmaxrun", "max length of consecutive PrD states in Viterbi parse"),
        ("COREscore", "max LLR sum in window of size c within Viterbi PrD [NaN if VITmaxrun<c]"),
        ("COREstart", "1-based start of CORE window"),
        ("COREend", "1-based end of CORE window"),
        ("CORElen", "length of CORE window"),
        ("PRDscore", "LLR sum over full Viterbi PrD region containing the CORE"),
        ("PRDstart", "1-based start of PRD region"),
        ("PRDend", "1-based end of PRD region"),
        ("PRDlen", "length of PRD region"),
        ("PROTlen", "protein length excluding terminal stop"),
        ("HMMall", "log-likelihood ratio for full HMM vs background HMM"),
        ("HMMvit", "log-likelihood ratio for Viterbi path vs background HMM"),
        ("COREaa", "AA sequence of CORE region"),
        ("STARTaa", "first 15 AA of PRD region"),
        ("ENDaa", "last 15 AA of PRD region"),
        ("PRDaa", "full AA sequence of PRD region"),
        ("FInumaa", "number of AAs predicted disordered by FoldIndex (runs >= 5)"),
        ("FImeanhydro", "mean hydropathicity <H>"),
        ("FImeancharge", "mean absolute charge <R>"),
        ("FImeancombo", "FoldIndex whole-protein disorder score"),
        ("FImaxrun", "longest disordered run"),
        ("PAPAcombo", "signed distance to PAPA decision surface"),
        ("PAPAprop", "max PAPA propensity in disordered region"),
        ("PAPAfi", "FI score (avg-of-avg) at PAPA centre"),
        ("PAPAllr", "PLAAC LLR (avg) at PAPA centre"),
        ("PAPAllr2", "PLAAC LLR (avg-of-avg) at PAPA centre"),
        ("PAPAcen", "1-based index of PAPA centre"),
        ("PAPAaa", "AA sequence of PAPA window"),
    ]
    print("############################ Description of output columns "
          "############################")
    for col, desc in docs:
        print(f"## {col}: {desc}")
    print("##########################################################"
          "#########################")


def score_all_fastas(
    cfg: PLAACConfig,
    fg: List[float],
    bg: List[float],
    llr: List[float],
    hmm1: HMM,
    hmm0: HMM,
) -> None:
    """
    Score every sequence in the input FASTA and write one summary row per
    protein to stdout.

    Parameters
    ----------
    cfg : PLAACConfig
        Run configuration.
    fg, bg : list of float
        Normalised foreground / background AA frequency vectors.
    llr : list of float
        Per-AA log-likelihood ratios.
    hmm1 : HMM
        Two-state prion HMM.
    hmm0 : HMM
        Background-only HMM (for LLR scoring).
    """
    if cfg.print_headers:
        _print_column_docs()

    print(_SUMMARY_HEADER)

    model = _PLAACModel(
        fg=fg, bg=bg, llr=llr, hmm1=hmm1, hmm0=hmm0,
        alpha=cfg.alpha, core_length=cfg.core_length,
        window_fi=cfg.window_fi, window_papa=cfg.window_papa,
        adjust_prolines=cfg.adjust_prolines,
    )

    for rec in read_fasta(cfg.input_file):
        seq_str = rec.sequence.rstrip("*")
        aa = seq_to_ints(seq_str)
        if len(aa) < 1:
            continue

        r = _score_single_sequence(aa, seq_str, model)

        # -- Format output --
        w = sys.stdout.write
        mw_disp = r.mw_score if r.mw_score else "NaN"
        w(f"{rec.name}\t{mw_disp}\t{r.mw_start}\t{r.mw_end}\t{r.mw_length}\t")
        w(f"{r.llr:.3f}\t{r.llr_start}\t{r.llr_end}\t{r.llr_length}\t{r.nllr:.3f}\t")
        w(f"{r.viterbi_max_run}\t{r.core_score:.3f}\t"
          f"{r.core_start}\t{r.core_end}\t{r.core_length}\t"
          f"{r.prd_score:.3f}\t{r.prd_start}\t{r.prd_end}\t"
          f"{r.prd_length}\t{r.protein_length}\t"
          f"{r.hmm_all:.3f}\t{r.hmm_vit:.3f}\t")
        w(f"{r.core_seq}\t{r.prd_start_seq}\t{r.prd_end_seq}\t{r.prd_seq}")
        w(f"\t{r.fi_num_disordered}\t{r.fi_mean_hydro:.3f}\t"
          f"{r.fi_mean_charge:.3f}\t{r.fi_mean_combo:.3f}\t{r.fi_max_run}\t"
          f"{r.papa_combo:.3f}\t{r.papa_prop:.3f}\t"
          f"{r.papa_fi:.3f}\t{r.papa_llr:.3f}\t{r.papa_llr2:.3f}\t"
          f"{r.papa_center}\t{r.papa_seq}")
        print()


def plot_some_fastas(
    cfg: PLAACConfig,
    fg: List[float],
    bg: List[float],
    llr: List[float],
    hmm1: HMM,
    hmm0: HMM,
) -> None:
    """
    Produce per-residue output for selected sequences (for generating plots).

    Parameters
    ----------
    cfg : PLAACConfig
        Run configuration (``cfg.plot_list`` selects which sequences to plot).
    fg, bg, llr : list of float
        Frequency vectors and log-likelihood ratios.
    hmm1, hmm0 : HMM
        Prion HMM and background HMM.
    """
    plot_all = cfg.plot_list == "all"
    aliases: Dict[str, str] = {}
    ordering: Dict[str, str] = {}
    if not plot_all:
        aliases, ordering = read_gene_list(cfg.plot_list)

    lod_papa1 = AA.lod_papa1
    ww_llr = cfg.window_papa

    # Header
    hdr = ("ORDER\tSEQid\tAANUM\tAA\tVIT\tMAP\tCHARGE\tHYDRO\tFI\t"
           "PLAAC\tPAPA\tFIx2\tPLAACx2\tPAPAx2")
    for i in range(hmm1.num_classes):
        hdr += f"\tHMM.{hmm1.class_names[i]}"
    print(hdr)

    gene_count = 1
    for rec in read_fasta(cfg.input_file):
        nm = rec.name
        if not (plot_all or nm in aliases or f">{nm}" in aliases):
            continue

        gene_id = str(gene_count)
        if nm in aliases:
            nm = aliases[rec.name]
        if rec.name in ordering:
            gene_id = ordering[rec.name]

        seq_str = rec.sequence.rstrip("*")
        aa = seq_to_ints(seq_str)
        gene_count += 1

        hmm1.decode(aa)
        dr = compute_disorder(
            aa, cfg.window_fi, cfg.window_papa, ww_llr,
            llr, lod_papa1, adjust_prolines=cfg.adjust_prolines,
        )

        w = sys.stdout.write
        for i in range(len(aa)):
            w(f"{gene_id}\t{nm}\t{i + 1}\t{AA_NAMES[aa[i]]}\t"
              f"{hmm1.viterbi_path[i]}\t{hmm1.map_path[i]}\t"
              f"{dr.charge[i]:.4f}\t{dr.hydro[i]:.4f}\t{dr.fi[i]:.8f}\t"
              f"{dr.plaac_llr[i]:.4f}\t{dr.papa[i]:.8f}\t"
              f"{dr.fix2[i]:.8f}\t{dr.plaacllr_x2[i]:.4f}\t{dr.papa_x2[i]:.8f}")
            for j in range(len(hmm1.posterior)):
                w(f"\t{hmm1.posterior[j][i]:.4f}")
            print()
        print("########################################################")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Fully configured parser for PLAAC options.
    """
    parser = argparse.ArgumentParser(
        prog="plaac",
        description=(
            "PLAAC: Prion-Like Amino Acid Composition - identify prion-like "
            "domains in protein sequences."
        ),
        epilog=(
            "Example: python plaac.py -i input.fa > output.txt\n\n"
            "To make plots, use -p to specify a gene list:\n"
            "  python plaac.py -i input.fa -p genelist.txt > plot_data.txt\n"
            "  Rscript plaac_plot.r plot_data.txt plotname.pdf"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-i", "--input", dest="input_file", default="",
        help="Input protein FASTA file (required for scoring).",
    )
    parser.add_argument(
        "-b", "--bg-fasta", dest="bg_fasta", default="",
        help=(
            "Background FASTA file for computing AA frequencies. "
            "Ignored if -B is used. Defaults to input file if omitted."
        ),
    )
    parser.add_argument(
        "-B", "--bg-freqs", dest="bg_freq_file", default="",
        help="File of pre-computed background AA frequencies (one per line).",
    )
    parser.add_argument(
        "-F", "--fg-freqs", dest="fg_freq_file", default="",
        help=(
            "File of prion-like foreground AA frequencies. "
            "Defaults to S. cerevisiae 28-domain frequencies."
        ),
    )
    parser.add_argument(
        "-c", "--core-length", type=int, default=60,
        help="Minimal contiguous prion-like domain length for HMM. Default: 60.",
    )
    parser.add_argument(
        "-a", "--alpha", type=float, default=1.0,
        help=(
            "Mixing weight for S. cerevisiae vs. input background frequencies "
            "(0 = input only, 1 = S.cer only). Default: 1.0."
        ),
    )
    parser.add_argument(
        "-w", "--window-fi", type=int, default=41,
        help="Window size for FoldIndex disorder predictions. Default: 41.",
    )
    parser.add_argument(
        "-W", "--window-papa", type=int, default=41,
        help="Window size for PAPA algorithm. Default: 41.",
    )
    parser.add_argument(
        "-p", "--plot-list", dest="plot_list", default="",
        help=(
            "File listing sequence names to plot (one per line), "
            "or 'all' to plot everything."
        ),
    )
    parser.add_argument(
        "-H", "--hmm-dot", dest="hmm_dot_file", default="",
        help="Write HMM structure in Graphviz DOT format to this file.",
    )
    parser.add_argument(
        "-d", "--print-headers", action="store_true", default=False,
        help="Print column documentation at the top of the output.",
    )
    parser.add_argument(
        "-s", "--skip-parameters", action="store_true", default=False,
        help="Skip printing run-time parameters at the top of the output.",
    )

    return parser


def parse_args(argv: Optional[List[str]] = None) -> PLAACConfig:
    """
    Parse command-line arguments and return a :class:`PLAACConfig`.

    Parameters
    ----------
    argv : list of str, optional
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    PLAACConfig
        Populated configuration dataclass.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    return PLAACConfig(
        input_file=args.input_file,
        bg_fasta=args.bg_fasta,
        bg_freq_file=args.bg_freq_file,
        fg_freq_file=args.fg_freq_file,
        plot_list=args.plot_list,
        hmm_dot_file=args.hmm_dot_file,
        core_length=args.core_length,
        window_fi=args.window_fi,
        window_papa=args.window_papa,
        alpha=args.alpha,
        print_headers=args.print_headers,
        skip_parameters=args.skip_parameters,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run(cfg: PLAACConfig) -> None:
    """
    Execute a full PLAAC analysis with the given configuration.

    This is the main orchestration function -- equivalent to the Java
    ``main()`` method.

    Parameters
    ----------
    cfg : PLAACConfig
        All run-time parameters.
    """
    props = AA
    fg_freq: List[float] = list(props.prd_freq_scer_28)
    bg_scer: List[float] = _normalize(list(props.bg_freq_scer))

    # -- Resolve background frequencies --
    bg_raw = [0.0] * 22
    if cfg.bg_freq_file:
        bg_raw = read_aa_params(cfg.bg_freq_file)
    elif cfg.bg_fasta:
        bg_raw = [float(x) for x in compute_aa_freq(cfg.bg_fasta)]
    elif cfg.input_file:
        bg_raw = [float(x) for x in compute_aa_freq(cfg.input_file)]

    # -- Resolve foreground frequencies --
    if cfg.fg_freq_file:
        fg_freq = read_aa_params(cfg.fg_freq_file)

    # -- Early-exit modes --
    if cfg.bg_fasta and not cfg.input_file:
        write_aa_params(bg_raw)
        return

    if cfg.bg_freq_file and not cfg.input_file:
        write_aa_params(bg_raw)
        return

    if not cfg.input_file and not cfg.bg_fasta:
        build_parser().print_help()
        return

    # -- Validate alpha --
    alpha = cfg.alpha
    if not 0 <= alpha <= 1:
        print("# warning: invalid alpha; using alpha = 1.0", file=sys.stderr)
        alpha = 1.0

    # -- Normalise frequencies --
    fg_freq[0] = fg_freq[21] = 0.0
    fg_freq = _normalize(fg_freq)

    bg_raw[0] = bg_raw[21] = 0.0
    bg_this = _normalize(bg_raw)

    # Weighted mixture: alpha * S.cer + (1-alpha) * input
    bg_combo = _normalize(
        [alpha * bg_scer[i] + (1 - alpha) * bg_this[i] for i in range(22)]
    )

    # Small epsilon for X and * slots so the HMM never sees -inf
    eps = 1e-5
    fg_freq[0] = fg_freq[21] = eps
    bg_combo[0] = bg_combo[21] = eps
    fg = _normalize(fg_freq)
    bg = _normalize(bg_combo)

    # Per-AA log-likelihood ratios
    llr = [0.0] * 22
    for j in range(1, 21):
        llr[j] = math.log(fg[j] / bg[j])

    # -- Print run-time parameters --
    if not cfg.skip_parameters:
        print(
            "############################ parameters at run-time "
            "####################################"
        )
        print(
            f"## alpha={alpha}; core_length={cfg.core_length}; "
            f"window_fi={cfg.window_fi}; window_papa={cfg.window_papa}; "
            f"adjust_prolines={cfg.adjust_prolines};"
        )
        print(f"## fg_used: {{{format_aa_params(fg)}}}")
        print(f"## bg_scer: {{{format_aa_params(bg_scer)}}}")
        print(f"## bg_input: {{{format_aa_params(bg_this)}}}")
        print(f"## bg_used: {{{format_aa_params(bg)}}}")
        print(f"## plaac_llr: {{{format_aa_params(llr)}}}")
        print(f"## papa_lods: {{{format_aa_params(AA.lod_papa1)}}}")
        print(
            "###############################################################"
            "########################"
        )

    # -- Build HMMs --
    hmm1 = build_prion_hmm(fg, bg)
    hmm0 = build_background_hmm(bg)

    if cfg.hmm_dot_file:
        hmm1.write_dot(cfg.hmm_dot_file, show_emissions=True)

    # -- Score or plot --
    if cfg.input_file and not cfg.plot_list:
        score_all_fastas(cfg, fg, bg, llr, hmm1, hmm0)
    elif cfg.input_file and cfg.plot_list:
        plot_some_fastas(cfg, fg, bg, llr, hmm1, hmm0)


def main(argv: Optional[List[str]] = None) -> None:
    """
    CLI entry point for PLAAC.

    Parameters
    ----------
    argv : list of str, optional
        Argument list (defaults to ``sys.argv[1:]``).
    """
    cfg = parse_args(argv)
    run(cfg)


if __name__ == "__main__":
    main()
