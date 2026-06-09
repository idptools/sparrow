from __future__ import annotations
from math import ceil, log2
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Dict

AMINO_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWY")  # 20 canonical amino acids
SYMBOL_BITS = ceil(log2(len(AMINO_ALPHABET)))     # = 5 bits
ALPHABET_SET = set(AMINO_ALPHABET)

@dataclass(frozen=True)
class Token:
    """One compressed token in LZ78-style (parent_index, next_symbol).

    parent_index: int  -> index into the dictionary (1-based); 0 = empty prefix
    next_symbol:  str  -> a single amino-acid letter
    """
    parent_index: int
    next_symbol: str


class KMerCompressor:
    """K-mer (phrase) dictionary compressor tailored for 20-letter amino strings.

    This is a clean, simple LZ78-style algorithm (variable-length fragments):
      • Parses the input greedily into the longest phrase already in the dictionary,
        then extends it by one symbol to create a new dictionary entry.
      • Emits a Token(parent_index, next_symbol) for each phrase.

    Rationale for "k-mer-based": phrases are variable-length k-mers discovered
    on-the-fly. The resulting compressed representation is the token stream plus
    the implicit dictionary reconstruction order.
    """

    def __init__(self, alphabet: Iterable[str] = AMINO_ALPHABET):
        self.alphabet = tuple(alphabet)
        self._alpha_set = set(self.alphabet)
        if len(self._alpha_set) != len(self.alphabet):
            raise ValueError("Alphabet has duplicates.")
        if len(self.alphabet) != 20:
            # Not strictly required, but helpful to catch mistakes
            pass

    def _validate(self, s: str) -> None:
        bad = set(s) - self._alpha_set
        if bad:
            raise ValueError(f"Input contains non-alphabet symbols: {sorted(bad)}")

    def compress(self, s: str) -> List[Token]:
        """Compress string `s` into a list of Tokens.

        Time: O(n) amortized with hash map lookups.
        Space: O(n) dictionary size in the worst case.
        """
        self._validate(s)

        # Dictionary maps phrase -> index (1-based). Start empty; LZ78 indexes
        # stored phrases, where each phrase = previous_phrase + next_symbol.
        dictionary: Dict[str, int] = {}
        tokens: List[Token] = []

        i, n = 0, len(s)
        while i < n:
            # Find the longest phrase starting at i that is in the dictionary
            j = i
            phrase_index = 0
            while j < n:
                candidate = s[i:j+1]
                if candidate in dictionary:
                    phrase_index = dictionary[candidate]
                    j += 1
                else:
                    break

            # If we exited because candidate not in dictionary (or j==i):
            if j < n:
                # next symbol extends the longest known phrase
                next_symbol = s[j]
                tokens.append(Token(phrase_index, next_symbol))
                # Add new phrase = (known phrase) + next_symbol
                new_phrase = s[i:j+1]
                dictionary[new_phrase] = len(dictionary) + 1
                # Advance input pointer past the consumed (known phrase + 1 symbol)
                i = j + 1
            else:
                # Ran out of input exactly on a known phrase; emit a token that
                # extends it with a sentinel-like symbol by splitting last char.
                # To keep tokens canonical, emit (parent_of_last_char, last_char).
                # That guarantees invertibility without adding sentinels.
                known = s[i:j]
                if not known:
                    break
                # Split known = prefix + last_char so that we can emit a token
                # that reconstructs the last step. Find prefix index.
                prefix, last_char = known[:-1], known[-1]
                prefix_index = dictionary[prefix] if prefix else 0
                tokens.append(Token(prefix_index, last_char))
                # No need to add new dictionary entry; we're done.
                break

        return tokens

    def decompress(self, tokens: Iterable[Token]) -> str:
        """Invert the token stream back to the original string.

        The dictionary is rebuilt in the same order it was constructed during
        compression. Each token defines a new phrase: dict[parent] + next_symbol.
        """
        # dictionary index -> phrase; index 0 reserved for empty string
        dict_idx_to_phrase: List[str] = [""]  # slot 0 = empty
        out_chars: List[str] = []

        for t in tokens:
            if not isinstance(t.next_symbol, str) or len(t.next_symbol) != 1:
                raise ValueError("Token next_symbol must be a single character.")
            if t.parent_index < 0 or t.parent_index >= len(dict_idx_to_phrase):
                raise ValueError("Invalid parent_index in token stream.")
            phrase = dict_idx_to_phrase[t.parent_index] + t.next_symbol
            out_chars.append(phrase)
            dict_idx_to_phrase.append(phrase)

        return "".join(out_chars)

    # -------------------------- Metrics & Utilities --------------------------

    @staticmethod
    def bit_cost(tokens: Iterable[Token], symbol_bits: int = SYMBOL_BITS) -> int:
        """Approximate bit cost of the compressed representation.

        Each token requires log2(D) bits for the parent index where D is the
        current dictionary size (including the empty entry) and `symbol_bits`
        for the next_symbol. We sum the per-token costs as the dictionary grows.
        """
        total = 0
        dict_size = 1  # includes the empty entry at index 0
        for t in tokens:
            # Bits to store parent index in [0, dict_size-1]
            parent_bits = 1 if dict_size <= 2 else ceil(log2(dict_size))
            total += parent_bits + symbol_bits
            dict_size += 1
        return total

    @staticmethod
    def theoretical_raw_bits(n_chars: int, symbol_bits: int = SYMBOL_BITS) -> int:
        return n_chars * symbol_bits

    @classmethod
    def compression_ratio(cls, s: str, tokens: Iterable[Token]) -> float:
        """Return compressed_bits / raw_bits (smaller is better)."""
        tokens = list(tokens)
        raw = cls.theoretical_raw_bits(len(s))
        comp = cls.bit_cost(tokens)
        return comp / raw if raw else 1.0


