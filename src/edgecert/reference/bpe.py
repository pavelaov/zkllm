"""Byte-level BPE tokenizers for GPT-2 and Qwen3 in pure Python (no `tokenizers` package).

Both models use byte-level BPE: text -> (normalizer) -> regex pre-tokenization -> UTF-8 bytes of each piece mapped to printable unicode characters (GPT-2 ``bytes_to_unicode``) -> BPE merges by rank -> vocabulary ids.  Decoding is the exact inverse (ids -> strings -> bytes -> UTF-8).

Pre-tokenizer regexes (copied from the model files):

* GPT-2 (``ByteLevel`` pre-tokenizer, ``use_regex=True``)::

      's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+

* Qwen3 (``Split`` with behaviour ``Isolated`` then ``ByteLevel`` without regex, NFC normalizer)::

      (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|
      \\s*[\\r\\n]+|\\s+(?!\\S)|\\s+

Python's ``re`` has no ``\\p{..}`` classes.  We emulate them exactly *with respect to Python's unicodedata tables*: ``\\p{L}`` is the set of code points whose general category starts with ``L`` (Lu, Ll, Lt, Lm, Lo), ``\\p{N}`` the set with ``N`` (Nd, Nl, No); both are expanded into explicit character-class ranges.  ``\\s`` is replaced by the Unicode ``White_Space`` property list (what Oniguruma/Rust use), not by Python's ``str.isspace`` set.  Documented approximations:

1. Unicode version skew: Python 3.14 ships Unicode 16 tables, the Hugging Face ``tokenizers`` build may use an older version; only code points assigned between the two versions can differ. 2. ``(?i:...)`` case folding follows Python ``re`` (it matches e.g. ``'S``; exotic folds such as U+017F LATIN SMALL LETTER LONG S may differ). 3. Added/special tokens (``<|endoftext|>``, ``<|im_start|>`` ...) are split out before normalization when ``allow_special=True`` (the HF default for special tokens). 4. The Qwen3 NFC normalizer is applied, so ``decode(encode(t)) == t`` holds for NFC-normalized text (all prompts in data/prompts are ASCII).
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

WHITE_SPACE = [(0x09, 0x0D), (0x20, 0x20), (0x85, 0x85), (0xA0, 0xA0), (0x1680, 0x1680), (0x2000, 0x200A),
               (0x2028, 0x2029), (0x202F, 0x202F), (0x205F, 0x205F), (0x3000, 0x3000)]

@lru_cache(maxsize=None)
def bytes_to_unicode() -> Dict[int, str]:
    """GPT-2's reversible byte -> printable unicode character map."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

def _ranges_to_class(ranges: Iterable[Tuple[int, int]]) -> str:
    def esc(cp: int) -> str:
        return f"\\U{cp:08x}"
    return "".join(esc(a) if a == b else f"{esc(a)}-{esc(b)}" for a, b in ranges)

@lru_cache(maxsize=None)
def _category_ranges(prefix: str) -> Tuple[Tuple[int, int], ...]:
    out: List[Tuple[int, int]] = []
    start = None
    for cp in range(0x110000):
        hit = unicodedata.category(chr(cp)).startswith(prefix)
        if hit and start is None:
            start = cp
        elif not hit and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return tuple(out)

@lru_cache(maxsize=None)
def unicode_classes() -> Dict[str, str]:
    """Character-class bodies (no brackets) for \\p{L}, \\p{N} and White_Space."""
    return {"L": _ranges_to_class(_category_ranges("L")), "N": _ranges_to_class(_category_ranges("N")),
            "S": _ranges_to_class(WHITE_SPACE)}

@lru_cache(maxsize=None)
def gpt2_pattern() -> "re.Pattern[str]":
    c = unicode_classes()
    L, N, S = c["L"], c["N"], c["S"]
    pat =(rf"'s|'t|'re|'ve|'m|'ll|'d| ?[{L}]+| ?[{N}]+| ?[^{S}{L}{N}]+|[{S}]+(?![^{S}])|[{S}]+")
    return re.compile(pat)

@lru_cache(maxsize=None)
def qwen2_pattern() -> "re.Pattern[str]":
    c = unicode_classes()
    L, N, S = c["L"], c["N"], c["S"]
    pat = (rf"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n{L}{N}]?[{L}]+|[{N}]| ?[^{S}{L}{N}]+[\r\n]*|"
           rf"[{S}]*[\r\n]+|[{S}]+(?![^{S}])|[{S}]+")
    return re.compile(pat)

def split_isolated(pattern: "re.Pattern[str]", text: str) -> List[str]:
    """HF ``Split(behavior='Isolated')``: every match is a piece and so is every gap."""
    out: List[str] = []
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            out.append(text[pos:m.start()])
        if m.end() > m.start():
            out.append(m.group(0))
        pos = m.end()
    if pos < len(text):
        out.append(text[pos:])
    return out

class ByteBPE:
    """Byte-level BPE with merge ranks (lowest rank merged first, as in GPT-2 / HF BPE)."""

    def __init__(self, name: str, vocab: Dict[str, int], merges: Sequence[Tuple[str, str]],
                 pattern: "re.Pattern[str]", normalizer: Optional[str] = None,
                 added: Optional[Dict[str, int]] = None, pattern_src: str = ""):
        self.name = name
        self.vocab = dict(vocab)
        self.added = dict(added or {})
        self.id_to_tok: Dict[int, str] = {i: t for t, i in self.vocab.items()}
        for t, i in self.added.items():
            self.id_to_tok[i] = t
        self.ranks = {tuple(m): i for i, m in enumerate(merges)}
        self.pattern = pattern
        self.pattern_src = pattern_src
        self.normalizer = normalizer
        self.b2u = bytes_to_unicode()
        self.u2b = {u: b for b, u in self.b2u.items()}
        self._cache: Dict[str, Tuple[int, ...]] = {}
        self._special_re = re.compile("(" + "|".join(re.escape(t) for t in sorted(self.added, key=len,
                                                                                   reverse=True)) + ")") \
            if self.added else None

    @property
    def vocab_size(self) -> int:
        return max(self.id_to_tok) + 1

    # ------------------------------------------------------------- BPE core
    def _bpe(self, piece: str) -> Tuple[int, ...]:
        hit = self._cache.get(piece)
        if hit is not None:
            return hit
        word = [self.b2u[b] for b in piece.encode("utf-8")]
        ranks = self.ranks
        while len(word) > 1:
            best, best_i = None, -1
            for i in range(len(word) - 1):
                r = ranks.get((word[i], word[i + 1]))
                if r is not None and (best is None or r < best):
                    best, best_i = r, i
            if best is None:
                break
            a, b = word[best_i], word[best_i + 1]
            new: List[str] = []
            i = 0
            while i < len(word):          # merge every occurrence of the best pair, left to right
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new.append(a + b)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = new
        try:
            ids = tuple(self.vocab[t] for t in word)
        except KeyError as e:   # cannot happen for a complete byte-level vocabulary
            raise ValueError(f"{self.name}: token {e} missing from vocabulary") from None
        if len(self._cache) < 200_000:
            self._cache[piece] = ids
        return ids

    def pretokenize(self, text: str) -> List[str]:
        if self.normalizer == "NFC":
            text = unicodedata.normalize("NFC", text)
        return split_isolated(self.pattern, text)

    def encode(self, text: str, allow_special: bool = True) -> List[int]:
        parts = self._special_re.split(text) if (allow_special and self._special_re) else [text]
        ids: List[int] = []
        for part in parts:
            if not part:
                continue
            if allow_special and part in self.added:
                ids.append(self.added[part])
                continue
            for piece in self.pretokenize(part):
                ids.extend(self._bpe(piece))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        out = bytearray()
        for i in ids:
            t = self.id_to_tok[int(i)]
            if int(i) in self.added.values() and t in self.added:
                out += t.encode("utf-8")
            else:
                out += bytes(self.u2b[ch] for ch in t)
        return out.decode("utf-8", errors="replace")

    def tokens(self, ids: Iterable[int]) -> List[str]:
        return [self.id_to_tok[int(i)] for i in ids]

    # ---------------------------------------------------------- loaders
    @classmethod
    def gpt2(cls, model_dir) -> "ByteBPE":
        d = Path(model_dir)
        vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        merges = []
        for line in (d / "merges.txt").read_text(encoding="utf-8").split("\n"):
            if not line or line.startswith("#version"):
                continue
            a, b = line.split(" ")
            merges.append((a, b))
        added = {"<|endoftext|>": 50256}
        return cls("gpt2", vocab, merges, gpt2_pattern(), None, added,
                   pattern_src=r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

    @classmethod
    def qwen3(cls, model_dir) -> "ByteBPE":
        d = Path(model_dir)
        tj = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
        m = tj["model"]
        assert m["type"] == "BPE" and not m.get("byte_fallback")
        merges = [tuple(x) if isinstance(x, list) else tuple(x.split(" ")) for x in m["merges"]]
        added = {a["content"]: a["id"] for a in tj["added_tokens"]}
        norm = (tj.get("normalizer") or {}).get("type")
        pre = tj["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
        return cls("qwen3", m["vocab"], merges, qwen2_pattern(), norm, added, pattern_src=pre)

def load_tokenizer(family: str, model_dir) -> ByteBPE:
    return ByteBPE.gpt2(model_dir) if family == "gpt2" else ByteBPE.qwen3(model_dir)
