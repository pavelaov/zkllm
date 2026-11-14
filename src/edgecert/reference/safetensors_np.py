"""Minimal numpy reader for the safetensors format (no torch, no safetensors package).

File layout: 8-byte little-endian u64 header length N, then N bytes of UTF-8 JSON mapping tensor name -> {"dtype", "shape", "data_offsets": [begin, end]} (offsets relative to the end of the header) plus an optional "__metadata__" entry, then the raw little-endian tensor bytes.

Supported dtypes: F64, F32, F16, BF16, I64, I32, I16, I8, U8, BOOL.  BF16 is widened exactly to float32 by placing the 16 stored bits in the high half of a float32 (``uint16 << 16``), which is the definition of bfloat16.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

_DT = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32,
       "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}
_ITEMSIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
             "BOOL": 1}

def bf16_to_f32(raw_u16: np.ndarray) -> np.ndarray:
    """Exact bfloat16 -> float32 widening: the bf16 bits are the top 16 bits of the f32."""
    u = np.asarray(raw_u16, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return u.view(np.float32)

class SafeTensors:
    """Lazy reader: ``st[name]`` returns a numpy array (BF16 widened to float32)."""

    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            if n <= 0 or n > 100_000_000:
                raise ValueError(f"{self.path}: implausible safetensors header length {n}")
            hdr = json.loads(f.read(n).decode("utf-8"))
        self.metadata = hdr.pop("__metadata__", {}) or {}
        self.header: Dict[str, dict] = hdr
        self.data_start = 8 + n
        self.file_size = self.path.stat().st_size
        for name, e in hdr.items():
            if e["dtype"] not in _ITEMSIZE:
                raise ValueError(f"{name}: unsupported dtype {e['dtype']}")
            b, t = e["data_offsets"]
            nbytes = int(np.prod(e["shape"], dtype=np.int64)) * _ITEMSIZE[e["dtype"]] if e["shape"] else \
                _ITEMSIZE[e["dtype"]]
            if t - b != nbytes or self.data_start + t > self.file_size:
                raise ValueError(f"{name}: data_offsets inconsistent with dtype/shape or file size")
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")

    def keys(self) -> List[str]:
        return list(self.header.keys())

    def __contains__(self, name: str) -> bool:
        return name in self.header

    def info(self, name: str) -> Tuple[str, List[int]]:
        e = self.header[name]
        return e["dtype"], list(e["shape"])

    def raw(self, name: str) -> np.ndarray:
        """The stored bytes reinterpreted with the stored dtype (BF16 -> uint16 bit patterns)."""
        e = self.header[name]
        b, t = e["data_offsets"]
        buf = self._mm[self.data_start + b: self.data_start + t]
        dt = np.uint16 if e["dtype"] == "BF16" else _DT[e["dtype"]]
        return np.frombuffer(buf, dtype=np.dtype(dt).newbyteorder("<")).reshape(e["shape"])

    def __getitem__(self, name: str) -> np.ndarray:
        e = self.header[name]
        r = self.raw(name)
        if e["dtype"] == "BF16":
            return bf16_to_f32(r)
        return np.array(r, dtype=r.dtype.newbyteorder("="))

    def items(self) -> Iterator[Tuple[str, np.ndarray]]:
        for k in self.header:
            yield k, self[k]

def load(path) -> Dict[str, np.ndarray]:
    st = SafeTensors(path)
    return {k: st[k] for k in st.keys()}
