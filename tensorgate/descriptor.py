"""TensorDescriptor — full numerical and structural description of a tensor."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .layout import Layout
from .tensor import as_numpy


def _dtype_str(dtype) -> str:
    return str(np.dtype(dtype))


def _content_hash(arr: np.ndarray, meta: dict) -> str:
    """Deterministic content identity.

    Hashes canonical little-endian bytes of the array together with
    dtype, shape, and relevant representation metadata. Two materially
    different tensors cannot collide merely because JSON metadata matches.
    """
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(_dtype_str(arr.dtype).encode("utf-8"))
    view = np.ascontiguousarray(arr)
    if view.dtype.byteorder == ">":
        view = view.byteswap().view(view.dtype.newbyteorder("<"))
    h.update(view.tobytes())
    for key in ("scale", "zero_point", "quantization", "layout"):
        if key in meta and meta[key] is not None:
            h.update(f"{key}={meta[key]}".encode("utf-8"))
    return h.hexdigest()


@dataclass
class TensorDescriptor:
    """Complete description of a tensor's numerical and structural state."""

    dtype: str
    shape: Tuple[int, ...]
    rank: int
    element_count: int
    byte_size: int
    min: Optional[float]
    max: Optional[float]
    mean: Optional[float]
    std: Optional[float]
    rms: Optional[float]
    abs_max: Optional[float]
    finite: bool
    nan_count: int
    pos_inf_count: int
    neg_inf_count: int
    zero_count: int
    sparsity: float
    layout: str
    scale: Optional[float] = None
    zero_point: Optional[int] = None
    quantization: Optional[str] = None
    provenance: List[Dict[str, Any]] = field(default_factory=list)
    content_hash: str = ""
    software_id: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shape"] = list(self.shape)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TensorDescriptor":
        d = dict(d)
        d["shape"] = tuple(d["shape"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def summary(self) -> str:
        lines = [
            f"dtype={self.dtype}  shape={self.shape}  rank={self.rank}",
            f"elements={self.element_count}  bytes={self.byte_size}  layout={self.layout}",
            f"min={self.min}  max={self.max}  mean={self.mean}  std={self.std}",
            f"rms={self.rms}  abs_max={self.abs_max}",
            f"finite={self.finite}  nan={self.nan_count}  +inf={self.pos_inf_count}  -inf={self.neg_inf_count}",
            f"zeros={self.zero_count}  sparsity={self.sparsity:.6f}",
        ]
        if self.scale is not None or self.zero_point is not None:
            lines.append(f"scale={self.scale}  zero_point={self.zero_point}  quant={self.quantization}")
        lines.append(f"hash={self.content_hash[:16]}…")
        return "\n".join(lines)


def build_descriptor(
    arr: np.ndarray,
    *,
    scale: Optional[float] = None,
    zero_point: Optional[int] = None,
    quantization: Optional[str] = None,
    provenance: Optional[List[Dict[str, Any]]] = None,
    software_id: str = "",
) -> TensorDescriptor:
    """Construct a TensorDescriptor from a NumPy array. Pure inspection."""
    arr = np.asanyarray(arr)
    flat = arr.ravel()
    n = flat.size
    dtype = _dtype_str(arr.dtype)
    layout = Layout.from_numpy(arr).value

    if np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.complexfloating):
        nan_mask = np.isnan(flat)
        pos_inf = np.isposinf(flat)
        neg_inf = np.isneginf(flat)
        nan_count = int(nan_mask.sum())
        pos_inf_count = int(pos_inf.sum())
        neg_inf_count = int(neg_inf.sum())
        finite_mask = np.isfinite(flat)
        finite_vals = flat[finite_mask]
        finite = bool(finite_mask.all())
    else:
        nan_count = pos_inf_count = neg_inf_count = 0
        finite_vals = flat
        finite = True

    zero_count = int((flat == 0).sum()) if n else 0
    sparsity = float(zero_count / n) if n else 0.0

    if finite_vals.size:
        f = finite_vals.astype(np.float64, copy=False)
        mn = float(f.min())
        mx = float(f.max())
        mean = float(f.mean())
        std = float(f.std())
        rms = float(np.sqrt(np.mean(f * f)))
        abs_max = float(np.max(np.abs(f)))
    else:
        mn = mx = mean = std = rms = abs_max = None

    meta = {
        "scale": scale,
        "zero_point": zero_point,
        "quantization": quantization,
        "layout": layout,
    }
    ch = _content_hash(arr, meta)

    return TensorDescriptor(
        dtype=dtype,
        shape=tuple(int(s) for s in arr.shape),
        rank=arr.ndim,
        element_count=int(n),
        byte_size=int(arr.nbytes),
        min=mn,
        max=mx,
        mean=mean,
        std=std,
        rms=rms,
        abs_max=abs_max,
        finite=finite,
        nan_count=nan_count,
        pos_inf_count=pos_inf_count,
        neg_inf_count=neg_inf_count,
        zero_count=zero_count,
        sparsity=sparsity,
        layout=layout,
        scale=scale,
        zero_point=zero_point,
        quantization=quantization,
        provenance=list(provenance or []),
        content_hash=ch,
        software_id=software_id,
    )
