"""Quantization and dequantization with full metadata."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .descriptor import build_descriptor
from .inspect import analyze
from .provenance import make_record
from .tensor import as_numpy, ArrayLike


def quantize(
    tensor: ArrayLike,
    *,
    bits: int = 8,
    scheme: str = "symmetric",
    signed: bool = True,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Quantize to integer representation.

    scheme:
      - symmetric  : scale = abs_max / qmax, zero_point = 0
      - asymmetric : scale = (max - min) / (qmax - qmin), zero_point computed
      - affine     : alias for asymmetric

    Returns integer array + metadata (scale, zero_point, scheme, bits, …).
    """
    arr = as_numpy(tensor, copy=True).astype(np.float64)
    desc = analyze(arr)
    scheme = scheme.lower().strip()
    if scheme == "affine":
        scheme = "asymmetric"

    if bits < 2 or bits > 16:
        raise ValueError("bits must be in [2, 16]")

    if signed:
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** bits - 1

    if scheme == "symmetric":
        abs_max = desc.abs_max if desc.abs_max is not None else 0.0
        scale = max(abs_max, eps) / qmax
        zero_point = 0
        q = np.round(arr / scale).astype(np.int32)
        q = np.clip(q, qmin, qmax)
    elif scheme == "asymmetric":
        src_min = desc.min if desc.min is not None else 0.0
        src_max = desc.max if desc.max is not None else 1.0
        scale = max(src_max - src_min, eps) / (qmax - qmin)
        zero_point = int(np.round(qmin - src_min / scale))
        zero_point = int(np.clip(zero_point, qmin, qmax))
        q = np.round(arr / scale + zero_point).astype(np.int32)
        q = np.clip(q, qmin, qmax)
    else:
        raise ValueError(f"Unknown quantization scheme '{scheme}'")

    if bits <= 8:
        storage = np.int8 if signed else np.uint8
    else:
        storage = np.int16 if signed else np.uint16
    out = q.astype(storage)

    out_desc = build_descriptor(out, scale=scale, zero_point=zero_point, quantization=scheme)
    record = make_record(
        operation="quantize",
        parameters={
            "bits": bits,
            "scheme": scheme,
            "signed": signed,
            "scale": scale,
            "zero_point": zero_point,
            "qmin": qmin,
            "qmax": qmax,
        },
        input_hash=desc.content_hash,
        output_hash=out_desc.content_hash,
    )
    meta = {
        "bits": bits,
        "scheme": scheme,
        "signed": signed,
        "scale": float(scale),
        "zero_point": int(zero_point),
        "qmin": qmin,
        "qmax": qmax,
        "dtype": str(out.dtype),
        "provenance": [record.to_dict()],
        "input_hash": desc.content_hash,
        "output_hash": out_desc.content_hash,
        "output_descriptor": out_desc.to_dict(),
    }
    return out, meta


def dequantize(
    tensor: ArrayLike,
    *,
    scale: float,
    zero_point: int = 0,
    dtype: str = "float32",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Dequantize integer tensor back to floating point."""
    arr = as_numpy(tensor, copy=True)
    desc = analyze(arr, scale=scale, zero_point=zero_point, quantization="stored")
    out = (arr.astype(np.float64) - zero_point) * scale
    out = out.astype(np.dtype(dtype))
    out_desc = build_descriptor(out)
    record = make_record(
        operation="dequantize",
        parameters={"scale": scale, "zero_point": zero_point, "dtype": dtype},
        input_hash=desc.content_hash,
        output_hash=out_desc.content_hash,
    )
    meta = {
        "scale": scale,
        "zero_point": zero_point,
        "dtype": dtype,
        "provenance": [record.to_dict()],
        "input_hash": desc.content_hash,
        "output_hash": out_desc.content_hash,
    }
    return out, meta
