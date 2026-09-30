"""Explicit, recorded normalization transforms."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .descriptor import build_descriptor
from .inspect import analyze
from .provenance import ProvenanceRecord, make_record
from .tensor import as_numpy, ArrayLike


def normalize(
    tensor: ArrayLike,
    method: str = "symmetric",
    *,
    target_min: float = -1.0,
    target_max: float = 1.0,
    eps: float = 1e-12,
    clip_outliers: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply an explicit normalization. Returns (result, metadata).

    Methods:
      - range        : map [min, max] → [target_min, target_max]
      - maxabs       : divide by abs_max (→ [-1, 1] if signed)
      - standard     : (x - mean) / std
      - symmetric    : map [-abs_max, abs_max] → [target_min, target_max]
      - affine       : same as range (alias)
      - none         : identity (still records a no-op)

    Never silent. Metadata always describes source range, scale, offset, clipping.
    Degenerate cases (all-zero, constant, empty) are handled without producing NaNs.
    """
    arr = as_numpy(tensor, copy=True)
    desc = analyze(arr)
    method = method.lower().strip()

    if method in ("none", "identity"):
        meta = {
            "method": "none",
            "source_min": desc.min,
            "source_max": desc.max,
            "scale": 1.0,
            "offset": 0.0,
            "clipped": 0,
            "transformation": "identity",
        }
        out = arr
    elif method in ("maxabs", "max-absolute", "max_absolute"):
        abs_max = desc.abs_max if desc.abs_max is not None else 0.0
        if abs_max < eps:
            scale = 1.0
            out = arr.astype(np.float64)
            degenerate = True
        else:
            scale = 1.0 / abs_max
            out = arr.astype(np.float64) * scale
            degenerate = False
        meta = {
            "method": "maxabs",
            "source_abs_max": abs_max,
            "scale": scale,
            "offset": 0.0,
            "clipped": 0,
            "target_range": [-1.0, 1.0],
            "degenerate": degenerate,
        }
    elif method in ("standard", "zscore", "standard-score"):
        mean = desc.mean if desc.mean is not None else 0.0
        std = desc.std if desc.std is not None else 0.0
        if std < eps:
            out = np.zeros_like(arr, dtype=np.float64)
            scale = 0.0
            offset = 0.0
            degenerate = True
        else:
            scale = 1.0 / std
            out = (arr.astype(np.float64) - mean) * scale
            offset = -mean * scale
            degenerate = False
        meta = {
            "method": "standard",
            "source_mean": mean,
            "source_std": std,
            "scale": scale,
            "offset": offset,
            "clipped": 0,
            "degenerate": degenerate,
        }
    elif method in ("range", "affine", "minmax"):
        src_min = desc.min if desc.min is not None else 0.0
        src_max = desc.max if desc.max is not None else 1.0
        span = src_max - src_min
        if span < eps:
            mid = 0.5 * (target_min + target_max)
            out = np.full_like(arr, mid, dtype=np.float64)
            scale = 0.0
            offset = mid
            clipped = 0
            degenerate = True
        else:
            scale = (target_max - target_min) / span
            offset = target_min - src_min * scale
            out = arr.astype(np.float64) * scale + offset
            clipped = 0
            degenerate = False
            if clip_outliers:
                before = out.copy()
                out = np.clip(out, target_min, target_max)
                clipped = int(np.sum(before != out))
        meta = {
            "method": method,
            "source_min": src_min,
            "source_max": src_max,
            "target_min": target_min,
            "target_max": target_max,
            "scale": scale,
            "offset": offset,
            "clipped": clipped,
            "degenerate": degenerate,
        }
    elif method == "symmetric":
        abs_max = desc.abs_max if desc.abs_max is not None else 0.0
        mid = 0.5 * (target_min + target_max)
        half = 0.5 * (target_max - target_min)
        if abs_max < eps:
            out = np.full_like(arr, mid, dtype=np.float64) if arr.size else arr.astype(np.float64)
            scale = 0.0
            degenerate = True
            clipped = 0
        else:
            scale = half / abs_max
            out = arr.astype(np.float64) * scale
            if mid != 0.0:
                out = out + mid
            clipped = 0
            degenerate = False
            if clip_outliers:
                before = out.copy()
                out = np.clip(out, target_min, target_max)
                clipped = int(np.sum(before != out))
        meta = {
            "method": "symmetric",
            "source_abs_max": abs_max,
            "target_min": target_min,
            "target_max": target_max,
            "scale": scale,
            "offset": mid if mid != 0 else 0.0,
            "clipped": clipped,
            "degenerate": degenerate,
        }
    else:
        raise ValueError(
            f"Unknown normalization method '{method}'. "
            "Supported: range, maxabs, standard, symmetric, affine, none"
        )

    in_hash = desc.content_hash
    out_desc = build_descriptor(out)
    record = make_record(
        operation=f"normalize/{method}",
        parameters={k: v for k, v in meta.items() if k != "transformation"},
        input_hash=in_hash,
        output_hash=out_desc.content_hash,
    )
    meta["provenance"] = [record.to_dict()]
    meta["input_hash"] = in_hash
    meta["output_hash"] = out_desc.content_hash
    meta["output_descriptor"] = out_desc.to_dict()
    return out, meta
