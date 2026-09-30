"""General transforms: dtype conversion, clipping, and source→target adaptation helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .descriptor import build_descriptor
from .inspect import analyze
from .provenance import make_record
from .tensor import as_numpy, ArrayLike, TensorSpec


def convert_dtype(
    tensor: ArrayLike,
    dtype: str,
    *,
    casting: str = "safe",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Convert dtype explicitly. Returns (result, metadata).

    casting follows numpy: 'safe' raises on lossy conversions unless overridden.
    """
    arr = as_numpy(tensor, copy=True)
    desc = analyze(arr)
    target = np.dtype(dtype)
    try:
        out = arr.astype(target, casting=casting, copy=False)
    except TypeError:
        out = arr.astype(target, casting="unsafe", copy=False)

    out_desc = build_descriptor(out)
    record = make_record(
        operation="convert_dtype",
        parameters={"source_dtype": desc.dtype, "target_dtype": str(target), "casting": casting},
        input_hash=desc.content_hash,
        output_hash=out_desc.content_hash,
    )
    meta = {
        "source_dtype": desc.dtype,
        "target_dtype": str(target),
        "casting": casting,
        "provenance": [record.to_dict()],
        "input_hash": desc.content_hash,
        "output_hash": out_desc.content_hash,
    }
    return out, meta


def clip(
    tensor: ArrayLike,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Clip values to [min_value, max_value]. Records clipped count."""
    arr = as_numpy(tensor, copy=True)
    desc = analyze(arr)
    before = arr.copy()
    out = np.clip(arr, min_value if min_value is not None else -np.inf,
                  max_value if max_value is not None else np.inf)
    changed = int(np.sum(before != out))
    out_desc = build_descriptor(out)
    record = make_record(
        operation="clip",
        parameters={"min_value": min_value, "max_value": max_value, "clipped_count": changed},
        input_hash=desc.content_hash,
        output_hash=out_desc.content_hash,
    )
    meta = {
        "min_value": min_value,
        "max_value": max_value,
        "clipped_count": changed,
        "provenance": [record.to_dict()],
        "input_hash": desc.content_hash,
        "output_hash": out_desc.content_hash,
    }
    return out, meta


def transform(
    tensor: ArrayLike,
    source: Optional[TensorSpec] = None,
    target: Optional[TensorSpec] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """High-level source→target transform using TensorSpec constraints.

    Applies dtype conversion, range mapping, and clipping as required by target.
    All steps are recorded. Prefer `adapt()` from compatibility for full reports.
    """
    from .compatibility import adapt
    if target is None:
        arr = as_numpy(tensor, copy=True)
        return arr, {"method": "identity", "provenance": []}
    adapted, report = adapt(tensor, target)
    return adapted, {
        "compatible": report.compatible,
        "issues": report.issues,
        "transformations": report.transformations,
        "provenance": report.provenance.to_list(),
        "error_metrics": report.error_metrics,
    }
