"""Thin WaveBridge adapter.

TensorGate does NOT know about WAV, lasers, PAM, BPW34, or optical modulation.
This adapter only:
  Tensor → TensorGate descriptor / transformation → WaveBridge-compatible numerical payload
and the reverse.

WaveBridge remains the owner of physical transport.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..inspect import analyze
from ..normalize import normalize
from ..tensor import as_numpy, ArrayLike, TensorSpec
from ..compatibility import adapt, compare


def to_wavebridge_payload(
    tensor: ArrayLike,
    *,
    method: str = "symmetric",
    target_min: float = -1.0,
    target_max: float = 1.0,
    dtype: str = "float32",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Prepare a tensor as a numerical payload suitable for WaveBridge transport.

    Returns normalized array + full TensorGate metadata (descriptor, provenance).
    Caller is responsible for handing the array to WaveBridge.
    """
    arr = as_numpy(tensor)
    desc = analyze(arr)
    out, meta = normalize(arr, method=method, target_min=target_min, target_max=target_max)
    out = out.astype(np.dtype(dtype))
    payload_meta = {
        "tensorgate_descriptor": analyze(out).to_dict(),
        "source_descriptor": desc.to_dict(),
        "normalization": meta,
        "note": "Numerical payload only. Physical encoding is WaveBridge's concern.",
    }
    return out, payload_meta


def from_wavebridge_payload(
    payload: ArrayLike,
    *,
    source_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Accept a numerical array recovered from WaveBridge and re-describe it.

    Does not attempt to invert physical channel effects — that is WaveBridge.
    """
    arr = as_numpy(payload)
    desc = analyze(arr)
    meta = {
        "tensorgate_descriptor": desc.to_dict(),
        "source_meta": source_meta or {},
        "note": "Recovered numerical array. Channel inversion belongs to WaveBridge.",
    }
    return arr, meta
