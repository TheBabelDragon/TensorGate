"""Thin WaveBridge adapter.

TensorGate does NOT know about WAV, lasers, PAM, BPW34, or optical modulation.
This adapter only prepares / describes numerical arrays at the boundary:

    Tensor
      → TensorGate analysis
      → explicit normalization / adaptation
      → numerical payload for WaveBridge
      → (WaveBridge owns transport)

    WaveBridge recovered numerical payload
      → TensorGate description
      → comparison / provenance

WaveBridge remains the owner of physical transport and PCM encoding.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..compatibility import compare
from ..inspect import analyze
from ..normalize import normalize
from ..tensor import ArrayLike, as_numpy


def to_wavebridge_payload(
    tensor: ArrayLike,
    *,
    method: str = "symmetric",
    target_min: float = -1.0,
    target_max: float = 1.0,
    dtype: str = "float32",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Prepare a tensor as a numerical payload suitable for WaveBridge.

    Returns (normalized_array, metadata) where metadata carries:
      - source_descriptor / tensorgate_descriptor
      - full normalization provenance
      - content hashes

    Caller hands the array to WaveBridge (encode_field / encode_field_state).
    No PCM, WAV, or physical encoding is performed here.
    """
    arr = as_numpy(tensor)
    source_desc = analyze(arr)
    out, norm_meta = normalize(
        arr, method=method, target_min=target_min, target_max=target_max
    )
    out = out.astype(np.dtype(dtype), copy=False)
    payload_desc = analyze(out)
    meta: Dict[str, Any] = {
        "source_descriptor": source_desc.to_dict(),
        "tensorgate_descriptor": payload_desc.to_dict(),
        "normalization": {
            k: v for k, v in norm_meta.items() if k != "output_descriptor"
        },
        "payload_dtype": str(out.dtype),
        "note": "Numerical payload only. Physical encoding is WaveBridge's concern.",
    }
    return out, meta


def from_wavebridge_payload(
    payload: ArrayLike,
    *,
    source_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Accept a numerical array recovered from WaveBridge and re-describe it.

    Does not invert physical channel effects — that is WaveBridge.
    """
    arr = as_numpy(payload)
    desc = analyze(arr)
    meta: Dict[str, Any] = {
        "tensorgate_descriptor": desc.to_dict(),
        "source_meta": source_meta or {},
        "note": "Recovered numerical array. Channel inversion belongs to WaveBridge.",
    }
    return arr, meta


def compare_roundtrip(
    original: ArrayLike,
    recovered: ArrayLike,
) -> Dict[str, Any]:
    """Explicit numerical comparison of source vs recovered payload."""
    return compare(original, recovered)
