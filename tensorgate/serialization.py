"""Deterministic TensorGate artifact format (v0).

Simple, transparent layout:
  - <path>.npy          : raw array (numpy)
  - <path>.tg.json      : descriptor + provenance + transformation history

No heavy binary container in v0. Sufficient for reproducibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .descriptor import TensorDescriptor, build_descriptor
from .errors import SerializationError
from .inspect import analyze
from .provenance import ProvenanceRecord
from .tensor import as_numpy, ArrayLike
from . import __software_id__, __version__


def save_tensor(
    tensor: ArrayLike,
    path: Union[str, Path],
    *,
    descriptor: Optional[TensorDescriptor] = None,
    provenance: Optional[list] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save tensor data + metadata side-car.

    Writes:
      path with .npy suffix for data
      path with .tg.json suffix for metadata
    """
    path = Path(path)
    arr = as_numpy(tensor)
    if descriptor is None:
        descriptor = analyze(arr)

    data_path = path.with_suffix(".npy")
    meta_path = path.with_suffix(".tg.json")

    np.save(data_path, arr)

    meta = {
        "tensorgate_version": __version__,
        "software_id": __software_id__,
        "descriptor": descriptor.to_dict(),
        "provenance": provenance or descriptor.provenance,
        "extra": extra or {},
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return data_path


def load_tensor(
    path: Union[str, Path],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load tensor + metadata. Returns (array, meta_dict)."""
    path = Path(path)
    if path.suffix == ".npy":
        data_path = path
        meta_path = path.with_suffix(".tg.json")
    elif path.suffix == ".json" and path.name.endswith(".tg.json"):
        meta_path = path
        data_path = path.with_name(path.name.replace(".tg.json", ".npy"))
    else:
        data_path = path.with_suffix(".npy")
        meta_path = path.with_suffix(".tg.json")

    if not data_path.exists():
        raise SerializationError(f"Tensor data not found: {data_path}")

    arr = np.load(data_path)
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        desc_dict = meta.get("descriptor")
        if desc_dict and "content_hash" in desc_dict:
            current = analyze(arr)
            if current.content_hash != desc_dict["content_hash"]:
                raise SerializationError(
                    f"Content hash mismatch on load: stored={desc_dict['content_hash'][:16]}… "
                    f"computed={current.content_hash[:16]}…"
                )
    return arr, meta
