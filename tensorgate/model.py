"""Hugging Face / Safetensors multi-shard model inspection.

This module provides first-class support for inspecting local Hugging Face
model directories that use the standard safetensors layout:

  model.safetensors.index.json   — weight_map + metadata
  model-0000N-of-0000M.safetensors
  config.json                    — optional model config

Design constraints (v0):
  - Never load the entire model into RAM.
  - Inspect one tensor at a time via safetensors.safe_open.
  - Reuse TensorSpec / TensorDescriptor / provenance machinery.
  - Produce a deterministic ModelManifest (content-addressable).
  - No inference, no Transformers, no quantization execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import __software_id__, __version__
from .errors import SerializationError, ValidationError
from .tensor import TensorSpec


def _require_safetensors():
    try:
        from safetensors import safe_open  # noqa: F401
        return True
    except ImportError as e:
        raise ImportError(
            "safetensors is required for model inspection. "
            'Install with: pip install "tensorgate[safetensors]" or pip install safetensors'
        ) from e


@dataclass(frozen=True)
class TensorMeta:
    """Lightweight TensorSpec-compatible description of one weight tensor.

    Does not hold the array data — only structural metadata.
    """

    name: str
    shape: Tuple[int, ...]
    dtype: str
    element_count: int
    source_shard: str
    byte_size: int = 0
    finite: bool = True
    nan_count: int = 0
    pos_inf_count: int = 0
    neg_inf_count: int = 0

    def to_tensor_spec(self) -> TensorSpec:
        return TensorSpec(
            dtype=self.dtype,
            shape=self.shape,
            rank=len(self.shape),
            allow_nan=self.nan_count > 0,
            allow_inf=(self.pos_inf_count + self.neg_inf_count) > 0,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "element_count": self.element_count,
            "source_shard": self.source_shard,
            "byte_size": self.byte_size,
            "finite": self.finite,
            "nan_count": self.nan_count,
            "pos_inf_count": self.pos_inf_count,
            "neg_inf_count": self.neg_inf_count,
        }


@dataclass
class ShardInfo:
    """One safetensors shard file."""

    filename: str
    path: str
    sha256: str
    size_bytes: int
    tensor_names: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "tensor_names": list(self.tensor_names),
        }


@dataclass
class ModelManifest:
    """Deterministic, content-addressable description of a multi-shard model."""

    model_id: str
    config: Dict[str, Any]
    shards: List[ShardInfo]
    tensors: List[TensorMeta]
    total_parameters: int
    total_bytes: int
    manifest_hash: str = ""
    software_id: str = ""
    tensorgate_version: str = ""
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "config": self.config,
            "shards": [s.to_dict() for s in self.shards],
            "tensors": [t.to_dict() for t in self.tensors],
            "total_parameters": self.total_parameters,
            "total_bytes": self.total_bytes,
            "manifest_hash": self.manifest_hash,
            "software_id": self.software_id,
            "tensorgate_version": self.tensorgate_version,
            "validation_errors": list(self.validation_errors),
            "validation_warnings": list(self.validation_warnings),
        }

    def summary(self) -> str:
        lines = [
            f"model_id={self.model_id}",
            f"shards={len(self.shards)}  tensors={len(self.tensors)}",
            f"total_parameters={self.total_parameters:,}  total_bytes={self.total_bytes:,}",
            f"manifest_hash={self.manifest_hash[:16]}…" if self.manifest_hash else "manifest_hash=",
        ]
        if self.config:
            arch = self.config.get("architectures") or self.config.get("model_type")
            if arch:
                lines.append(f"architecture={arch}")
        if self.validation_errors:
            lines.append(f"ERRORS ({len(self.validation_errors)}):")
            for e in self.validation_errors[:10]:
                lines.append(f"  - {e}")
        if self.validation_warnings:
            lines.append(f"warnings ({len(self.validation_warnings)}):")
            for w in self.validation_warnings[:5]:
                lines.append(f"  - {w}")
        lines.append("tensors (first 8):")
        for t in self.tensors[:8]:
            lines.append(
                f"  {t.name}: shape={t.shape} dtype={t.dtype} "
                f"elems={t.element_count:,} shard={t.source_shard}"
            )
        if len(self.tensors) > 8:
            lines.append(f"  … and {len(self.tensors) - 8} more")
        return "\n".join(lines)


def _file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _dtype_str(dtype) -> str:
    return str(np.dtype(dtype))


def _inspect_tensor_array(arr: np.ndarray) -> Tuple[bool, int, int, int]:
    """Return (finite, nan_count, pos_inf, neg_inf)."""
    if not (np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.complexfloating)):
        return True, 0, 0, 0
    flat = arr.ravel()
    nan_count = int(np.isnan(flat).sum())
    pos_inf = int(np.isposinf(flat).sum())
    neg_inf = int(np.isneginf(flat).sum())
    finite = (nan_count + pos_inf + neg_inf) == 0
    return finite, nan_count, pos_inf, neg_inf


def _canonical_manifest_payload(manifest: ModelManifest) -> dict:
    """Stable subset used for the overall manifest hash."""
    return {
        "model_id": manifest.model_id,
        "config": manifest.config,
        "shards": [
            {
                "filename": s.filename,
                "sha256": s.sha256,
                "size_bytes": s.size_bytes,
                "tensor_names": sorted(s.tensor_names),
            }
            for s in sorted(manifest.shards, key=lambda x: x.filename)
        ],
        "tensors": [
            {
                "name": t.name,
                "shape": list(t.shape),
                "dtype": t.dtype,
                "element_count": t.element_count,
                "source_shard": t.source_shard,
                "byte_size": t.byte_size,
            }
            for t in sorted(manifest.tensors, key=lambda x: x.name)
        ],
        "total_parameters": manifest.total_parameters,
        "total_bytes": manifest.total_bytes,
    }


def _manifest_hash(manifest: ModelManifest) -> str:
    payload = _canonical_manifest_payload(manifest)
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _load_index(model_dir: Path) -> dict:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        single = model_dir / "model.safetensors"
        if single.is_file():
            return {
                "metadata": {},
                "weight_map": {},
                "_single_file": "model.safetensors",
            }
        raise SerializationError(
            f"Neither model.safetensors.index.json nor model.safetensors found in {model_dir}"
        )
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SerializationError(f"Malformed model.safetensors.index.json: {e}") from e
    if not isinstance(data, dict):
        raise SerializationError("model.safetensors.index.json root must be an object")
    if "weight_map" not in data or not isinstance(data["weight_map"], dict):
        raise SerializationError("model.safetensors.index.json missing or invalid 'weight_map'")
    return data


def _load_config(model_dir: Path) -> Dict[str, Any]:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_parse_error": True}


def inspect_model(
    model_dir: Union[str, Path],
    *,
    model_id: Optional[str] = None,
    check_values: bool = True,
    raise_on_error: bool = False,
) -> ModelManifest:
    """Inspect a local Hugging Face / safetensors model directory.

    Never loads the entire model into RAM. Processes one tensor at a time.
    """
    _require_safetensors()
    from safetensors import safe_open

    model_dir = Path(model_dir).resolve()
    if not model_dir.is_dir():
        raise SerializationError(f"Model directory does not exist: {model_dir}")

    model_id = model_id or model_dir.name
    config = _load_config(model_dir)
    index = _load_index(model_dir)

    errors: List[str] = []
    warnings: List[str] = []

    weight_map: Dict[str, str] = dict(index.get("weight_map") or {})
    single_file = index.get("_single_file")

    if single_file:
        shard_names = [single_file]
    else:
        shard_names = sorted(set(weight_map.values()))

    on_disk = sorted(
        p.name for p in model_dir.glob("*.safetensors")
        if p.is_file() and p.name not in shard_names and p.name != "model.safetensors"
    )
    for extra in on_disk:
        warnings.append(f"Extra safetensors file not listed in index: {extra}")
        shard_names.append(extra)

    shards: List[ShardInfo] = []
    tensors: List[TensorMeta] = []
    seen_tensor_names: Dict[str, str] = {}

    for shard_name in shard_names:
        shard_path = model_dir / shard_name
        if not shard_path.is_file():
            msg = f"Missing shard: {shard_name}"
            errors.append(msg)
            if raise_on_error:
                raise ValidationError(msg)
            continue

        sha = _file_sha256(shard_path)
        size = shard_path.stat().st_size
        shard_tensor_names: List[str] = []

        try:
            with safe_open(str(shard_path), framework="np") as f:
                keys = list(f.keys())
                if single_file and not weight_map:
                    for k in keys:
                        weight_map[k] = shard_name

                for key in keys:
                    if key in seen_tensor_names:
                        msg = (
                            f"Duplicate tensor '{key}' in shard '{shard_name}' "
                            f"(already seen in '{seen_tensor_names[key]}')"
                        )
                        errors.append(msg)
                        if raise_on_error:
                            raise ValidationError(msg)
                        continue
                    seen_tensor_names[key] = shard_name
                    shard_tensor_names.append(key)

                    mapped = weight_map.get(key)
                    if mapped is not None and mapped != shard_name:
                        warnings.append(
                            f"Tensor '{key}' found in '{shard_name}' but index maps it to '{mapped}'"
                        )

                    try:
                        arr = f.get_tensor(key)
                    except Exception as e:
                        msg = f"Failed to read tensor '{key}' from {shard_name}: {e}"
                        errors.append(msg)
                        if raise_on_error:
                            raise ValidationError(msg) from e
                        continue

                    shape = tuple(int(s) for s in arr.shape)
                    dtype = _dtype_str(arr.dtype)
                    n_elem = int(arr.size)
                    byte_size = int(arr.nbytes)

                    finite, nan_c, pos_inf, neg_inf = True, 0, 0, 0
                    if check_values:
                        finite, nan_c, pos_inf, neg_inf = _inspect_tensor_array(arr)
                        if not finite:
                            warnings.append(
                                f"Non-finite values in '{key}': nan={nan_c} +inf={pos_inf} -inf={neg_inf}"
                            )

                    tensors.append(
                        TensorMeta(
                            name=key,
                            shape=shape,
                            dtype=dtype,
                            element_count=n_elem,
                            source_shard=shard_name,
                            byte_size=byte_size,
                            finite=finite,
                            nan_count=nan_c,
                            pos_inf_count=pos_inf,
                            neg_inf_count=neg_inf,
                        )
                    )
                    del arr
        except Exception as e:
            msg = f"Cannot open shard {shard_name}: {e}"
            errors.append(msg)
            if raise_on_error:
                raise ValidationError(msg) from e
            continue

        shards.append(
            ShardInfo(
                filename=shard_name,
                path=str(shard_path),
                sha256=sha,
                size_bytes=size,
                tensor_names=sorted(shard_tensor_names),
            )
        )

    index_keys = set(weight_map.keys())
    discovered = set(seen_tensor_names.keys())
    missing_in_shards = index_keys - discovered
    extra_in_shards = discovered - index_keys
    for name in sorted(missing_in_shards):
        mapped_shard = weight_map.get(name)
        if mapped_shard and (model_dir / mapped_shard).is_file():
            msg = f"Index lists tensor '{name}' in '{mapped_shard}' but it was not found in that shard"
            errors.append(msg)
            if raise_on_error:
                raise ValidationError(msg)
    for name in sorted(extra_in_shards):
        if not single_file:
            warnings.append(f"Tensor '{name}' present in shards but absent from index weight_map")

    tensors.sort(key=lambda t: t.name)
    shards.sort(key=lambda s: s.filename)

    total_params = sum(t.element_count for t in tensors)
    total_bytes = sum(s.size_bytes for s in shards)

    manifest = ModelManifest(
        model_id=model_id,
        config=config,
        shards=shards,
        tensors=tensors,
        total_parameters=total_params,
        total_bytes=total_bytes,
        software_id=__software_id__,
        tensorgate_version=__version__,
        validation_errors=errors,
        validation_warnings=warnings,
    )
    manifest.manifest_hash = _manifest_hash(manifest)
    return manifest


def load_model_directory(
    model_dir: Union[str, Path],
    **kwargs,
) -> ModelManifest:
    """Alias for inspect_model — accepts a local Hugging Face model directory."""
    return inspect_model(model_dir, **kwargs)
