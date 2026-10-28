"""Neuralese-compatible latent-state representation contract.

Versioned, validated description of latent-state tensors exchanged between
compatible neural-network components. Reuses TensorGate content hashing,
descriptors, compatibility reports, and computational lineage.

This module does NOT:
- load models or run inference
- claim semantic equivalence across models/layers from shape/dtype alone
- invent a new hash algorithm or descriptor format
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from . import __software_id__, __version__
from .descriptor import TensorDescriptor, build_descriptor, _content_hash
from .errors import CompatibilityError, SerializationError, ValidationError
from .lineage import (
    ComputationKind,
    ComputationalLineage,
    TensorIdentity,
    build_lineage,
    identity_from_descriptor,
)
from .tensor import ArrayLike, as_numpy

LATENT_SCHEMA_ID = "tensorgate.latent_state"
LATENT_SCHEMA_VERSION = "1.0.0"
_SUPPORTED_MAJOR = 1


class LatentCompatibilityLevel(str, Enum):
    EXACT = "exact"
    NUMERICAL_CONVERTIBLE = "numerical_convertible"
    INCOMPATIBLE = "incompatible"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class LatentStateSpec:
    schema_id: str = LATENT_SCHEMA_ID
    schema_version: str = LATENT_SCHEMA_VERSION
    shape: Tuple[int, ...] = ()
    dtype: str = ""
    layout: str = "C"
    representation_semantics: str = ""
    model_id: str = ""
    model_revision: str = ""
    architecture_id: str = ""
    runtime_id: str = ""
    layer_id: str = ""
    sequence_position: Optional[int] = None
    mask_id: str = ""
    reference_id: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_id", str(self.schema_id or LATENT_SCHEMA_ID))
        object.__setattr__(self, "schema_version", str(self.schema_version or LATENT_SCHEMA_VERSION))
        shape = self.shape
        if isinstance(shape, list):
            shape = tuple(shape)
        object.__setattr__(self, "shape", tuple(int(x) for x in shape))
        object.__setattr__(self, "dtype", str(self.dtype or ""))
        object.__setattr__(self, "layout", str(self.layout or "C"))
        object.__setattr__(self, "representation_semantics", str(self.representation_semantics or ""))
        object.__setattr__(self, "model_id", str(self.model_id or ""))
        object.__setattr__(self, "model_revision", str(self.model_revision or ""))
        object.__setattr__(self, "architecture_id", str(self.architecture_id or ""))
        object.__setattr__(self, "runtime_id", str(self.runtime_id or ""))
        object.__setattr__(self, "layer_id", str(self.layer_id or ""))
        if self.sequence_position is not None:
            object.__setattr__(self, "sequence_position", int(self.sequence_position))
        object.__setattr__(self, "mask_id", str(self.mask_id or ""))
        object.__setattr__(self, "reference_id", str(self.reference_id or ""))
        object.__setattr__(self, "extras", dict(self.extras or {}))
        self.validate()

    def validate(self) -> None:
        if self.schema_id != LATENT_SCHEMA_ID:
            raise ValidationError(
                f"unsupported schema_id {self.schema_id!r}; expected {LATENT_SCHEMA_ID!r}"
            )
        major = _schema_major(self.schema_version)
        if major is None or major != _SUPPORTED_MAJOR:
            raise ValidationError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"supported major={_SUPPORTED_MAJOR}"
            )
        if not self.dtype:
            raise ValidationError("dtype is required")
        try:
            np.dtype(self.dtype)
        except Exception as exc:
            raise ValidationError(f"invalid dtype {self.dtype!r}: {exc}") from exc
        if any(int(d) < 0 for d in self.shape):
            raise ValidationError(f"shape dimensions must be non-negative: {self.shape}")
        if self.layout not in ("C", "F", "unknown"):
            raise ValidationError(
                f"layout must be 'C', 'F', or 'unknown'; got {self.layout!r}"
            )

    def identity_dict(self) -> dict:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "layout": self.layout,
            "representation_semantics": self.representation_semantics,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "architecture_id": self.architecture_id,
            "layer_id": self.layer_id,
            "sequence_position": self.sequence_position,
        }

    def to_dict(self) -> dict:
        d = self.identity_dict()
        d["runtime_id"] = self.runtime_id
        d["mask_id"] = self.mask_id
        d["reference_id"] = self.reference_id
        d["extras"] = dict(self.extras)
        return d

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "LatentStateSpec":
        if not isinstance(d, dict):
            raise ValidationError("LatentStateSpec requires a dict")
        shape = d.get("shape") or ()
        if isinstance(shape, list):
            shape = tuple(shape)
        return cls(
            schema_id=str(d.get("schema_id") or LATENT_SCHEMA_ID),
            schema_version=str(d.get("schema_version") or LATENT_SCHEMA_VERSION),
            shape=tuple(shape),
            dtype=str(d.get("dtype") or ""),
            layout=str(d.get("layout") or "C"),
            representation_semantics=str(d.get("representation_semantics") or ""),
            model_id=str(d.get("model_id") or ""),
            model_revision=str(d.get("model_revision") or ""),
            architecture_id=str(d.get("architecture_id") or ""),
            runtime_id=str(d.get("runtime_id") or ""),
            layer_id=str(d.get("layer_id") or ""),
            sequence_position=d.get("sequence_position"),
            mask_id=str(d.get("mask_id") or ""),
            reference_id=str(d.get("reference_id") or ""),
            extras=dict(d.get("extras") or {}),
        )


def _schema_major(version: str) -> Optional[int]:
    try:
        return int(str(version).split(".")[0])
    except (TypeError, ValueError, IndexError):
        return None


def latent_content_hash(arr: np.ndarray, spec: LatentStateSpec) -> str:
    """Content hash: tensor bytes + identity-relevant latent metadata."""
    base = _content_hash(arr, {"layout": spec.layout})
    h = hashlib.sha256()
    h.update(base.encode("utf-8"))
    meta = {
        "architecture_id": spec.architecture_id,
        "layer_id": spec.layer_id,
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "representation_semantics": spec.representation_semantics,
        "schema_id": spec.schema_id,
        "schema_version": spec.schema_version,
        "sequence_position": spec.sequence_position,
    }
    for key in sorted(meta.keys()):
        val = meta[key]
        if val is None or val == "":
            continue
        h.update(f"{key}={val}".encode("utf-8"))
    return h.hexdigest()


def latent_identity(arr: ArrayLike, spec: LatentStateSpec, *, label: str = "") -> TensorIdentity:
    a = as_numpy(arr)
    if tuple(int(x) for x in a.shape) != tuple(spec.shape):
        raise ValidationError(
            f"array shape {a.shape} does not match LatentStateSpec.shape {spec.shape}"
        )
    if str(np.dtype(a.dtype)) != str(np.dtype(spec.dtype)):
        raise ValidationError(
            f"array dtype {a.dtype} does not match LatentStateSpec.dtype {spec.dtype}"
        )
    h = latent_content_hash(a, spec)
    return TensorIdentity(
        content_hash=h,
        shape=tuple(spec.shape),
        dtype=str(spec.dtype),
        schema_id=spec.schema_id,
        schema_version=spec.schema_version,
        label=label or spec.layer_id or "latent",
    )


@dataclass
class LatentStateDescription:
    tensor: TensorDescriptor
    spec: LatentStateSpec
    latent_id: str
    software_id: str = __software_id__

    def to_dict(self) -> dict:
        return {
            "tensor": self.tensor.to_dict(),
            "spec": self.spec.to_dict(),
            "latent_id": self.latent_id,
            "software_id": self.software_id,
        }


def describe_latent_state(
    arr: ArrayLike,
    spec: Optional[LatentStateSpec] = None,
    *,
    model_id: str = "",
    model_revision: str = "",
    architecture_id: str = "",
    layer_id: str = "",
    sequence_position: Optional[int] = None,
    representation_semantics: str = "",
    layout: str = "C",
) -> LatentStateDescription:
    a = as_numpy(arr)
    if spec is None:
        spec = LatentStateSpec(
            shape=tuple(int(x) for x in a.shape),
            dtype=str(np.dtype(a.dtype)),
            layout=layout,
            representation_semantics=representation_semantics,
            model_id=model_id,
            model_revision=model_revision,
            architecture_id=architecture_id,
            layer_id=layer_id,
            sequence_position=sequence_position,
        )
    else:
        if tuple(spec.shape) != tuple(int(x) for x in a.shape):
            raise ValidationError(f"spec.shape {spec.shape} != array shape {a.shape}")
        if str(np.dtype(spec.dtype)) != str(np.dtype(a.dtype)):
            raise ValidationError(f"spec.dtype {spec.dtype} != array dtype {a.dtype}")
    desc = build_descriptor(a)
    lid = latent_content_hash(a, spec)
    return LatentStateDescription(tensor=desc, spec=spec, latent_id=lid)


@dataclass
class LatentCompatibilityReport:
    level: LatentCompatibilityLevel
    compatible: bool
    issues: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    source_spec: Optional[Dict[str, Any]] = None
    target_spec: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "compatible": self.compatible,
            "issues": list(self.issues),
            "notes": list(self.notes),
            "source_spec": self.source_spec,
            "target_spec": self.target_spec,
        }


def check_latent_compatibility(
    source: LatentStateSpec,
    target: LatentStateSpec,
    *,
    allow_dtype_conversion: bool = False,
) -> LatentCompatibilityReport:
    if not isinstance(source, LatentStateSpec):
        source = LatentStateSpec.from_dict(source)  # type: ignore[arg-type]
    if not isinstance(target, LatentStateSpec):
        target = LatentStateSpec.from_dict(target)  # type: ignore[arg-type]

    issues: List[str] = []
    notes: List[str] = []

    if source.schema_id != target.schema_id:
        issues.append(f"schema_id mismatch: {source.schema_id!r} vs {target.schema_id!r}")
    src_major = _schema_major(source.schema_version)
    tgt_major = _schema_major(target.schema_version)
    if src_major != tgt_major:
        issues.append(
            f"schema major mismatch: {source.schema_version!r} vs {target.schema_version!r}"
        )
    if tuple(source.shape) != tuple(target.shape):
        issues.append(f"shape mismatch: {source.shape} vs {target.shape}")

    if source.model_id and target.model_id and source.model_id != target.model_id:
        issues.append(
            f"model_id mismatch: {source.model_id!r} vs {target.model_id!r} "
            "(matching shape does not imply semantic compatibility)"
        )
    if source.model_revision and target.model_revision and source.model_revision != target.model_revision:
        issues.append(
            f"model_revision mismatch: {source.model_revision!r} vs {target.model_revision!r}"
        )
    if source.architecture_id and target.architecture_id and source.architecture_id != target.architecture_id:
        issues.append(
            f"architecture_id mismatch: {source.architecture_id!r} vs {target.architecture_id!r}"
        )
    if source.layer_id and target.layer_id and source.layer_id != target.layer_id:
        issues.append(
            f"layer_id mismatch: {source.layer_id!r} vs {target.layer_id!r} "
            "(matching shape does not imply semantic compatibility)"
        )
    if (
        source.representation_semantics
        and target.representation_semantics
        and source.representation_semantics != target.representation_semantics
    ):
        issues.append(
            f"representation_semantics mismatch: "
            f"{source.representation_semantics!r} vs {target.representation_semantics!r}"
        )

    missing_meta = False
    if not source.model_id or not target.model_id:
        missing_meta = True
        notes.append("model_id missing on one or both sides; cannot claim producer match")
    if not source.layer_id or not target.layer_id:
        missing_meta = True
        notes.append("layer_id missing on one or both sides; cannot claim layer match")
    if not source.representation_semantics or not target.representation_semantics:
        missing_meta = True
        notes.append("representation_semantics missing; semantic interchangeability unproven")

    dtype_match = str(np.dtype(source.dtype)) == str(np.dtype(target.dtype))
    layout_match = source.layout == target.layout or "unknown" in (source.layout, target.layout)

    if not dtype_match:
        if allow_dtype_conversion:
            notes.append(
                f"dtype differs ({source.dtype} vs {target.dtype}); "
                "conversion permitted but must be explicit and recorded"
            )
        else:
            issues.append(f"dtype mismatch: {source.dtype} vs {target.dtype}")

    if not layout_match and source.layout != "unknown" and target.layout != "unknown":
        if allow_dtype_conversion:
            notes.append(f"layout differs ({source.layout} vs {target.layout})")
        else:
            issues.append(f"layout mismatch: {source.layout} vs {target.layout}")

    src_d = source.to_dict()
    tgt_d = target.to_dict()

    if issues:
        return LatentCompatibilityReport(
            level=LatentCompatibilityLevel.INCOMPATIBLE,
            compatible=False,
            issues=issues,
            notes=notes,
            source_spec=src_d,
            target_spec=tgt_d,
        )

    if not dtype_match and allow_dtype_conversion:
        return LatentCompatibilityReport(
            level=LatentCompatibilityLevel.NUMERICAL_CONVERTIBLE,
            compatible=True,
            issues=[],
            notes=notes,
            source_spec=src_d,
            target_spec=tgt_d,
        )

    if missing_meta:
        return LatentCompatibilityReport(
            level=LatentCompatibilityLevel.INCONCLUSIVE,
            compatible=False,
            issues=[],
            notes=notes
            + [
                "INCONCLUSIVE: required producer/layer/semantics metadata incomplete; "
                "do not treat as interchangeable"
            ],
            source_spec=src_d,
            target_spec=tgt_d,
        )

    return LatentCompatibilityReport(
        level=LatentCompatibilityLevel.EXACT,
        compatible=True,
        issues=[],
        notes=notes,
        source_spec=src_d,
        target_spec=tgt_d,
    )


def compare_latent_states(
    a: ArrayLike,
    b: ArrayLike,
    spec_a: LatentStateSpec,
    spec_b: LatentStateSpec,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Dict[str, Any]:
    arr_a = as_numpy(a)
    arr_b = as_numpy(b)
    compat = check_latent_compatibility(spec_a, spec_b)
    result: Dict[str, Any] = {
        "compatibility": compat.to_dict(),
        "shape_match": tuple(arr_a.shape) == tuple(arr_b.shape),
        "dtype_a": str(arr_a.dtype),
        "dtype_b": str(arr_b.dtype),
        "metrics": {},
        "status": "skipped",
    }
    if tuple(arr_a.shape) != tuple(arr_b.shape):
        result["status"] = "incompatible_shape"
        result["issues"] = [f"array shape mismatch: {arr_a.shape} vs {arr_b.shape}"]
        return result
    if not np.issubdtype(arr_a.dtype, np.number) or not np.issubdtype(arr_b.dtype, np.number):
        result["status"] = "non_numeric"
        return result
    x = arr_a.astype(np.float64, copy=False)
    y = arr_b.astype(np.float64, copy=False)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    n_finite = int(np.count_nonzero(finite_mask))
    n_total = int(x.size) if x.size else 0
    if n_finite == 0:
        result["status"] = "no_finite_values"
        result["metrics"] = {
            "n_finite": 0,
            "n_total": n_total,
            "max_abs_diff": None,
            "rmse": None,
            "cosine": None,
        }
        return result
    dx = x[finite_mask]
    dy = y[finite_mask]
    diff = dx - dy
    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    na = float(np.linalg.norm(dx))
    nb = float(np.linalg.norm(dy))
    cosine = float(np.dot(dx, dy) / (na * nb)) if na > 0 and nb > 0 else None
    close = bool(np.allclose(dx, dy, rtol=rtol, atol=atol))
    result["metrics"] = {
        "n_finite": n_finite,
        "n_total": n_total,
        "max_abs_diff": max_abs,
        "rmse": rmse,
        "cosine": cosine,
        "allclose": close,
        "rtol": rtol,
        "atol": atol,
    }
    result["status"] = "compared"
    return result


def serialize_latent_state(
    arr: ArrayLike,
    spec: LatentStateSpec,
    *,
    label: str = "",
    include_array: bool = True,
) -> dict:
    a = as_numpy(arr)
    if tuple(int(x) for x in a.shape) != tuple(spec.shape):
        raise ValidationError(f"array shape {a.shape} != spec.shape {spec.shape}")
    if str(np.dtype(a.dtype)) != str(np.dtype(spec.dtype)):
        raise ValidationError(f"array dtype {a.dtype} != spec.dtype {spec.dtype}")
    identity = latent_identity(a, spec, label=label)
    envelope: Dict[str, Any] = {
        "schema_id": LATENT_SCHEMA_ID,
        "schema_version": LATENT_SCHEMA_VERSION,
        "software_id": __software_id__,
        "spec": spec.to_dict(),
        "identity": identity.to_dict(),
        "latent_id": identity.content_hash,
    }
    if include_array:
        envelope["data"] = a.tolist()
    return envelope


def deserialize_latent_state(
    payload: Union[dict, str, bytes],
    *,
    require_data: bool = False,
) -> Tuple[Optional[np.ndarray], LatentStateSpec, TensorIdentity]:
    if isinstance(payload, (str, bytes)):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SerializationError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SerializationError("latent-state payload must be a dict")
    schema_id = str(payload.get("schema_id") or "")
    schema_version = str(payload.get("schema_version") or "")
    if schema_id and schema_id != LATENT_SCHEMA_ID:
        raise SerializationError(
            f"unsupported schema_id {schema_id!r}; expected {LATENT_SCHEMA_ID!r}"
        )
    major = _schema_major(schema_version or LATENT_SCHEMA_VERSION)
    if major != _SUPPORTED_MAJOR:
        raise SerializationError(f"unsupported schema major version {schema_version!r}")
    try:
        spec = LatentStateSpec.from_dict(payload.get("spec") or {})
    except ValidationError as exc:
        raise SerializationError(str(exc)) from exc
    arr: Optional[np.ndarray] = None
    if "data" in payload and payload["data"] is not None:
        try:
            arr = np.asarray(payload["data"], dtype=np.dtype(spec.dtype))
        except Exception as exc:
            raise SerializationError(f"cannot materialize data array: {exc}") from exc
        if tuple(arr.shape) != tuple(spec.shape):
            raise SerializationError(
                f"data shape {arr.shape} inconsistent with spec.shape {spec.shape}"
            )
    elif require_data:
        raise SerializationError("payload missing required data array")
    declared = payload.get("identity") or {}
    identity = TensorIdentity.from_dict(declared) or TensorIdentity(
        content_hash=str(payload.get("latent_id") or ""),
        shape=tuple(spec.shape),
        dtype=spec.dtype,
        schema_id=spec.schema_id,
        schema_version=spec.schema_version,
    )
    if arr is not None:
        recomputed = latent_content_hash(arr, spec)
        declared_hash = identity.content_hash or str(payload.get("latent_id") or "")
        if declared_hash and declared_hash != recomputed:
            raise SerializationError(
                f"latent identity mismatch: declared {declared_hash}, recomputed {recomputed}"
            )
        identity = TensorIdentity(
            content_hash=recomputed,
            shape=tuple(spec.shape),
            dtype=spec.dtype,
            schema_id=spec.schema_id,
            schema_version=spec.schema_version,
            label=identity.label,
        )
    return arr, spec, identity


def build_latent_lineage(
    *,
    operation: str,
    input_identity: TensorIdentity,
    output_identity: TensorIdentity,
    kind: ComputationKind = ComputationKind.NONDETERMINISTIC,
    model_id: str = "",
    model_version: str = "",
    layer_id: str = "",
    sequence_position: Optional[int] = None,
    parameters: Optional[Dict[str, Any]] = None,
    evidence_refs: Optional[List[Dict[str, Any]]] = None,
    notes: str = "",
) -> ComputationalLineage:
    params = dict(parameters or {})
    if layer_id:
        params.setdefault("layer_id", layer_id)
    if sequence_position is not None:
        params.setdefault("sequence_position", sequence_position)
    return build_lineage(
        operation=operation,
        input_hashes=[input_identity.content_hash],
        output_hash=output_identity.content_hash,
        parameters=params,
        kind=kind,
        model_id=model_id,
        model_version=model_version,
        input_identities=[input_identity],
        output_identity=output_identity,
        evidence_refs=evidence_refs,
        notes=notes,
    )
