"""Verified, deterministic model-artifact transformation for TensorGate.

Extends model inspection into a validated transformation workflow:

  - Accept local safetensors / HF-style model directories.
  - Validate manifests, shards, tensor names/shapes/dtypes before mutation.
  - Apply only explicitly selected, already-supported numerical transforms.
  - Never overwrite the source; write to a separate output directory.
  - Re-inspect the output and emit a machine-readable transformation report
    that conforms to the Shared Evidence Contract (computational lineage).

TensorGate remains a numerical compatibility + provenance boundary.
This module does not perform inference or network serving.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import __software_id__, __version__
from .compatibility import compare
from .descriptor import build_descriptor
from .errors import SerializationError, ValidationError
from .inspect import analyze
from .lineage import (
    ComputationKind,
    ComputationalLineage,
    TensorIdentity,
    build_lineage,
    identity_from_descriptor,
)
from .model import (
    ModelManifest,
    ShardInfo,
    TensorMeta,
    _file_sha256,
    _manifest_hash,
    _require_safetensors,
    inspect_model,
)
from .normalize import normalize
from .provenance import ProvenanceRecord, make_record
from .quantize import quantize
from .transform import clip, convert_dtype


# ---------------------------------------------------------------------------
# Supported operations (explicit, no silent fallbacks)
# ---------------------------------------------------------------------------

SUPPORTED_OPS = frozenset({
    "identity",
    "normalize",
    "convert_dtype",
    "clip",
    "quantize",
})

# Non-finite policies
NONFINITE_POLICIES = frozenset({"reject", "preserve", "error"})


@dataclass(frozen=True)
class TensorTransformSpec:
    """Explicit transformation request for one tensor (or a name pattern)."""

    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        op = self.operation.lower().strip()
        if op not in SUPPORTED_OPS:
            raise ValidationError(
                f"Unsupported transformation '{self.operation}'. "
                f"Supported: {sorted(SUPPORTED_OPS)}"
            )
        object.__setattr__(self, "operation", op)


@dataclass
class TensorTransformResult:
    """Per-tensor outcome of a verified transformation."""

    name: str
    applied: bool
    operation: str
    parameters: Dict[str, Any]
    input_shape: Tuple[int, ...]
    input_dtype: str
    input_hash: str
    output_shape: Tuple[int, ...]
    output_dtype: str
    output_hash: str
    error_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    rejected: bool = False
    reason: str = ""
    implementation_version: str = __software_id__

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "applied": self.applied,
            "operation": self.operation,
            "parameters": dict(self.parameters),
            "input_shape": list(self.input_shape),
            "input_dtype": self.input_dtype,
            "input_hash": self.input_hash,
            "output_shape": list(self.output_shape),
            "output_dtype": self.output_dtype,
            "output_hash": self.output_hash,
            "error_metrics": dict(self.error_metrics),
            "warnings": list(self.warnings),
            "rejected": self.rejected,
            "reason": self.reason,
            "implementation_version": self.implementation_version,
        }


@dataclass
class ModelTransformReport:
    """Deterministic, machine-readable report for a model transformation.

    Identity fields (hashes, shapes, dtypes, lineage) are stable.
    Volatile metadata (paths, timestamps) is excluded from identity.
    """

    source_model_id: str
    output_model_id: str
    source_manifest_hash: str
    output_manifest_hash: str
    tensors: List[TensorTransformResult]
    lineage: ComputationalLineage
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    rejected_operations: List[str] = field(default_factory=list)
    software_id: str = __software_id__
    tensorgate_version: str = __version__
    # Informational only — not part of deterministic identity
    source_path: str = ""
    output_path: str = ""

    def to_dict(self) -> dict:
        return {
            "source_model_id": self.source_model_id,
            "output_model_id": self.output_model_id,
            "source_manifest_hash": self.source_manifest_hash,
            "output_manifest_hash": self.output_manifest_hash,
            "tensors": [t.to_dict() for t in self.tensors],
            "lineage": self.lineage.to_dict(),
            "validation_errors": list(self.validation_errors),
            "validation_warnings": list(self.validation_warnings),
            "rejected_operations": list(self.rejected_operations),
            "software_id": self.software_id,
            "tensorgate_version": self.tensorgate_version,
            # paths are informational
            "source_path": self.source_path,
            "output_path": self.output_path,
        }

    def summary(self) -> str:
        applied = sum(1 for t in self.tensors if t.applied)
        rejected = sum(1 for t in self.tensors if t.rejected)
        lines = [
            f"source_model_id={self.source_model_id}",
            f"output_model_id={self.output_model_id}",
            f"source_manifest_hash={self.source_manifest_hash[:16]}…",
            f"output_manifest_hash={self.output_manifest_hash[:16]}…",
            f"tensors_total={len(self.tensors)}  applied={applied}  rejected={rejected}",
            f"software_id={self.software_id}",
        ]
        if self.validation_errors:
            lines.append(f"ERRORS ({len(self.validation_errors)}):")
            for e in self.validation_errors[:8]:
                lines.append(f"  - {e}")
        if self.rejected_operations:
            lines.append(f"rejected_ops: {self.rejected_operations}")
        if self.validation_warnings:
            lines.append(f"warnings ({len(self.validation_warnings)}):")
            for w in self.validation_warnings[:5]:
                lines.append(f"  - {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_specs(
    transformations: Dict[str, Union[TensorTransformSpec, Dict[str, Any], str]],
) -> Dict[str, TensorTransformSpec]:
    """Normalize user-supplied transformation map."""
    out: Dict[str, TensorTransformSpec] = {}
    for name, spec in transformations.items():
        if isinstance(spec, TensorTransformSpec):
            out[name] = spec
        elif isinstance(spec, str):
            out[name] = TensorTransformSpec(operation=spec)
        elif isinstance(spec, dict):
            op = spec.get("operation") or spec.get("op") or spec.get("method")
            if not op:
                raise ValidationError(
                    f"Transformation for '{name}' missing 'operation' key"
                )
            params = {k: v for k, v in spec.items() if k not in ("operation", "op")}
            # allow method as alias inside normalize
            if "method" in spec and op in ("normalize", "symmetric", "range", "maxabs", "standard"):
                if op != "normalize":
                    params["method"] = op
                    op = "normalize"
            out[name] = TensorTransformSpec(operation=str(op), parameters=params)
        else:
            raise ValidationError(
                f"Invalid transformation spec for '{name}': {type(spec)}"
            )
    return out


def _apply_one(
    arr: np.ndarray,
    spec: TensorTransformSpec,
    *,
    on_nonfinite: str,
) -> Tuple[np.ndarray, Dict[str, Any], List[str], bool, str]:
    """Apply a single supported transform. Returns (out, meta, warnings, rejected, reason)."""
    warnings: List[str] = []
    desc = analyze(arr)

    # Non-finite policy
    has_nonfinite = (desc.nan_count + desc.pos_inf_count + desc.neg_inf_count) > 0
    if has_nonfinite:
        msg = (
            f"non-finite values present (nan={desc.nan_count} "
            f"+inf={desc.pos_inf_count} -inf={desc.neg_inf_count})"
        )
        if on_nonfinite == "error":
            raise ValidationError(msg)
        if on_nonfinite == "reject":
            return arr, {}, [msg], True, msg
        # preserve
        warnings.append(msg + " — preserved per policy")

    op = spec.operation
    params = dict(spec.parameters)

    try:
        if op == "identity":
            out = arr.copy()
            meta = {
                "method": "identity",
                "input_hash": desc.content_hash,
                "output_hash": build_descriptor(out).content_hash,
            }
        elif op == "normalize":
            method = params.pop("method", "symmetric")
            out, meta = normalize(arr, method=method, **params)
        elif op == "convert_dtype":
            dtype = params.get("dtype") or params.get("target_dtype")
            if not dtype:
                raise ValidationError("convert_dtype requires 'dtype' parameter")
            casting = params.get("casting", "safe")
            out, meta = convert_dtype(arr, dtype=str(dtype), casting=casting)
        elif op == "clip":
            out, meta = clip(
                arr,
                min_value=params.get("min_value"),
                max_value=params.get("max_value"),
            )
        elif op == "quantize":
            bits = int(params.get("bits", 8))
            scheme = params.get("scheme", "symmetric")
            signed = bool(params.get("signed", True))
            out, meta = quantize(arr, bits=bits, scheme=scheme, signed=signed)
        else:
            # Should be unreachable due to TensorTransformSpec validation
            return arr, {}, [], True, f"unsupported operation '{op}'"
    except Exception as e:
        return arr, {}, [str(e)], True, str(e)

    return out, meta, warnings, False, ""


def _write_safetensors(path: Path, tensors: Dict[str, np.ndarray]) -> None:
    _require_safetensors()
    from safetensors.numpy import save_file
    save_file(tensors, str(path))


def _copy_config(source_dir: Path, output_dir: Path) -> None:
    cfg = source_dir / "config.json"
    if cfg.is_file():
        shutil.copy2(cfg, output_dir / "config.json")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transform_model(
    source_dir: Union[str, Path],
    output_dir: Union[str, Path],
    transformations: Dict[str, Union[TensorTransformSpec, Dict[str, Any], str]],
    *,
    model_id: Optional[str] = None,
    output_model_id: Optional[str] = None,
    on_nonfinite: str = "reject",
    check_values: bool = True,
    raise_on_error: bool = True,
    overwrite_output: bool = False,
) -> ModelTransformReport:
    """Apply verified transformations to selected tensors of a local model.

    Parameters
    ----------
    source_dir
        Path to an existing HF / safetensors model directory.
    output_dir
        Destination directory. Must not exist (or be empty) unless
        ``overwrite_output=True``. Source is never modified.
    transformations
        Mapping of tensor name → transform spec. Spec may be:
          - a string operation name ("normalize", "convert_dtype", …)
          - a dict with "operation" / "op" and parameters
          - a TensorTransformSpec instance
        Only named tensors are transformed; all others are copied unchanged.
        Use exact tensor names as they appear in the source manifest.
    model_id
        Override source model identifier.
    output_model_id
        Identifier recorded for the output artifact (default: source + "-transformed").
    on_nonfinite
        Policy when a selected tensor contains NaN/Inf:
          - "reject"  (default): skip that tensor, record rejection
          - "preserve": apply transform while keeping non-finite values
          - "error": raise ValidationError immediately
    check_values
        Whether to scan for non-finite values during pre- and post-inspection.
    raise_on_error
        If True, raise on structural validation failures of the source.
    overwrite_output
        If True, allow writing into an existing non-empty output_dir
        (still never touches source).

    Returns
    -------
    ModelTransformReport
        Deterministic report containing per-tensor hashes, error metrics,
        validation results, and a ComputationalLineage conforming to the
        Shared Evidence Contract.
    """
    if on_nonfinite not in NONFINITE_POLICIES:
        raise ValidationError(
            f"on_nonfinite must be one of {sorted(NONFINITE_POLICIES)}"
        )

    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()

    if not source_dir.is_dir():
        raise SerializationError(f"Source model directory does not exist: {source_dir}")

    # Safety: refuse to write into the source tree (before any mkdir)
    if output_dir == source_dir or str(output_dir).startswith(str(source_dir) + "/"):
        raise SerializationError(
            "output_dir must be distinct from source_dir and must not be nested inside it"
        )

    if output_dir.exists():
        if any(output_dir.iterdir()) and not overwrite_output:
            raise SerializationError(
                f"Output directory is not empty: {output_dir}. "
                "Pass overwrite_output=True to allow, or choose a different path."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    specs = _parse_specs(transformations)
    if not specs:
        raise ValidationError("transformations mapping must not be empty")

    # --- 1. Inspect source -------------------------------------------------
    source_manifest = inspect_model(
        source_dir,
        model_id=model_id,
        check_values=check_values,
        raise_on_error=raise_on_error,
    )
    if source_manifest.validation_errors and raise_on_error:
        raise ValidationError(
            "Source model has validation errors: "
            + "; ".join(source_manifest.validation_errors[:5])
        )

    tensor_by_name = {t.name: t for t in source_manifest.tensors}
    unknown = [n for n in specs if n not in tensor_by_name]
    if unknown:
        msg = f"Unknown tensor name(s) in transformations: {unknown}"
        if raise_on_error:
            raise ValidationError(msg)
        # otherwise record and continue for known ones

    # --- 2. Load & transform tensor-by-tensor ------------------------------
    _require_safetensors()
    from safetensors import safe_open

    # Group tensors by source shard for efficient reading
    shard_to_tensors: Dict[str, List[str]] = {}
    for t in source_manifest.tensors:
        shard_to_tensors.setdefault(t.source_shard, []).append(t.name)

    results: List[TensorTransformResult] = []
    output_tensors_by_shard: Dict[str, Dict[str, np.ndarray]] = {}
    all_input_hashes: List[str] = []
    all_output_hashes: List[str] = []
    rejected_ops: List[str] = []
    report_warnings: List[str] = list(source_manifest.validation_warnings)
    report_errors: List[str] = list(source_manifest.validation_errors)

    for unknown_name in unknown:
        report_errors.append(f"Unknown tensor in transformations: {unknown_name}")
        rejected_ops.append(unknown_name)

    for shard_name, names in shard_to_tensors.items():
        shard_path = source_dir / shard_name
        if not shard_path.is_file():
            # already recorded by inspect_model
            continue

        with safe_open(str(shard_path), framework="np") as f:
            out_shard: Dict[str, np.ndarray] = {}
            for name in names:
                arr = f.get_tensor(name)
                meta_t = tensor_by_name[name]
                in_desc = analyze(arr)
                all_input_hashes.append(in_desc.content_hash)

                if name not in specs:
                    # passthrough — preserve exact bytes/identity
                    out_shard[name] = arr
                    results.append(
                        TensorTransformResult(
                            name=name,
                            applied=False,
                            operation="passthrough",
                            parameters={},
                            input_shape=tuple(meta_t.shape),
                            input_dtype=meta_t.dtype,
                            input_hash=in_desc.content_hash,
                            output_shape=tuple(meta_t.shape),
                            output_dtype=meta_t.dtype,
                            output_hash=in_desc.content_hash,
                        )
                    )
                    all_output_hashes.append(in_desc.content_hash)
                    continue

                spec = specs[name]
                out_arr, meta, warns, rejected, reason = _apply_one(
                    arr, spec, on_nonfinite=on_nonfinite
                )
                report_warnings.extend(f"{name}: {w}" for w in warns)

                out_desc = build_descriptor(out_arr)
                err: Dict[str, Any] = {}
                if not rejected and out_arr.shape == arr.shape:
                    try:
                        # Compare in float64 when possible
                        a = arr.astype(np.float64, copy=False)
                        b = out_arr.astype(np.float64, copy=False)
                        if a.shape == b.shape:
                            err = compare(a, b)
                    except Exception:
                        err = {}

                if rejected:
                    rejected_ops.append(name)
                    # keep original on rejection
                    out_shard[name] = arr
                    results.append(
                        TensorTransformResult(
                            name=name,
                            applied=False,
                            operation=spec.operation,
                            parameters=dict(spec.parameters),
                            input_shape=tuple(arr.shape),
                            input_dtype=str(arr.dtype),
                            output_shape=tuple(arr.shape),
                            output_dtype=str(arr.dtype),
                            input_hash=in_desc.content_hash,
                            output_hash=in_desc.content_hash,
                            error_metrics={},
                            warnings=warns,
                            rejected=True,
                            reason=reason,
                        )
                    )
                    all_output_hashes.append(in_desc.content_hash)
                else:
                    out_shard[name] = out_arr
                    results.append(
                        TensorTransformResult(
                            name=name,
                            applied=True,
                            operation=spec.operation,
                            parameters=dict(spec.parameters),
                            input_shape=tuple(arr.shape),
                            input_dtype=str(arr.dtype),
                            input_hash=in_desc.content_hash,
                            output_shape=tuple(out_arr.shape),
                            output_dtype=str(out_arr.dtype),
                            output_hash=out_desc.content_hash,
                            error_metrics=err,
                            warnings=warns,
                            rejected=False,
                        )
                    )
                    all_output_hashes.append(out_desc.content_hash)

                del arr
                del out_arr

            output_tensors_by_shard[shard_name] = out_shard

    results.sort(key=lambda r: r.name)

    # --- 3. Write output artifacts -----------------------------------------
    weight_map: Dict[str, str] = {}
    for shard_name, tensors in output_tensors_by_shard.items():
        out_shard_path = output_dir / shard_name
        _write_safetensors(out_shard_path, tensors)
        for tname in tensors:
            weight_map[tname] = shard_name

    # index.json
    index = {
        "metadata": {"total_size": sum(
            (output_dir / s).stat().st_size for s in output_tensors_by_shard
        )},
        "weight_map": weight_map,
    }
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True), encoding="utf-8"
    )
    _copy_config(source_dir, output_dir)

    # Write the transformation report itself (deterministic core)
    out_model_id = output_model_id or (source_manifest.model_id + "-transformed")

    # --- 4. Re-inspect output ----------------------------------------------
    output_manifest = inspect_model(
        output_dir,
        model_id=out_model_id,
        check_values=check_values,
        raise_on_error=False,
    )
    report_errors.extend(output_manifest.validation_errors)
    report_warnings.extend(output_manifest.validation_warnings)

    # --- 5. Build lineage (Shared Evidence Contract) -----------------------
    # Aggregate input/output hashes for the model-level boundary crossing.
    # Use sorted unique hashes for stability.
    unique_in = sorted(set(all_input_hashes))
    unique_out = sorted(set(all_output_hashes))
    # Primary output identity is the output manifest hash
    lineage = build_lineage(
        operation="model_transform",
        input_hashes=unique_in or [source_manifest.manifest_hash],
        output_hash=output_manifest.manifest_hash,
        parameters={
            "selected_tensors": sorted(specs.keys()),
            "operations": {
                n: {"op": s.operation, "params": s.parameters}
                for n, s in sorted(specs.items())
            },
            "on_nonfinite": on_nonfinite,
        },
        kind=ComputationKind.DETERMINISTIC,
        input_identities=[
            TensorIdentity(
                content_hash=source_manifest.manifest_hash,
                label="source_manifest",
            )
        ],
        output_identity=TensorIdentity(
            content_hash=output_manifest.manifest_hash,
            label="output_manifest",
        ),
        notes="Verified model-artifact transformation",
    )

    report = ModelTransformReport(
        source_model_id=source_manifest.model_id,
        output_model_id=out_model_id,
        source_manifest_hash=source_manifest.manifest_hash,
        output_manifest_hash=output_manifest.manifest_hash,
        tensors=results,
        lineage=lineage,
        validation_errors=report_errors,
        validation_warnings=report_warnings,
        rejected_operations=rejected_ops,
        source_path=str(source_dir),
        output_path=str(output_dir),
    )

    # Persist report next to the model (deterministic JSON)
    report_path = output_dir / "tensorgate_transform_report.json"
    # Strip volatile path fields for the on-disk identity document
    report_dict = report.to_dict()
    report_dict.pop("source_path", None)
    report_dict.pop("output_path", None)
    report_path.write_text(
        json.dumps(report_dict, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )

    return report


def transform_model_report_path(output_dir: Union[str, Path]) -> Path:
    """Return the conventional path of the transformation report inside output_dir."""
    return Path(output_dir) / "tensorgate_transform_report.json"
