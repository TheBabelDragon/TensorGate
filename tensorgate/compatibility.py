"""Compatibility checking and adaptation with full transformation records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .descriptor import build_descriptor
from .errors import CompatibilityError
from .inspect import analyze
from .normalize import normalize
from .provenance import ProvenanceRecord, make_record
from .quantize import quantize, dequantize
from .tensor import as_numpy, ArrayLike, TensorSpec
from .transform import convert_dtype, clip


@dataclass
class CompatibilityReport:
    compatible: bool
    issues: List[str] = field(default_factory=list)
    transformations: List[str] = field(default_factory=list)
    provenance: ProvenanceRecord = field(default_factory=ProvenanceRecord)
    error_metrics: Dict[str, Any] = field(default_factory=dict)
    adapted_descriptor: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        return {
            "compatible": self.compatible,
            "issues": self.issues,
            "transformations": self.transformations,
            "provenance": self.provenance.to_list(),
            "error_metrics": self.error_metrics,
            "adapted_descriptor": self.adapted_descriptor,
        }


def check_compatibility(
    source: ArrayLike | TensorSpec,
    target: TensorSpec,
) -> CompatibilityReport:
    """Evaluate whether source satisfies target constraints. Does not mutate."""
    issues: List[str] = []
    if isinstance(source, TensorSpec):
        src_spec = source
        desc = None
    else:
        arr = as_numpy(source)
        desc = analyze(arr)
        src_spec = TensorSpec(
            dtype=desc.dtype,
            shape=desc.shape,
            rank=desc.rank,
            layout=None,
            min_value=desc.min,
            max_value=desc.max,
            allow_nan=desc.nan_count > 0,
            allow_inf=(desc.pos_inf_count + desc.neg_inf_count) > 0,
        )

    if target.dtype and src_spec.dtype and target.dtype != src_spec.dtype:
        issues.append(f"dtype mismatch: source={src_spec.dtype} target={target.dtype}")
    if target.shape and src_spec.shape and target.shape != src_spec.shape:
        issues.append(f"shape mismatch: source={src_spec.shape} target={target.shape}")
    if target.rank is not None and src_spec.rank is not None and target.rank != src_spec.rank:
        issues.append(f"rank mismatch: source={src_spec.rank} target={target.rank}")
    if target.min_value is not None and src_spec.min_value is not None:
        if src_spec.min_value < target.min_value - 1e-12:
            issues.append(f"min below target: source_min={src_spec.min_value} target_min={target.min_value}")
    if target.max_value is not None and src_spec.max_value is not None:
        if src_spec.max_value > target.max_value + 1e-12:
            issues.append(f"max above target: source_max={src_spec.max_value} target_max={target.max_value}")
    if not target.allow_nan and desc is not None and desc.nan_count > 0:
        issues.append(f"NaN present ({desc.nan_count}) but target.allow_nan=False")
    if not target.allow_inf and desc is not None and (desc.pos_inf_count + desc.neg_inf_count) > 0:
        issues.append("Infinities present but target.allow_inf=False")

    return CompatibilityReport(compatible=len(issues) == 0, issues=issues)


def compare(original: ArrayLike, reconstructed: ArrayLike) -> Dict[str, Any]:
    """Deterministic error analysis between two tensors of equal shape."""
    a = as_numpy(original).astype(np.float64)
    b = as_numpy(reconstructed).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for compare: {a.shape} vs {b.shape}")
    diff = a - b
    abs_diff = np.abs(diff)
    max_abs = float(np.max(abs_diff)) if abs_diff.size else 0.0
    mean_abs = float(np.mean(abs_diff)) if abs_diff.size else 0.0
    rms = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
    eps = 1e-12
    mask = np.abs(a) > eps
    rel = np.abs(diff[mask] / a[mask]) if mask.any() else np.array([])
    max_rel = float(np.max(rel)) if rel.size else 0.0
    mean_rel = float(np.mean(rel)) if rel.size else 0.0
    signal_power = float(np.mean(a * a)) if a.size else 0.0
    noise_power = float(np.mean(diff * diff)) if diff.size else 0.0
    sqnr_db = 10.0 * np.log10(signal_power / max(noise_power, eps)) if signal_power > 0 else float("inf")
    changed = int(np.sum(abs_diff > 0))
    return {
        "max_absolute_error": max_abs,
        "mean_absolute_error": mean_abs,
        "rms_error": rms,
        "max_relative_error": max_rel,
        "mean_relative_error": mean_rel,
        "sqnr_db": float(sqnr_db),
        "changed_element_count": changed,
        "element_count": int(a.size),
    }


def adapt(
    tensor: ArrayLike,
    target: TensorSpec,
    *,
    raise_on_incompatible: bool = False,
) -> Tuple[np.ndarray, CompatibilityReport]:
    """Adapt tensor to target spec. Returns (adapted_tensor, report).

    All numerical changes are recorded in the provenance chain.
    Never silently alters meaning.
    """
    arr = as_numpy(tensor, copy=True)
    original = arr.copy()
    desc = analyze(arr)
    prov = ProvenanceRecord()
    transforms: List[str] = []
    issues: List[str] = []

    if target.shape is not None and tuple(arr.shape) != target.shape:
        msg = f"shape mismatch: {arr.shape} vs {target.shape}"
        issues.append(msg)
        if raise_on_incompatible:
            raise CompatibilityError(msg)
    if target.rank is not None and arr.ndim != target.rank:
        msg = f"rank mismatch: {arr.ndim} vs {target.rank}"
        issues.append(msg)
        if raise_on_incompatible:
            raise CompatibilityError(msg)

    if not target.allow_nan and np.isnan(arr).any():
        issues.append("NaN present; target forbids NaN (no automatic fill)")
    if not target.allow_inf and np.isinf(arr).any():
        issues.append("Inf present; target forbids Inf (no automatic replace)")

    need_range = False
    tmin = target.min_value
    tmax = target.max_value
    if tmin is not None or tmax is not None:
        cur_min = float(np.nanmin(arr)) if arr.size else 0.0
        cur_max = float(np.nanmax(arr)) if arr.size else 0.0
        if (tmin is not None and cur_min < tmin - 1e-12) or (tmax is not None and cur_max > tmax + 1e-12):
            need_range = True

    if need_range and tmin is not None and tmax is not None:
        out, meta = normalize(arr, method="range", target_min=tmin, target_max=tmax)
        arr = out
        transforms.append(f"normalize/range → [{tmin}, {tmax}]")
        for p in meta.get("provenance", []):
            prov.append(make_record(
                operation=p["operation"],
                parameters=p["parameters"],
                input_hash=p["input_hash"],
                output_hash=p["output_hash"],
            ))
    elif need_range:
        out, meta = clip(arr, min_value=tmin, max_value=tmax)
        arr = out
        transforms.append(f"clip[{tmin}, {tmax}]")
        for p in meta.get("provenance", []):
            prov.append(make_record(
                operation=p["operation"],
                parameters=p["parameters"],
                input_hash=p["input_hash"],
                output_hash=p["output_hash"],
            ))

    if target.quantization and target.quantization not in ("none", None):
        bits = target.precision_bits or 8
        signed = target.signed if target.signed is not None else True
        out, meta = quantize(arr, bits=bits, scheme=target.quantization, signed=signed)
        arr = out
        transforms.append(f"quantize/{target.quantization}/{bits}bit")
        for p in meta.get("provenance", []):
            prov.append(make_record(
                operation=p["operation"],
                parameters=p["parameters"],
                input_hash=p["input_hash"],
                output_hash=p["output_hash"],
            ))

    if target.dtype and str(arr.dtype) != target.dtype:
        out, meta = convert_dtype(arr, target.dtype, casting="unsafe")
        arr = out
        transforms.append(f"convert_dtype → {target.dtype}")
        for p in meta.get("provenance", []):
            prov.append(make_record(
                operation=p["operation"],
                parameters=p["parameters"],
                input_hash=p["input_hash"],
                output_hash=p["output_hash"],
            ))

    final_desc = analyze(arr)
    try:
        recon = arr.astype(np.float64)
        err = compare(original.astype(np.float64), recon)
    except Exception:
        err = {}

    report = CompatibilityReport(
        compatible=len(issues) == 0,
        issues=issues,
        transformations=transforms,
        provenance=prov,
        error_metrics=err,
        adapted_descriptor=final_desc.to_dict(),
    )
    if raise_on_incompatible and not report.compatible:
        raise CompatibilityError("; ".join(issues))
    return arr, report
