"""Boundary helper for metafield-operator-abi.

TensorGate does NOT redefine Wilson–Dirac mathematics.
It only describes the numerical tensor boundary around an operator invocation:

    input tensor
      → TensorGate compatibility validation
      → Operator ABI (frozen math)
      → backend / operator
      → output tensor
      → TensorGate validation

Complex arrays are inspected via their real/imag views when needed;
the mathematical operator contract remains authoritative.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..compatibility import adapt, check_compatibility, compare
from ..inspect import analyze
from ..tensor import ArrayLike, TensorSpec, as_numpy


def _to_inspectable(tensor: ArrayLike) -> np.ndarray:
    """Convert complex tensors to a real view for numerical inspection."""
    arr = as_numpy(tensor)
    if np.iscomplexobj(arr):
        return np.stack([arr.real, arr.imag], axis=-1)
    return arr


def validate_inputs(
    *tensors: ArrayLike,
    specs: Optional[Sequence[Optional[TensorSpec]]] = None,
) -> Dict[str, Any]:
    """Validate one or more input tensors against optional TensorSpecs.

    Does not mutate inputs. Reports structured compatibility per tensor.
    """
    results: List[Dict[str, Any]] = []
    for i, t in enumerate(tensors):
        view = _to_inspectable(t)
        desc = analyze(view)
        entry: Dict[str, Any] = {
            "index": i,
            "descriptor": desc.to_dict(),
            "is_complex": bool(np.iscomplexobj(as_numpy(t))),
        }
        if specs is not None and i < len(specs) and specs[i] is not None:
            entry["compatibility"] = check_compatibility(view, specs[i]).to_dict()
        results.append(entry)
    return {"inputs": results}


def validate_output(
    tensor: ArrayLike,
    expected: Optional[TensorSpec] = None,
) -> Dict[str, Any]:
    """Validate an operator output tensor."""
    view = _to_inspectable(tensor)
    desc = analyze(view)
    out: Dict[str, Any] = {
        "descriptor": desc.to_dict(),
        "is_complex": bool(np.iscomplexobj(as_numpy(tensor))),
    }
    if expected is not None:
        out["compatibility"] = check_compatibility(view, expected).to_dict()
    return out


def prepare_input(
    tensor: ArrayLike,
    target: Optional[TensorSpec] = None,
    *,
    raise_on_incompatible: bool = False,
) -> tuple:
    """Optionally adapt an input tensor to a numerical target spec.

    Returns (array, report_dict). Adaptation only occurs when target is given
    and permits it. Never silently mutates when target is None.
    """
    arr = as_numpy(tensor)
    if target is None:
        return arr, {"descriptor": analyze(_to_inspectable(arr)).to_dict(), "adapted": False}
    adapted, report = adapt(arr, target, raise_on_incompatible=raise_on_incompatible)
    return adapted, report.to_dict()


def compare_tensors(
    original: ArrayLike,
    reconstructed: ArrayLike,
) -> Dict[str, Any]:
    """Compare two tensors with explicit error metrics (real view if complex)."""
    a = _to_inspectable(original)
    b = _to_inspectable(reconstructed)
    return compare(a, b)
