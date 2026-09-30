"""Boundary helper for metafield-operator-abi.

TensorGate does NOT redefine Wilson–Dirac mathematics.
It only describes the numerical tensor boundary around an operator invocation:

    input tensor → TensorGate validation → Operator ABI → backend → output → TensorGate validation
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..inspect import analyze
from ..compatibility import adapt, check_compatibility
from ..tensor import as_numpy, ArrayLike, TensorSpec


def validate_inputs(
    *tensors: ArrayLike,
    specs: Optional[list] = None,
) -> Dict[str, Any]:
    """Validate one or more input tensors against optional TensorSpecs."""
    results = []
    for i, t in enumerate(tensors):
        desc = analyze(t)
        entry = {"index": i, "descriptor": desc.to_dict()}
        if specs and i < len(specs) and specs[i] is not None:
            entry["compatibility"] = check_compatibility(t, specs[i]).to_dict()
        results.append(entry)
    return {"inputs": results}


def validate_output(
    tensor: ArrayLike,
    expected: Optional[TensorSpec] = None,
) -> Dict[str, Any]:
    desc = analyze(tensor)
    out = {"descriptor": desc.to_dict()}
    if expected is not None:
        out["compatibility"] = check_compatibility(tensor, expected).to_dict()
    return out
