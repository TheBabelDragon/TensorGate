"""Thin MetaField adapter.

Treats MetaField tensors as numerical state only.
Does NOT import MetaField internals or absorb physics semantics.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..inspect import analyze
from ..compatibility import adapt, check_compatibility
from ..tensor import as_numpy, ArrayLike, TensorSpec


def describe_state(tensor: ArrayLike) -> Dict[str, Any]:
    """Produce a TensorGate descriptor for a MetaField numerical state array."""
    return analyze(tensor).to_dict()


def prepare_for_operator(
    tensor: ArrayLike,
    target: Optional[TensorSpec] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Validate / adapt a tensor before handing it to an operator ABI.

    The frozen Wilson–Dirac (or any other) ABI remains authoritative.
    TensorGate only ensures numerical compatibility at the boundary.
    """
    if target is None:
        arr = as_numpy(tensor)
        return arr, {"descriptor": analyze(arr).to_dict(), "adapted": False}
    adapted, report = adapt(tensor, target)
    return adapted, report.to_dict()


def accept_operator_output(
    tensor: ArrayLike,
    expected: Optional[TensorSpec] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Validate an operator output tensor."""
    arr = as_numpy(tensor)
    desc = analyze(arr)
    report = {"descriptor": desc.to_dict()}
    if expected is not None:
        report["compatibility"] = check_compatibility(arr, expected).to_dict()
    return arr, report
