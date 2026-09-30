"""Inspection and description entry points."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .descriptor import TensorDescriptor, build_descriptor
from .tensor import as_numpy, ArrayLike
from . import __software_id__


def analyze(
    tensor: ArrayLike,
    *,
    scale: Optional[float] = None,
    zero_point: Optional[int] = None,
    quantization: Optional[str] = None,
) -> TensorDescriptor:
    """Inspect a tensor and return a complete TensorDescriptor.

    Never mutates the input. Pure observation.
    """
    arr = as_numpy(tensor)
    return build_descriptor(
        arr,
        scale=scale,
        zero_point=zero_point,
        quantization=quantization,
        software_id=__software_id__,
    )


def describe(tensor: ArrayLike, **kwargs) -> str:
    """Human-readable summary of a tensor."""
    return analyze(tensor, **kwargs).summary()
