"""
TensorGate — canonical numerical compatibility boundary for tensors
and neural-network weights across the BabelDragon compute stack.

TensorGate is where tensors go when they need to cross a boundary.

It is NOT a training framework, inference engine, field OS, transport
protocol, or operator implementation. It owns numerical representation
and compatibility only.
"""

from __future__ import annotations

__version__ = "0.1.0"
__software_id__ = f"tensorgate/{__version__}"

from .errors import (
    TensorGateError,
    CompatibilityError,
    ValidationError,
    SerializationError,
)
from .descriptor import TensorDescriptor
from .tensor import TensorSpec, BackendTensorSpec, as_numpy
from .inspect import analyze, describe
from .normalize import normalize
from .transform import transform, convert_dtype, clip
from .quantize import quantize, dequantize
from .compatibility import check_compatibility, adapt, compare, CompatibilityReport
from .serialization import save_tensor, load_tensor
from .provenance import ProvenanceRecord, TransformationRecord
from .layout import Layout

__all__ = [
    "__version__",
    "__software_id__",
    "TensorGateError",
    "CompatibilityError",
    "ValidationError",
    "SerializationError",
    "TensorDescriptor",
    "TensorSpec",
    "BackendTensorSpec",
    "as_numpy",
    "analyze",
    "describe",
    "normalize",
    "transform",
    "convert_dtype",
    "clip",
    "quantize",
    "dequantize",
    "check_compatibility",
    "adapt",
    "compare",
    "CompatibilityReport",
    "save_tensor",
    "load_tensor",
    "ProvenanceRecord",
    "TransformationRecord",
    "Layout",
]
