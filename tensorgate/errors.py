"""TensorGate exception hierarchy."""

from __future__ import annotations


class TensorGateError(Exception):
    """Base class for all TensorGate errors."""


class CompatibilityError(TensorGateError):
    """Raised when a tensor cannot be adapted to a target specification."""


class ValidationError(TensorGateError):
    """Raised when a tensor fails validation against a descriptor or spec."""


class SerializationError(TensorGateError):
    """Raised on load/save failures or corrupt artifacts."""
