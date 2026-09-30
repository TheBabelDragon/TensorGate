"""TensorSpec, BackendTensorSpec, and array conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from .layout import Layout


ArrayLike = Union[np.ndarray, Any]  # Any covers optional torch.Tensor


def as_numpy(obj: ArrayLike, *, copy: bool = False) -> np.ndarray:
    """Convert a tensor-like object to a NumPy array without requiring torch.

    Accepts np.ndarray and, if torch is installed, torch.Tensor.
    Never silently alters values; only converts representation.
    """
    if isinstance(obj, np.ndarray):
        return np.array(obj, copy=copy) if copy else obj

    # Optional torch support
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            t = obj.detach().cpu()
            if t.dtype == torch.bfloat16:
                t = t.to(torch.float32)
            arr = t.numpy()
            return np.array(arr, copy=True) if copy else arr
    except ImportError:
        pass

    # Fallback: try array interface
    try:
        return np.asarray(obj)
    except Exception as e:
        raise TypeError(
            f"Cannot convert object of type {type(obj).__name__} to numpy array: {e}"
        ) from e


@dataclass(frozen=True)
class TensorSpec:
    """Declarative requirements for a tensor (source or target).

    All fields are optional constraints. Missing fields mean \"no constraint\".
    """

    dtype: Optional[str] = None          # e.g. \"float32\", \"int8\"
    shape: Optional[Tuple[int, ...]] = None
    rank: Optional[int] = None
    layout: Optional[Layout] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allow_nan: bool = False
    allow_inf: bool = False
    signed: Optional[bool] = None
    scale: Optional[float] = None
    zero_point: Optional[int] = None
    quantization: Optional[str] = None   # \"none\" | \"symmetric\" | \"asymmetric\" | \"affine\"
    alignment: Optional[int] = None      # byte alignment if required
    precision_bits: Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.layout is not None:
            d[\"layout\"] = self.layout.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> \"TensorSpec\":
        d = dict(d)
        if \"layout\" in d and d[\"layout\"] is not None:
            d[\"layout\"] = Layout(d[\"layout\"])
        if \"shape\" in d and d[\"shape\"] is not None:
            d[\"shape\"] = tuple(d[\"shape\"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class BackendTensorSpec:
    """Backend capability declaration (CPU / CUDA / TPU / FPGA / ASIC / custom).

    Used for pre-execution compatibility checks. No fake backends are
    implemented; this is a neutral representation only.
    """

    name: str
    supported_dtypes: Tuple[str, ...] = ()
    supported_layouts: Tuple[Layout, ...] = ()
    supported_ranks: Optional[Tuple[int, ...]] = None  # None = any
    max_rank: Optional[int] = None
    max_elements: Optional[int] = None
    alignment: Optional[int] = None
    quantization_formats: Tuple[str, ...] = ()
    notes: str = \"\"

    def accepts(self, spec: TensorSpec) -> bool:
        if spec.dtype and self.supported_dtypes and spec.dtype not in self.supported_dtypes:
            return False
        if spec.layout and self.supported_layouts and spec.layout not in self.supported_layouts:
            return False
        if spec.rank is not None:
            if self.supported_ranks is not None and spec.rank not in self.supported_ranks:
                return False
            if self.max_rank is not None and spec.rank > self.max_rank:
                return False
        if spec.quantization and self.quantization_formats:
            if spec.quantization not in self.quantization_formats and spec.quantization != \"none\":
                return False
        return True
