"""Layout / memory-order descriptors."""

from __future__ import annotations

from enum import Enum


class Layout(str, Enum):
    """Known memory layouts. Unknown layouts are represented as None."""

    C = "C"          # row-major / C-contiguous
    F = "F"          # column-major / Fortran-contiguous
    UNKNOWN = "unknown"

    @classmethod
    def from_numpy(cls, arr) -> "Layout":
        import numpy as np
        if not isinstance(arr, np.ndarray):
            return cls.UNKNOWN
        if arr.flags["C_CONTIGUOUS"]:
            return cls.C
        if arr.flags["F_CONTIGUOUS"]:
            return cls.F
        return cls.UNKNOWN
