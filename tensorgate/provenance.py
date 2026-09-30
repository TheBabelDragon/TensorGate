"""Provenance and transformation records — inspectable numerical history."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from . import __software_id__


@dataclass
class TransformationRecord:
    """Single explicit transformation step.

    Every numerical change MUST produce one of these. No silent mutations.
    """

    operation: str
    parameters: Dict[str, Any]
    input_hash: str
    output_hash: str
    software_id: str = __software_id__
    # Informational only — never used as identity
    timestamp_unix: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TransformationRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ProvenanceRecord:
    """Chain of transformation records from original to current state."""

    chain: List[TransformationRecord] = field(default_factory=list)

    def append(self, record: TransformationRecord) -> "ProvenanceRecord":
        self.chain.append(record)
        return self

    def to_list(self) -> List[dict]:
        return [r.to_dict() for r in self.chain]

    @classmethod
    def from_list(cls, items: List[dict]) -> "ProvenanceRecord":
        return cls(chain=[TransformationRecord.from_dict(i) for i in items])

    def extend_from(self, other: "ProvenanceRecord") -> "ProvenanceRecord":
        self.chain.extend(other.chain)
        return self


def make_record(
    operation: str,
    parameters: Dict[str, Any],
    input_hash: str,
    output_hash: str,
    notes: str = "",
) -> TransformationRecord:
    return TransformationRecord(
        operation=operation,
        parameters=parameters,
        input_hash=input_hash,
        output_hash=output_hash,
        software_id=__software_id__,
        timestamp_unix=time.time(),
        notes=notes,
    )
