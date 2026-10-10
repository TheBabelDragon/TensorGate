"""Computational lineage for the Shared Evidence Contract.

Records input identities/versions, transformation or model version, and
output identity. Separates reproducible deterministic calculations from
nondeterministic model inference.

Does not import MetaField or CANtheon. Evidence links are plain dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from . import __software_id__
from .errors import ValidationError, CompatibilityError
from .provenance import ProvenanceRecord


class ComputationKind(str, Enum):
    """Deterministic transforms vs nondeterministic inference."""

    DETERMINISTIC = "DETERMINISTIC"
    NONDETERMINISTIC = "NONDETERMINISTIC"
    UNKNOWN = "UNKNOWN"


@dataclass
class TensorIdentity:
    """Stable identity for a tensor at a boundary."""

    content_hash: str
    shape: tuple = ()
    dtype: str = ""
    schema_id: str = ""
    schema_version: str = ""
    label: str = ""

    def to_dict(self) -> dict:
        return {
            "content_hash": self.content_hash,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["TensorIdentity"]:
        if not d:
            return None
        shape = d.get("shape") or ()
        if isinstance(shape, list):
            shape = tuple(shape)
        return cls(
            content_hash=str(d.get("content_hash") or ""),
            shape=tuple(shape),
            dtype=str(d.get("dtype") or ""),
            schema_id=str(d.get("schema_id") or ""),
            schema_version=str(d.get("schema_version") or ""),
            label=str(d.get("label") or ""),
        )


@dataclass
class ComputationalLineage:
    """Full computational lineage for one TensorGate boundary crossing."""

    kind: ComputationKind = ComputationKind.DETERMINISTIC
    transformation_id: str = ""
    transformation_version: str = __software_id__
    model_id: str = ""
    model_version: str = ""
    inputs: List[TensorIdentity] = field(default_factory=list)
    output: Optional[TensorIdentity] = None
    provenance: ProvenanceRecord = field(default_factory=ProvenanceRecord)
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    schema_id: str = "tensorgate.lineage"
    schema_version: str = "0.1.0"
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: Dict[str, Any] = {
            "kind": self.kind.value if isinstance(self.kind, ComputationKind) else str(self.kind),
            "transformation_id": self.transformation_id,
            "transformation_version": self.transformation_version,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "inputs": [i.to_dict() for i in self.inputs],
            "output": self.output.to_dict() if self.output else None,
            "provenance": self.provenance.to_list(),
            "evidence_refs": list(self.evidence_refs),
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
        }
        if self.extras:
            d["extras"] = dict(self.extras)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ComputationalLineage":
        kind_raw = d.get("kind", "DETERMINISTIC")
        try:
            kind = ComputationKind(str(kind_raw).upper())
        except ValueError:
            kind = ComputationKind.UNKNOWN
        inputs = [
            TensorIdentity.from_dict(i)
            for i in (d.get("inputs") or [])
            if i
        ]
        inputs = [i for i in inputs if i is not None]
        output = TensorIdentity.from_dict(d.get("output"))
        prov = ProvenanceRecord.from_list(d.get("provenance") or [])
        return cls(
            kind=kind,
            transformation_id=str(d.get("transformation_id") or ""),
            transformation_version=str(d.get("transformation_version") or __software_id__),
            model_id=str(d.get("model_id") or ""),
            model_version=str(d.get("model_version") or ""),
            inputs=inputs,
            output=output,
            provenance=prov,
            evidence_refs=list(d.get("evidence_refs") or []),
            schema_id=str(d.get("schema_id") or "tensorgate.lineage"),
            schema_version=str(d.get("schema_version") or "0.1.0"),
            extras=dict(d.get("extras") or {}),
        )


def identity_from_descriptor(desc: Any, *, label: str = "") -> TensorIdentity:
    """Build TensorIdentity from a TensorDescriptor (duck-typed)."""
    shape = getattr(desc, "shape", ()) or ()
    if hasattr(shape, "__iter__") and not isinstance(shape, (str, bytes)):
        shape = tuple(int(x) for x in shape)
    else:
        shape = ()
    return TensorIdentity(
        content_hash=str(getattr(desc, "content_hash", "") or ""),
        shape=shape,
        dtype=str(getattr(desc, "dtype", "") or ""),
        label=label,
    )


def build_lineage(
    *,
    operation: str,
    input_hashes: List[str],
    output_hash: str,
    parameters: Optional[Dict[str, Any]] = None,
    kind: ComputationKind = ComputationKind.DETERMINISTIC,
    model_id: str = "",
    model_version: str = "",
    input_identities: Optional[List[TensorIdentity]] = None,
    output_identity: Optional[TensorIdentity] = None,
    evidence_refs: Optional[List[Dict[str, Any]]] = None,
    notes: str = "",
    prior: Optional[ProvenanceRecord] = None,
) -> ComputationalLineage:
    """Construct a ComputationalLineage with one transformation step."""
    if kind == ComputationKind.DETERMINISTIC and model_id:
        raise ValidationError(
            "DETERMINISTIC lineage must not set model_id "
            "(use NONDETERMINISTIC for model inference)"
        )
    if not input_hashes:
        raise ValidationError("at least one input_hash is required")
    if not output_hash:
        raise ValidationError("output_hash is required")

    from .provenance import make_record

    rec = make_record(
        operation=operation,
        parameters=parameters or {},
        input_hash=input_hashes[0],
        output_hash=output_hash,
        notes=notes,
    )
    prov = ProvenanceRecord()
    if prior is not None:
        prov.extend_from(prior)
    prov.append(rec)

    inputs = list(input_identities or [])
    if not inputs:
        inputs = [
            TensorIdentity(content_hash=h, label=f"input_{i}")
            for i, h in enumerate(input_hashes)
        ]
    output = output_identity or TensorIdentity(content_hash=output_hash, label="output")

    return ComputationalLineage(
        kind=kind,
        transformation_id=operation,
        transformation_version=__software_id__,
        model_id=model_id,
        model_version=model_version,
        inputs=inputs,
        output=output,
        provenance=prov,
        evidence_refs=list(evidence_refs or []),
    )


def check_lineage_schema(
    lineage: ComputationalLineage,
    *,
    expected_schema_id: str = "tensorgate.lineage",
    expected_major: int = 0,
) -> None:
    """Fail-closed schema check for lineage envelopes."""
    if lineage.schema_id and lineage.schema_id != expected_schema_id:
        raise CompatibilityError(
            f"incompatible lineage schema_id: {lineage.schema_id!r} "
            f"(expected {expected_schema_id!r})"
        )
    try:
        major = int(str(lineage.schema_version).split(".")[0])
    except ValueError:
        raise CompatibilityError(
            f"unparseable lineage schema_version: {lineage.schema_version!r}"
        )
    if major != expected_major:
        raise CompatibilityError(
            f"incompatible lineage major version: {lineage.schema_version} "
            f"(supported major={expected_major})"
        )


def attach_evidence_ref(
    lineage: ComputationalLineage,
    *,
    event_id: str = "",
    source_id: str = "",
    schema_id: str = "",
    schema_version: str = "",
    extras: Optional[Dict[str, Any]] = None,
) -> ComputationalLineage:
    """Append an upstream evidence reference (CANtheon / MetaField event)."""
    ref: Dict[str, Any] = {}
    if event_id:
        ref["event_id"] = event_id
    if source_id:
        ref["source_id"] = source_id
    if schema_id:
        ref["schema_id"] = schema_id
    if schema_version:
        ref["schema_version"] = schema_version
    if extras:
        ref["extras"] = dict(extras)
    if ref:
        lineage.evidence_refs.append(ref)
    return lineage
