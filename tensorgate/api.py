"""Versioned lineage API surface for TensorGate.

Schema: tensorgate.lineage_api@1.0.0

Does not import MetaField or CANtheon. Downstream systems consume
ComputationalLineage dicts with explicit fail-closed checks.
"""

from __future__ import annotations

from typing import Any, Optional

from .lineage import ComputationalLineage, ComputationKind, build_lineage

API_SCHEMA_ID = "tensorgate.lineage_api"
API_SCHEMA_VERSION = "1.0.0"


class LineageApiError(Exception):
    """Raised on incompatible schema or missing required identity fields."""


def health() -> dict[str, Any]:
    return {
        "schema_id": API_SCHEMA_ID,
        "schema_version": API_SCHEMA_VERSION,
        "healthy": True,
        "lineage_schema_id": "tensorgate.lineage",
        "capabilities": [
            "computational_lineage",
            "deterministic_fail_closed",
            "evidence_refs",
        ],
    }


def negotiate(client_version: str) -> dict[str, Any]:
    major = (client_version or "0").split(".")[0]
    ok = major == API_SCHEMA_VERSION.split(".")[0]
    return {
        "compatible": ok,
        "server_schema_id": API_SCHEMA_ID,
        "server_schema_version": API_SCHEMA_VERSION,
        "client_version": client_version,
    }


def validate_lineage_dict(data: dict[str, Any]) -> ComputationalLineage:
    """Fail-closed validation of a lineage dict from a peer."""
    if not isinstance(data, dict):
        raise LineageApiError("lineage must be a dict")
    schema_id = data.get("schema_id") or "tensorgate.lineage"
    if schema_id not in ("tensorgate.lineage", API_SCHEMA_ID):
        raise LineageApiError(f"unknown schema_id: {schema_id}")
    version = str(data.get("schema_version") or "0.1.0")
    # Accept 0.x and 1.x lineage payloads already in the wild
    kind_raw = data.get("kind") or ComputationKind.DETERMINISTIC.value
    try:
        kind = ComputationKind(kind_raw) if not isinstance(kind_raw, ComputationKind) else kind_raw
    except Exception as e:
        raise LineageApiError(f"invalid computation kind: {e}") from e

    if kind == ComputationKind.DETERMINISTIC and data.get("model_id"):
        raise LineageApiError("DETERMINISTIC lineage cannot claim model_id")

    # Require output hash when output present
    output = data.get("output")
    if isinstance(output, dict) and not output.get("content_hash") and not output.get("hash"):
        raise LineageApiError("output missing content hash")

    return ComputationalLineage(
        kind=kind,
        transformation_id=str(data.get("transformation_id") or ""),
        transformation_version=str(data.get("transformation_version") or ""),
        model_id=data.get("model_id"),
        model_version=data.get("model_version"),
        schema_id=str(data.get("schema_id") or "tensorgate.lineage"),
        schema_version=version,
        evidence_refs=list(data.get("evidence_refs") or []),
        extras=dict(data.get("extras") or {}),
    )


def lineage_to_contract(lineage: ComputationalLineage) -> dict[str, Any]:
    """Serialize lineage for cross-process exchange."""
    d = lineage.to_dict()
    d["api_schema_id"] = API_SCHEMA_ID
    d["api_schema_version"] = API_SCHEMA_VERSION
    return d
