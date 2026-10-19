"""Versioned lineage API surface for TensorGate.

Schema: tensorgate.lineage_api@1.0.0

Does not import MetaField or CANtheon. Downstream systems consume
ComputationalLineage dicts with explicit fail-closed checks.
"""

from __future__ import annotations

from typing import Any

from .lineage import ComputationalLineage, ComputationKind

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

    kind_raw = str(data.get("kind") or "DETERMINISTIC").upper()
    try:
        kind = ComputationKind(kind_raw)
    except Exception as e:
        raise LineageApiError(f"invalid computation kind: {e}") from e

    model_id = str(data.get("model_id") or "")
    if kind == ComputationKind.DETERMINISTIC and model_id:
        raise LineageApiError("DETERMINISTIC lineage cannot claim model_id")

    output = data.get("output")
    if isinstance(output, dict):
        if not (output.get("content_hash") or output.get("hash")):
            raise LineageApiError("output missing content hash")

    return ComputationalLineage.from_dict(data)


def lineage_to_contract(lineage: ComputationalLineage) -> dict[str, Any]:
    """Serialize lineage for cross-process exchange."""
    d = lineage.to_dict()
    d["api_schema_id"] = API_SCHEMA_ID
    d["api_schema_version"] = API_SCHEMA_VERSION
    return d
