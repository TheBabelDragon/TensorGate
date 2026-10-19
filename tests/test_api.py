"""Lineage API contract tests — fail-closed on malformed peer data."""

from __future__ import annotations

import pytest

from tensorgate.api import (
    API_SCHEMA_VERSION,
    LineageApiError,
    health,
    lineage_to_contract,
    negotiate,
    validate_lineage_dict,
)
from tensorgate.lineage import ComputationKind, ComputationalLineage, build_lineage


def test_health():
    h = health()
    assert h["healthy"] is True
    assert h["schema_version"] == API_SCHEMA_VERSION


def test_negotiate_compatible():
    assert negotiate("1.0.0")["compatible"] is True


def test_negotiate_incompatible_major():
    assert negotiate("9.0.0")["compatible"] is False


def test_reject_unknown_schema():
    with pytest.raises(LineageApiError, match="unknown schema"):
        validate_lineage_dict({"schema_id": "evil.schema", "kind": "DETERMINISTIC"})


def test_reject_missing_output_hash():
    with pytest.raises(LineageApiError, match="content hash"):
        validate_lineage_dict(
            {
                "schema_id": "tensorgate.lineage",
                "kind": "DETERMINISTIC",
                "output": {"shape": [1, 2], "dtype": "float32"},
            }
        )


def test_reject_deterministic_model_id():
    with pytest.raises(LineageApiError, match="model_id"):
        validate_lineage_dict(
            {
                "schema_id": "tensorgate.lineage",
                "kind": "DETERMINISTIC",
                "model_id": "some-model",
                "output": {"content_hash": "abc123"},
            }
        )


def test_accept_valid_deterministic():
    lin = validate_lineage_dict(
        {
            "schema_id": "tensorgate.lineage",
            "schema_version": "0.1.0",
            "kind": "DETERMINISTIC",
            "transformation_id": "normalize",
            "model_id": "",
            "output": {"content_hash": "deadbeef"},
        }
    )
    assert lin.kind == ComputationKind.DETERMINISTIC
    assert not lin.model_id


def test_lineage_to_contract_roundtrip():
    lin = build_lineage(
        kind=ComputationKind.DETERMINISTIC,
        transformation_id="identity",
    )
    d = lineage_to_contract(lin)
    assert d["api_schema_id"] == "tensorgate.lineage_api"
    assert d["api_schema_version"] == API_SCHEMA_VERSION
    # Must not invent validity for peer payloads missing hashes when re-validated
    with pytest.raises(LineageApiError):
        validate_lineage_dict(
            {
                "schema_id": "tensorgate.lineage",
                "kind": "DETERMINISTIC",
                "output": {"dtype": "float32"},  # no hash
            }
        )
