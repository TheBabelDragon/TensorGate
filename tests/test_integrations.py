"""Integration adapter tests — no WaveBridge or Operator ABI required."""

from __future__ import annotations

import numpy as np
import pytest

from tensorgate import TensorSpec, analyze, compare
from tensorgate.integrations.wavebridge import (
    compare_roundtrip,
    from_wavebridge_payload,
    to_wavebridge_payload,
)
from tensorgate.integrations.operator_abi import (
    compare_tensors,
    prepare_input,
    validate_inputs,
    validate_output,
)


def test_wavebridge_adapter_import_without_wavebridge():
    """Adapters must import with NumPy only."""
    import tensorgate.integrations.wavebridge as wb
    assert hasattr(wb, "to_wavebridge_payload")


def test_operator_abi_adapter_import_without_abi():
    import tensorgate.integrations.operator_abi as oa
    assert hasattr(oa, "validate_inputs")


def test_wavebridge_payload_deterministic():
    rng = np.random.default_rng(0)
    w = rng.standard_normal(128).astype(np.float32)
    p1, m1 = to_wavebridge_payload(w, method="symmetric")
    p2, m2 = to_wavebridge_payload(w, method="symmetric")
    np.testing.assert_array_equal(p1, p2)
    assert m1["tensorgate_descriptor"]["content_hash"] == m2["tensorgate_descriptor"]["content_hash"]
    assert "normalization" in m1
    assert "provenance" in m1["normalization"]


def test_wavebridge_metadata_survives_boundary():
    w = np.array([-2.0, 0.5, 1.5], dtype=np.float32)
    payload, meta = to_wavebridge_payload(w, method="symmetric")
    recovered, rmeta = from_wavebridge_payload(payload, source_meta=meta)
    assert "tensorgate_descriptor" in rmeta
    assert rmeta["source_meta"]["source_descriptor"]["content_hash"] == meta["source_descriptor"]["content_hash"]
    err = compare_roundtrip(payload, recovered)
    assert err["max_absolute_error"] == 0.0


def test_wavebridge_no_silent_mutation():
    w = np.array([-0.8, 0.2, 0.7], dtype=np.float32)
    payload, meta = to_wavebridge_payload(w, method="symmetric")
    assert meta["normalization"]["provenance"][0]["input_hash"]
    assert meta["normalization"]["provenance"][0]["output_hash"]
    assert meta["normalization"]["scale"] is not None


def test_operator_validate_inputs_no_mutation():
    psi = np.random.randn(8).astype(np.float64)
    U = np.random.randn(8, 8).astype(np.float64)
    report = validate_inputs(
        psi, U,
        specs=[TensorSpec(dtype="float64", rank=1), TensorSpec(dtype="float64", rank=2)],
    )
    assert len(report["inputs"]) == 2
    assert report["inputs"][0]["compatibility"]["compatible"] is True
    assert report["inputs"][1]["compatibility"]["compatible"] is True
    assert psi.dtype == np.float64


def test_operator_incompatible_reported():
    a = np.zeros((2, 3), dtype=np.float32)
    report = validate_inputs(a, specs=[TensorSpec(shape=(5, 5))])
    compat = report["inputs"][0]["compatibility"]
    assert compat["compatible"] is False
    assert compat["status"] == "incompatible"


def test_operator_prepare_no_target_identity():
    a = np.array([1.0, 2.0], dtype=np.float32)
    out, meta = prepare_input(a)
    np.testing.assert_array_equal(out, a)
    assert meta["adapted"] is False


def test_operator_complex_inspection():
    z = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
    report = validate_output(z)
    assert report["is_complex"] is True
    assert report["descriptor"]["element_count"] == 4


def test_operator_compare_explicit():
    a = np.array([1.0, 2.0, 3.0])
    b = a + 0.01
    err = compare_tensors(a, b)
    assert err["max_absolute_error"] > 0
    assert err["changed_element_count"] == 3
