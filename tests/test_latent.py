"""Tests for Neuralese latent-state representation contract."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tensorgate.errors import SerializationError, ValidationError
from tensorgate.latent import (
    LATENT_SCHEMA_ID,
    LATENT_SCHEMA_VERSION,
    LatentCompatibilityLevel,
    LatentStateSpec,
    build_latent_lineage,
    check_latent_compatibility,
    compare_latent_states,
    describe_latent_state,
    deserialize_latent_state,
    latent_content_hash,
    latent_identity,
    serialize_latent_state,
)
from tensorgate.lineage import ComputationKind, check_lineage_schema


def _spec(**kwargs) -> LatentStateSpec:
    defaults = dict(
        shape=(2, 4),
        dtype="float32",
        representation_semantics="hidden_state",
        model_id="toy-mlp",
        model_revision="rev1",
        architecture_id="mlp-v1",
        layer_id="block.0",
    )
    defaults.update(kwargs)
    return LatentStateSpec(**defaults)


def test_valid_spec_and_roundtrip():
    arr = np.arange(8, dtype=np.float32).reshape(2, 4)
    spec = _spec()
    env = serialize_latent_state(arr, spec, label="h0")
    out, spec2, ident = deserialize_latent_state(env, require_data=True)
    assert out is not None
    assert np.allclose(out, arr)
    assert spec2.model_id == "toy-mlp"
    assert ident.content_hash == env["latent_id"]
    assert env["schema_id"] == LATENT_SCHEMA_ID
    assert env["schema_version"] == LATENT_SCHEMA_VERSION


def test_stable_identity_same_content_and_meta():
    arr = np.ones((2, 4), dtype=np.float32)
    spec = _spec()
    h1 = latent_content_hash(arr, spec)
    h2 = latent_content_hash(arr.copy(), _spec())
    assert h1 == h2
    id1 = latent_identity(arr, spec)
    id2 = latent_identity(arr, _spec())
    assert id1.content_hash == id2.content_hash


def test_identity_changes_with_content_or_meta():
    arr = np.ones((2, 4), dtype=np.float32)
    base = latent_content_hash(arr, _spec())
    mutated = arr.copy()
    mutated[0, 0] = 2.0
    assert latent_content_hash(mutated, _spec()) != base
    assert latent_content_hash(arr, _spec(layer_id="block.1")) != base
    assert latent_content_hash(arr, _spec(model_id="other")) != base


def test_matching_shape_not_semantic_compat_across_models():
    a = _spec(model_id="model-a", layer_id="enc.0")
    b = _spec(model_id="model-b", layer_id="enc.0")
    report = check_latent_compatibility(a, b)
    assert report.level == LatentCompatibilityLevel.INCOMPATIBLE
    assert report.compatible is False
    assert any("model_id" in i for i in report.issues)


def test_matching_shape_not_semantic_compat_across_layers():
    a = _spec(layer_id="block.0")
    b = _spec(layer_id="block.7")
    report = check_latent_compatibility(a, b)
    assert report.level == LatentCompatibilityLevel.INCOMPATIBLE
    assert any("layer_id" in i for i in report.issues)


def test_shape_dtype_schema_incompatibilities():
    a = _spec(shape=(2, 4))
    b = _spec(shape=(4, 2))
    r = check_latent_compatibility(a, b)
    assert r.level == LatentCompatibilityLevel.INCOMPATIBLE
    r2 = check_latent_compatibility(_spec(dtype="float32"), _spec(dtype="float64"))
    assert r2.level == LatentCompatibilityLevel.INCOMPATIBLE
    with pytest.raises(ValidationError):
        LatentStateSpec(
            shape=(2,),
            dtype="float32",
            schema_version="2.0.0",
            representation_semantics="x",
            model_id="m",
            layer_id="l",
        )
    with pytest.raises(ValidationError):
        LatentStateSpec(shape=(1,), dtype="float32", schema_version="9.0.0")


def test_missing_producer_metadata_inconclusive():
    a = LatentStateSpec(shape=(2, 4), dtype="float32")
    b = LatentStateSpec(shape=(2, 4), dtype="float32")
    r = check_latent_compatibility(a, b)
    assert r.level == LatentCompatibilityLevel.INCONCLUSIVE
    assert r.compatible is False


def test_exact_compatibility_when_fully_specified():
    r = check_latent_compatibility(_spec(), _spec())
    assert r.level == LatentCompatibilityLevel.EXACT
    assert r.compatible is True


def test_numerical_convertible_dtype():
    r = check_latent_compatibility(
        _spec(dtype="float32"), _spec(dtype="float16"), allow_dtype_conversion=True
    )
    assert r.level == LatentCompatibilityLevel.NUMERICAL_CONVERTIBLE
    assert r.compatible is True


def test_invalid_serialized_metadata():
    with pytest.raises(SerializationError):
        deserialize_latent_state({"schema_id": "other.schema", "schema_version": "1.0.0"})
    with pytest.raises(SerializationError):
        deserialize_latent_state({"schema_id": LATENT_SCHEMA_ID, "schema_version": "99.0.0"})


def test_identity_mismatch_on_deserialize():
    arr = np.ones((2, 4), dtype=np.float32)
    env = serialize_latent_state(arr, _spec())
    env["latent_id"] = "deadbeef"
    env["identity"]["content_hash"] = "deadbeef"
    with pytest.raises(SerializationError, match="identity mismatch"):
        deserialize_latent_state(env, require_data=True)


def test_nan_inf_and_empty():
    arr = np.array([[np.nan, 1.0], [np.inf, -1.0]], dtype=np.float32)
    spec = LatentStateSpec(
        shape=(2, 2), dtype="float32", representation_semantics="test", model_id="m", layer_id="l"
    )
    desc = describe_latent_state(arr, spec)
    assert desc.tensor.finite is False
    empty = np.zeros((0, 4), dtype=np.float32)
    espec = LatentStateSpec(
        shape=(0, 4), dtype="float32", representation_semantics="empty", model_id="m", layer_id="l"
    )
    assert describe_latent_state(empty, espec).tensor.shape == (0, 4)


def test_scalar_tensor():
    arr = np.array(3.14, dtype=np.float64)
    spec = LatentStateSpec(
        shape=(), dtype="float64", representation_semantics="scalar", model_id="m", layer_id="l"
    )
    env = serialize_latent_state(arr, spec)
    out, _, _ = deserialize_latent_state(env, require_data=True)
    assert out is not None and out.shape == () and float(out) == pytest.approx(3.14)


def test_float16_roundtrip():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    spec = LatentStateSpec(
        shape=(3,), dtype="float16", representation_semantics="h", model_id="m", layer_id="l"
    )
    env = serialize_latent_state(arr, spec)
    out, s2, _ = deserialize_latent_state(env, require_data=True)
    assert s2.dtype == "float16" and out is not None and out.dtype == np.float16


def test_complex_if_supported():
    arr = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    spec = LatentStateSpec(
        shape=(2,), dtype="complex64", representation_semantics="c", model_id="m", layer_id="l"
    )
    env = serialize_latent_state(arr, spec)
    out, _, ident = deserialize_latent_state(env, require_data=True)
    assert out is not None and np.allclose(out, arr)
    assert ident.content_hash == env["latent_id"]


def test_compare_latent_states_metrics():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    sa = LatentStateSpec(
        shape=(3,), dtype="float32", representation_semantics="x", model_id="m", layer_id="l"
    )
    sb = LatentStateSpec(
        shape=(3,), dtype="float32", representation_semantics="x", model_id="m", layer_id="l"
    )
    r = compare_latent_states(a, b, sa, sb)
    assert r["status"] == "compared"
    assert r["metrics"]["max_abs_diff"] == pytest.approx(0.0)
    b2 = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    r2 = compare_latent_states(a, b2, sa, sb)
    assert r2["metrics"]["max_abs_diff"] == pytest.approx(1.0)


def test_compare_does_not_silently_reshape():
    a = np.ones((2, 2), dtype=np.float32)
    b = np.ones((4,), dtype=np.float32)
    sa = LatentStateSpec(
        shape=(2, 2), dtype="float32", representation_semantics="x", model_id="m", layer_id="l"
    )
    sb = LatentStateSpec(
        shape=(4,), dtype="float32", representation_semantics="x", model_id="m", layer_id="l"
    )
    assert compare_latent_states(a, b, sa, sb)["status"] == "incompatible_shape"


def test_latent_lineage_nondeterministic_default():
    arr = np.ones((2, 4), dtype=np.float32)
    ident = latent_identity(arr, _spec(), label="in")
    out_ident = latent_identity(arr * 2, _spec(), label="out")
    lin = build_latent_lineage(
        operation="model_forward/hidden",
        input_identity=ident,
        output_identity=out_ident,
        model_id="toy-mlp",
        model_version="rev1",
        layer_id="block.0",
    )
    assert lin.kind == ComputationKind.NONDETERMINISTIC
    assert lin.model_id == "toy-mlp"
    check_lineage_schema(lin)


def test_latent_lineage_deterministic_transform():
    arr = np.ones((2, 4), dtype=np.float32)
    ident = latent_identity(arr, _spec(), label="in")
    out = arr.astype(np.float64)
    out_ident = latent_identity(out, _spec(dtype="float64"), label="out")
    lin = build_latent_lineage(
        operation="convert_dtype/float64",
        input_identity=ident,
        output_identity=out_ident,
        kind=ComputationKind.DETERMINISTIC,
        parameters={"from": "float32", "to": "float64"},
    )
    assert lin.kind == ComputationKind.DETERMINISTIC
    check_lineage_schema(lin)


def test_describe_latent_state_api():
    arr = np.zeros((1, 8), dtype=np.float32)
    d = describe_latent_state(
        arr, model_id="m", layer_id="l0", representation_semantics="hidden_state"
    )
    assert d.spec.model_id == "m" and d.latent_id and d.tensor.shape == (1, 8)


def test_json_roundtrip_string():
    arr = np.linspace(0, 1, 8, dtype=np.float32).reshape(2, 4)
    env = serialize_latent_state(arr, _spec())
    out, _, ident = deserialize_latent_state(json.dumps(env, sort_keys=True), require_data=True)
    assert out is not None and ident.content_hash == env["latent_id"]


def test_existing_core_api_still_imports():
    import tensorgate as tg

    assert hasattr(tg, "TensorSpec")
    assert hasattr(tg, "check_compatibility")
    assert hasattr(tg, "ComputationalLineage")
    assert hasattr(tg, "LatentStateSpec")
    assert hasattr(tg, "check_latent_compatibility")
