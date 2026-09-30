"""Core TensorGate numerical tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tensorgate import (
    analyze,
    describe,
    normalize,
    convert_dtype,
    clip,
    quantize,
    dequantize,
    check_compatibility,
    adapt,
    compare,
    save_tensor,
    load_tensor,
    TensorSpec,
    Layout,
)
from tensorgate.compatibility import compare as tg_compare


def zeros():
    return np.zeros((4, 4), dtype=np.float32)

def ones():
    return np.ones((3, 5), dtype=np.float64)

def mixed():
    return np.array([-0.8, 0.2, 0.7, -0.1, 0.0], dtype=np.float32)

def huge_range():
    return np.array([1e-8, 1e8, -1e7, 0.0], dtype=np.float64)

def tiny_range():
    return np.array([0.001, 0.0011, 0.0009], dtype=np.float32)

def negative_only():
    return np.array([-3.0, -1.0, -0.5], dtype=np.float32)

def positive_only():
    return np.array([0.1, 2.0, 5.0], dtype=np.float32)

def with_nan():
    a = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    return a

def with_inf():
    return np.array([1.0, np.inf, -np.inf], dtype=np.float32)

def sparse():
    a = np.zeros(100, dtype=np.float32)
    a[50] = 1.0
    return a

def repeated():
    return np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)

def large():
    rng = np.random.default_rng(42)
    return rng.standard_normal((64, 64)).astype(np.float32)


def test_descriptor_basic():
    a = mixed()
    d = analyze(a)
    assert d.dtype == "float32"
    assert d.shape == (5,)
    assert d.rank == 1
    assert d.element_count == 5
    assert d.finite is True
    assert d.nan_count == 0
    assert abs(d.min + 0.8) < 1e-6
    assert abs(d.max - 0.7) < 1e-6
    assert d.content_hash


def test_nan_detection():
    d = analyze(with_nan())
    assert d.nan_count == 1
    assert d.finite is False


def test_inf_detection():
    d = analyze(with_inf())
    assert d.pos_inf_count == 1
    assert d.neg_inf_count == 1
    assert d.finite is False


def test_sparsity():
    d = analyze(sparse())
    assert d.zero_count == 99
    assert abs(d.sparsity - 0.99) < 1e-6


def test_hash_deterministic():
    a = mixed()
    h1 = analyze(a).content_hash
    h2 = analyze(a.copy()).content_hash
    assert h1 == h2


def test_hash_differs_on_value_change():
    a = mixed()
    b = a.copy()
    b[0] = 0.0
    assert analyze(a).content_hash != analyze(b).content_hash


def test_maxabs_normalize():
    a = mixed()
    out, meta = normalize(a, method="maxabs")
    assert meta["method"] == "maxabs"
    assert abs(np.max(np.abs(out)) - 1.0) < 1e-5
    assert "provenance" in meta


def test_symmetric_normalize():
    a = mixed()
    out, meta = normalize(a, method="symmetric", target_min=-1.0, target_max=1.0)
    assert meta["method"] == "symmetric"
    assert np.max(out) <= 1.0 + 1e-5
    assert np.min(out) >= -1.0 - 1e-5


def test_affine_normalize():
    a = positive_only()
    out, meta = normalize(a, method="affine", target_min=0.0, target_max=1.0)
    assert abs(out.min()) < 1e-5
    assert abs(out.max() - 1.0) < 1e-5


def test_standard_normalize():
    a = large()
    out, meta = normalize(a, method="standard")
    assert abs(out.mean()) < 1e-5
    assert abs(out.std() - 1.0) < 1e-4


def test_no_silent_change():
    a = np.array([-0.8, 0.2, 0.7], dtype=np.float32)
    out, meta = normalize(a, method="symmetric")
    assert "provenance" in meta
    assert meta["provenance"][0]["operation"].startswith("normalize/")
    assert "scale" in meta
    assert meta["source_abs_max"] is not None


def test_quantize_dequantize_roundtrip():
    a = mixed()
    q, qmeta = quantize(a, bits=8, scheme="symmetric")
    recon, dmeta = dequantize(q, scale=qmeta["scale"], zero_point=qmeta["zero_point"])
    err = compare(a, recon)
    assert err["max_absolute_error"] < 0.05


def test_asymmetric_quantize():
    a = positive_only()
    q, meta = quantize(a, bits=8, scheme="asymmetric", signed=False)
    assert meta["scheme"] == "asymmetric"
    assert q.dtype == np.uint8


def test_convert_dtype():
    a = mixed()
    out, meta = convert_dtype(a, "float64")
    assert out.dtype == np.float64
    assert meta["source_dtype"] == "float32"


def test_clip():
    a = mixed()
    out, meta = clip(a, min_value=-0.5, max_value=0.5)
    assert out.min() >= -0.5
    assert out.max() <= 0.5
    assert meta["clipped_count"] > 0


def test_compatibility_success():
    a = mixed()
    spec = TensorSpec(dtype="float32", rank=1, allow_nan=False)
    report = check_compatibility(a, spec)
    assert report.compatible


def test_compatibility_dtype_mismatch():
    a = mixed()
    spec = TensorSpec(dtype="float64")
    report = check_compatibility(a, spec)
    assert not report.compatible
    assert any("dtype" in i for i in report.issues)


def test_compatibility_shape_mismatch():
    a = mixed()
    spec = TensorSpec(shape=(10,))
    report = check_compatibility(a, spec)
    assert not report.compatible


def test_adapt_range():
    a = np.array([-2.0, 0.5, 3.0], dtype=np.float32)
    target = TensorSpec(min_value=-1.0, max_value=1.0, dtype="float32")
    adapted, report = adapt(a, target)
    assert adapted.min() >= -1.0 - 1e-6
    assert adapted.max() <= 1.0 + 1e-6
    assert report.provenance.chain or report.transformations


def test_serialization_roundtrip():
    a = large()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t"
        save_tensor(a, p)
        loaded, meta = load_tensor(p)
        np.testing.assert_array_equal(a, loaded)
        assert meta["descriptor"]["content_hash"] == analyze(a).content_hash


def test_hash_stable_across_save_load():
    a = mixed()
    h0 = analyze(a).content_hash
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t"
        save_tensor(a, p)
        loaded, _ = load_tensor(p)
        assert analyze(loaded).content_hash == h0


def test_compare_identical():
    a = mixed()
    err = compare(a, a.copy())
    assert err["max_absolute_error"] == 0.0
    assert err["changed_element_count"] == 0


def test_compare_different():
    a = mixed()
    b = a + 0.1
    err = compare(a, b)
    assert err["max_absolute_error"] > 0.09
    assert err["changed_element_count"] == a.size


@pytest.mark.parametrize("factory", [
    zeros, ones, mixed, huge_range, tiny_range,
    negative_only, positive_only, sparse, repeated, large,
])
def test_pathological_analyze(factory):
    a = factory()
    d = analyze(a)
    assert d.element_count == a.size
    assert d.content_hash


def test_all_zeros_normalize():
    a = zeros()
    out, meta = normalize(a, method="maxabs")
    assert np.all(out == 0)
