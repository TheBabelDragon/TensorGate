"""Edge cases, corrupted artifacts, CLI, and flagship end-to-end."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tensorgate import (
    analyze,
    normalize,
    adapt,
    compare,
    check_compatibility,
    save_tensor,
    load_tensor,
    quantize,
    dequantize,
    TensorSpec,
)
from tensorgate.errors import SerializationError


def test_empty_array_descriptor():
    a = np.array([], dtype=np.float32)
    d = analyze(a)
    assert d.element_count == 0
    assert d.min is None
    assert d.max is None
    assert d.finite is True
    assert d.content_hash


def test_empty_normalize_maxabs():
    a = np.array([], dtype=np.float32)
    out, meta = normalize(a, method="maxabs")
    assert out.size == 0
    assert meta.get("degenerate") is True or meta["scale"] == 1.0


def test_all_zero_maxabs_no_explosion():
    a = np.zeros(8, dtype=np.float32)
    out, meta = normalize(a, method="maxabs")
    assert np.all(out == 0)
    assert meta["scale"] == 1.0
    assert meta.get("degenerate") is True


def test_constant_standard_no_nan():
    a = np.full(5, 7.0, dtype=np.float64)
    out, meta = normalize(a, method="standard")
    assert np.all(out == 0)
    assert not np.any(np.isnan(out))
    assert meta.get("degenerate") is True


def test_constant_range_maps_to_midpoint():
    a = np.full(4, 2.0, dtype=np.float32)
    out, meta = normalize(a, method="range", target_min=-1.0, target_max=1.0)
    assert np.allclose(out, 0.0)
    assert meta.get("degenerate") is True


def test_compatibility_status_adaptable():
    a = np.array([0.0, 2.0], dtype=np.float32)
    target = TensorSpec(dtype="float16", min_value=-1.0, max_value=1.0)
    report = check_compatibility(a, target)
    assert report.status == "adaptable"
    assert not report.compatible


def test_compatibility_status_incompatible_shape():
    a = np.zeros((2, 3), dtype=np.float32)
    target = TensorSpec(shape=(5, 5))
    report = check_compatibility(a, target)
    assert report.status == "incompatible"


def test_corrupted_sidecar_rejected():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t"
        save_tensor(a, p)
        data_path = p.with_suffix(".npy")
        bad = a + 99.0
        np.save(data_path, bad)
        with pytest.raises(SerializationError):
            load_tensor(p)


def test_mismatched_sidecar_hash():
    a = np.array([1.0, 2.0], dtype=np.float32)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t"
        save_tensor(a, p)
        meta_path = p.with_suffix(".tg.json")
        meta = json.loads(meta_path.read_text())
        meta["descriptor"]["content_hash"] = "deadbeef" * 8
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(SerializationError):
            load_tensor(p)


def test_flagship_end_to_end():
    rng = np.random.default_rng(0)
    source = rng.normal(0, 0.5, size=(32, 16)).astype(np.float32)
    source[0, 0] = 4.0

    desc0 = analyze(source)
    assert desc0.abs_max is not None and desc0.abs_max > 1.0

    target = TensorSpec(dtype="float16", min_value=-1.0, max_value=1.0)
    adapted, report = adapt(source, target)

    assert report.status in ("adaptable", "compatible")
    assert adapted.dtype == np.float16
    assert float(adapted.astype(np.float32).min()) >= -1.0 - 1e-3
    assert float(adapted.astype(np.float32).max()) <= 1.0 + 1e-3
    assert report.transformations
    assert report.provenance.chain or report.transformations

    err = compare(source, adapted.astype(np.float32))
    assert "max_absolute_error" in err
    assert err["changed_element_count"] > 0


def test_int8_quantize_roundtrip():
    a = np.linspace(-1, 1, 64, dtype=np.float32)
    q, meta = quantize(a, bits=8, scheme="symmetric", signed=True)
    assert q.dtype == np.int8
    recon, _ = dequantize(q, scale=meta["scale"], zero_point=meta["zero_point"])
    err = compare(a, recon)
    assert err["max_absolute_error"] < 0.02


def test_uint8_quantize():
    a = np.linspace(0, 1, 32, dtype=np.float32)
    q, meta = quantize(a, bits=8, scheme="asymmetric", signed=False)
    assert q.dtype == np.uint8
    assert meta["zero_point"] >= 0


def test_cli_inspect(tmp_path):
    a = np.array([1.0, -1.0, 0.5], dtype=np.float32)
    npy = tmp_path / "w.npy"
    np.save(npy, a)
    import os
    result = subprocess.run(
        [sys.executable, "-m", "tensorgate.cli", "inspect", str(npy)],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )
    assert result.returncode == 0
    assert "float32" in result.stdout


def test_hash_stable_across_processes():
    code = (
        "import numpy as np\n"
        "from tensorgate import analyze\n"
        "a = np.array([-0.8, 0.2, 0.7], dtype=np.float32)\n"
        "print(analyze(a).content_hash)\n"
    )
    import os
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    r1 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    r2 = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r1.returncode == 0 and r2.returncode == 0
    assert r1.stdout.strip() == r2.stdout.strip()
    assert len(r1.stdout.strip()) == 64


def test_no_silent_mutation_regression():
    a = np.array([-0.8, 0.2, 0.7], dtype=np.float32)
    out, meta = normalize(a, method="symmetric")
    assert "provenance" in meta
    assert meta["provenance"][0]["input_hash"]
    assert meta["provenance"][0]["output_hash"]
    assert meta["scale"] is not None
    expected = a.astype(np.float64) / 0.8
    np.testing.assert_allclose(out, expected, rtol=1e-5)
