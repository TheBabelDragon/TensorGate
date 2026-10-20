"""Quantum-state numerical integrity tests for TensorGate."""
from __future__ import annotations
import math
import tempfile
from pathlib import Path
import numpy as np
import pytest
from tensorgate import analyze, compare, convert_dtype, load_tensor, save_tensor, SerializationError

EXPERIMENT_VERSION = "1.0.0"
DEFAULT_SEED = 42
TOL_C128, TOL_C64, TOL_FID, TOL_NORM = 1e-15, 1e-6, 1e-12, 1e-14

def squared_norm(psi):
    return float(np.real(np.vdot(psi, psi)))

def max_amp_err(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b))
    return 0.0 if d.size == 0 else float(np.max(d))

def fidelity(psi, phi):
    return float(np.abs(np.vdot(psi, phi)) ** 2)

def is_norm(psi, tol=TOL_NORM):
    return abs(squared_norm(psi) - 1.0) <= tol

def normalize(psi):
    n = math.sqrt(squared_norm(psi))
    if n == 0:
        raise ValueError("zero vector")
    return psi / n

def basis(d, i, dtype=np.complex128):
    v = np.zeros(d, dtype=dtype); v[i] = 1.0; return v

def superpos(d, dtype=np.complex128):
    return np.ones(d, dtype=dtype) / math.sqrt(d)

def bell_phi_plus(dtype=np.complex128):
    v = np.zeros(4, dtype=dtype); v[0] = v[3] = 1/math.sqrt(2); return v

def seeded(dim, seed, dtype=np.complex128):
    rng = np.random.default_rng(seed)
    return normalize((rng.standard_normal(dim) + 1j * rng.standard_normal(dim)).astype(dtype))

def roundtrip(psi):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s"
        save_tensor(psi, p)
        return load_tensor(p)

class TestMetrics:
    def test_norm_basis(self):
        assert abs(squared_norm(basis(4, 1)) - 1) < TOL_NORM
    def test_fidelity_phase(self):
        psi = superpos(4)
        phi = (psi * np.exp(1j * 0.7)).astype(psi.dtype)
        assert abs(fidelity(psi, phi) - 1) < TOL_FID

class TestRoundTrip:
    @pytest.mark.parametrize("d,i", [(2,0),(2,1),(4,0),(8,3)])
    def test_basis(self, d, i):
        psi = basis(d, i)
        rec, meta = roundtrip(psi)
        assert max_amp_err(psi, rec) <= TOL_C128
        assert abs(fidelity(psi, rec) - 1) < TOL_FID
        assert meta["descriptor"]["content_hash"] == analyze(psi).content_hash

    def test_bell(self):
        psi = bell_phi_plus()
        rec, _ = roundtrip(psi)
        assert max_amp_err(psi, rec) <= TOL_C128
        assert abs(fidelity(psi, rec) - 1) < TOL_FID

    @pytest.mark.parametrize("d", [2, 8, 16])
    def test_random(self, d):
        psi = seeded(d, DEFAULT_SEED + d)
        rec, _ = roundtrip(psi)
        assert max_amp_err(psi, rec) <= TOL_C128
        assert abs(fidelity(psi, rec) - 1) < TOL_FID

    def test_c64(self):
        psi = seeded(8, DEFAULT_SEED, dtype=np.complex64)
        rec, _ = roundtrip(psi)
        assert rec.dtype == np.complex64
        assert max_amp_err(psi, rec) <= TOL_C64

    def test_repeated(self):
        psi = seeded(16, DEFAULT_SEED)
        cur = psi
        for _ in range(5):
            cur, _ = roundtrip(cur)
        assert max_amp_err(psi, cur) <= TOL_C128

class TestDtype:
    def test_c128_to_c64(self):
        psi = seeded(16, DEFAULT_SEED)
        out, meta = convert_dtype(psi, "complex64")
        assert out.dtype == np.complex64
        err = max_amp_err(psi, out.astype(np.complex128))
        assert err < 1e-5 and math.isfinite(err)
        assert meta["input_hash"] != meta["output_hash"]

    def test_compare_complex(self):
        psi = seeded(8, DEFAULT_SEED)
        out, _ = convert_dtype(psi, "complex64")
        r = compare(psi, out)
        assert r["max_absolute_error"] >= 0

class TestEdge:
    def test_zero(self):
        z = np.zeros(4, dtype=np.complex128)
        rec, _ = roundtrip(z)
        assert np.array_equal(rec, z)

    def test_empty(self):
        e = np.array([], dtype=np.complex128)
        rec, _ = roundtrip(e)
        assert rec.shape == (0,)

    def test_nan(self):
        psi = superpos(4).copy(); psi[0] = np.nan
        d = analyze(psi)
        assert d.nan_count >= 1 and not d.finite

    def test_inf(self):
        psi = superpos(4).copy(); psi[1] = np.inf
        d = analyze(psi)
        assert not d.finite

    def test_hash_mismatch(self):
        psi = basis(2, 0)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s"
            save_tensor(psi, p)
            np.save(p.with_suffix(".npy"), basis(2, 1))
            with pytest.raises(SerializationError, match="Content hash mismatch"):
                load_tensor(p)

    def test_missing(self):
        with tempfile.TemporaryDirectory() as td:
            with pytest.raises(SerializationError, match="not found"):
                load_tensor(Path(td) / "nope")
