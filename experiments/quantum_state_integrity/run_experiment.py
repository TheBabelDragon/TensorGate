#!/usr/bin/env python3
"""Quantum-state numerical integrity experiment runner. Exit 1 on assertion failure."""
from __future__ import annotations
import argparse, json, math, platform, sys, tempfile, traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tensorgate import __version__ as TG_VERSION, analyze, convert_dtype, load_tensor, save_tensor, SerializationError

EXPERIMENT_VERSION, DEFAULT_SEED = "1.0.0", 42
TOL_C128, TOL_C64, TOL_FID, TOL_NORM = 1e-15, 1e-6, 1e-12, 1e-14

def squared_norm(psi): return float(np.real(np.vdot(psi, psi)))
def max_amp_err(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b))
    return 0.0 if d.size == 0 else float(np.max(d))
def fidelity(psi, phi): return float(np.abs(np.vdot(psi, phi)) ** 2)
def is_norm(psi, tol=TOL_NORM): return abs(squared_norm(psi) - 1.0) <= tol
def normalize(psi):
    n = math.sqrt(squared_norm(psi))
    if n == 0: raise ValueError("zero")
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
        p = Path(td) / "s"; save_tensor(psi, p); return load_tensor(p)

class CaseResult:
    def __init__(self, name):
        self.name, self.passed, self.category = name, False, "valid"
        self.max_amplitude_error = self.fidelity = self.norm_deviation = None
        self.dtype_conversion = self.detail = ""; self.error = None
    def to_dict(self):
        return {k: getattr(self, k) for k in
                ("name","passed","category","max_amplitude_error","fidelity",
                 "norm_deviation","dtype_conversion","detail","error")}

def run_valid(name, psi, tol, require_norm=True):
    r = CaseResult(name); r.category = "valid_roundtrip"
    try:
        rec, meta = roundtrip(psi)
        mae = max_amp_err(psi, rec)
        r.max_amplitude_error = mae
        r.norm_deviation = abs(squared_norm(rec) - (1.0 if require_norm else squared_norm(psi)))
        if require_norm and is_norm(psi):
            r.fidelity = fidelity(psi, rec)
            fid_ok = abs(r.fidelity - 1.0) < TOL_FID
        else:
            fid_ok = True
        r.passed = bool(mae <= tol and fid_ok and rec.dtype == psi.dtype and rec.shape == psi.shape)
        if not r.passed:
            r.detail = f"mae={mae} fid={r.fidelity}"
    except Exception as e:
        r.passed = False; r.error = f"{type(e).__name__}: {e}"
    return r

def run_lossy(name, psi, target):
    r = CaseResult(name); r.category = "lossy_conversion"
    r.dtype_conversion = f"{psi.dtype}->{target}"
    try:
        out, meta = convert_dtype(psi, target)
        mae = max_amp_err(psi.astype(np.complex128), out.astype(np.complex128))
        r.max_amplitude_error = mae
        r.fidelity = fidelity(psi.astype(np.complex128), out.astype(np.complex128)) if is_norm(psi) else None
        r.passed = math.isfinite(mae) and out.dtype == np.dtype(target)
        if str(psi.dtype) != target and meta.get("input_hash") == meta.get("output_hash"):
            r.passed = False; r.detail = "expected distinct hashes"
    except Exception as e:
        r.passed = False; r.error = str(e)
    return r

def run_invalid(name, fn):
    r = CaseResult(name); r.category = "invalid_input"
    try:
        fn(); r.passed = True; r.detail = "behaved as expected"
    except Exception as e:
        r.passed = False; r.error = str(e)
    return r

def build_cases(seed):
    results = []
    for d in (2, 4, 8):
        for i in range(min(d, 2)):
            results.append(run_valid(f"basis_d{d}_i{i}", basis(d, i), TOL_C128))
    for d in (2, 4, 8):
        results.append(run_valid(f"superpos_d{d}", superpos(d), TOL_C128))
    results.append(run_valid("bell_phi_plus", bell_phi_plus(), TOL_C128))
    for d in (2, 8, 16):
        results.append(run_valid(f"random_d{d}", seeded(d, seed + d), TOL_C128))
    results.append(run_valid("complex64_rt", seeded(8, seed, dtype=np.complex64), TOL_C64))
    results.append(run_lossy("c128_to_c64", seeded(16, seed), "complex64"))
    results.append(run_valid("zero", np.zeros(4, dtype=np.complex128), TOL_C128, False))
    results.append(run_valid("empty", np.array([], dtype=np.complex128), TOL_C128, False))

    def _hash_mm():
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"s"; save_tensor(basis(2,0), p)
            np.save(p.with_suffix(".npy"), basis(2,1))
            try: load_tensor(p); raise AssertionError("expected error")
            except SerializationError: pass
    results.append(run_invalid("hash_mismatch", _hash_mm))

    def _nan():
        psi = superpos(4).copy(); psi[0] = np.nan
        d = analyze(psi); assert d.nan_count >= 1 and not d.finite
    results.append(run_invalid("nan_detected", _nan))
    return results

def aggregate(results, seed):
    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    maes = [r.max_amplitude_error for r in results if r.max_amplitude_error is not None]
    fids = [r.fidelity for r in results if r.fidelity is not None]
    norms = [r.norm_deviation for r in results if r.norm_deviation is not None]
    return {
        "experiment_version": EXPERIMENT_VERSION, "random_seed": seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "tensorgate": TG_VERSION},
        "cases_executed": len(results), "cases_passed": len(passed), "cases_failed": len(failed),
        "maximum_amplitude_error": max(maes) if maes else None,
        "minimum_observed_fidelity": min(fids) if fids else None,
        "maximum_norm_deviation": max(norms) if norms else None,
        "dtype_conversions_exercised": sorted({r.dtype_conversion for r in results if r.dtype_conversion}),
        "failures": [r.to_dict() for r in failed],
        "all_cases": [r.to_dict() for r in results],
    }

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    report = aggregate(build_cases(args.seed), args.seed)
    text = json.dumps(report, indent=2, sort_keys=True)
    out = args.out or (Path(__file__).resolve().parent / "report.json")
    out.write_text(text)
    print(f"Wrote {out}")
    print(f"cases={report['cases_executed']} passed={report['cases_passed']} failed={report['cases_failed']}")
    return 1 if report["cases_failed"] else 0

if __name__ == "__main__":
    sys.exit(main())
