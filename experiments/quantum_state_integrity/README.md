# Quantum-State Numerical Integrity Experiment

**Experiment version:** 1.0.0  
**Branch:** `feat/quantum-state-integrity`  
**Scope:** Classical numerical validation of complex-valued state vectors under TensorGate serialization, deserialization, validation, and dtype-conversion boundaries.

This is **not** a quantum simulator. No gates, circuits, hardware backends, or claims of quantum advantage are involved.

## Hypothesis

TensorGate preserves complex-valued quantum state vectors (amplitude arrays) across its existing supported boundaries:

1. `save_tensor` / `load_tensor` round-trips retain numerical values within dtype-appropriate tolerances.
2. Content hashes and provenance side-cars continue to enforce integrity contracts.
3. Lossy dtype conversions (e.g. `complex128` → `complex64`) are explicit and produce measurable, reportable error.
4. Invalid or non-finite inputs are detected according to existing contracts (analyze / SerializationError).
5. Repeated lossless round-trips introduce no unexplained numerical drift.

## Methodology

### Inputs (deterministic fixtures)

| Class | Description |
|-------|-------------|
| Computational basis | basis vectors in C^d |
| Equal superposition | equal amplitude superpositions |
| Bell amplitude vectors | Phi+ and Psi- as length-4 arrays |
| Seeded random normalized | Fixed numpy RNG seed normalized complex vectors, dims 2..32 |
| Global-phase variants | phase-invariant fidelity checks |
| Edge | Zero vector, empty array, NaN, Inf |

### Independent reference metrics

Computed **outside** TensorGate internals:

- **Squared norm:** sum |psi_i|^2
- **Max absolute amplitude error:** max |psi_i - phi_i|
- **State fidelity:** |vdot(psi, phi)|^2 (only when both states pass independent normalization checks)
- **Norm deviation:** | ||psi||^2 - 1 |

Global-phase-invariant fidelity is used only after independent verification that both states are normalized. Reconstructed data is **never** renormalized to hide defects.

### Tolerances

| Dtype / quantity | Tolerance |
|------------------|-----------|
| complex128 amplitude | 1e-15 |
| complex64 amplitude | 1e-6 |
| Fidelity (normalized) | 1e-12 |
| Norm deviation | 1e-14 |

Lossy conversions report measured error rather than requiring bitwise equality.

### Public APIs exercised

- analyze / build_descriptor
- save_tensor / load_tensor
- convert_dtype
- compare
- Content-hash mismatch -> SerializationError

## Execution

From the repository root:

```bash
# Automated tests (pytest)
PYTHONPATH=. python -m pytest tests/test_quantum_state_integrity.py -v

# Experiment runner -> machine-readable JSON report
PYTHONPATH=. python experiments/quantum_state_integrity/run_experiment.py
PYTHONPATH=. python experiments/quantum_state_integrity/run_experiment.py --seed 42 --out report.json
```

The runner exits with status **1** if any required correctness assertion fails.

## Report fields

The JSON report includes:

- experiment_version, random_seed
- Runtime: Python, platform, NumPy, TensorGate versions
- cases_executed / cases_passed / cases_failed
- maximum_amplitude_error, minimum_observed_fidelity, maximum_norm_deviation
- dtype_conversions_exercised
- invalid_input_handling summary
- Per-case failure details sufficient to reproduce

## Core library note (complex support)

Prior to this branch, build_descriptor and compare assumed real floating-point semantics (np.isposinf / forced float64 cast), which raised on complex dtypes even though complexfloating was already present in the control flow. Minimal, targeted fixes were applied so that:

- Complex NaN / Inf accounting uses isnan / isinf (and real/imag components where signed inf is needed).
- Aggregate descriptor stats for complex arrays use magnitude.
- compare preserves complex arithmetic and reports modulus-based errors.

These changes are required for the experiment to exercise the public API on complex state vectors; they do not alter real-tensor behaviour (existing tests remain green).

## Limitations

- TensorGate remains a numerical compatibility layer. It does not implement quantum operators, measurement, or hardware.
- Only dtypes and operations already supported (or minimally corrected for complex) are tested.
- Quantization / normalization paths aimed at real weights are not applied to quantum states in this experiment.
- Empty and zero vectors are checked for amplitude identity only; fidelity is undefined for the zero vector and is not asserted there.
- Content hashes intentionally change under dtype conversion; the experiment verifies the existing hash contract rather than requiring hash stability across representation changes.

## Interpretation

- All valid lossless cases pass within tolerance -> amplitudes and norms survive the boundary.
- Fidelity ~ 1 for normalized pairs (including global-phase equivalents) -> relative geometry is preserved.
- Lossy conversion cases pass with finite reported error -> the boundary is explicit, not silent.
- Invalid-input cases pass -> contracts (hash mismatch, missing data, NaN/Inf detection) hold.
- Any failure is listed in the report with name, metrics, and exception detail; the process exit code reflects correctness.

## Related tests

`tests/test_quantum_state_integrity.py` mirrors the experiment fixtures and assertions under pytest for CI integration.
