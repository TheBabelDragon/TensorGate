# TensorGate

**TensorGate is where tensors go when they need to cross a boundary.**

Canonical numerical compatibility boundary for tensors and neural-network weights across the BabelDragon compute stack.

```
arbitrary tensor / neural-network weights
        ↓
    inspect
        ↓
    describe
        ↓
    normalize / transform
        ↓
    validate
        ↓
    serialize
        ↓
    hand to another model, operator, accelerator, or physical transport
```

## What TensorGate is

- Numerical representation and compatibility layer
- Explicit, inspectable transformations (never silent)
- Deterministic content hashing and provenance chains
- Computational lineage (input/output identities, transform vs model version)
- Thin adapters for WaveBridge / MetaField / operator ABI boundaries

## What TensorGate is NOT

| System | Role |
|--------|------|
| **WaveBridge** | Physical waveform transport (WAV, optical, serial) |
| **MetaField** | Field / intelligence semantics and learned geometry |
| **metafield-operator-abi** | Mathematical operator contracts (Wilson–Dirac etc.) |
| **field-os** | Observation / state admission and replay |
| **Aurora** | Distributed artifact identity / history / transport |
| **MultiFlow / FreightFlow** | Scheduling and freight coordination (independent) |

```
neural weights
     ↓
  TensorGate
     ↓
normalized tensor
     ↓
  ┌───────┬────────┐
  ↓       ↓        ↓
model   operator WaveBridge
runtime   ABI      physical transport
```

## Install

```bash
pip install -e .
# optional
pip install -e ".[torch]"
pip install -e ".[dev]"
```

Python ≥ 3.10. Core dependency: **numpy**. PyTorch is optional.

## Quick start

```python
import numpy as np
from tensorgate import analyze, normalize, adapt, TensorSpec, compare

w = np.random.randn(128, 64).astype(np.float32)

# Inspect
desc = analyze(w)
print(desc.summary())

# Explicit normalize (always returns metadata)
out, meta = normalize(w, method="symmetric")
print(meta["scale"], meta["provenance"])

# Adapt to a target constraint set
target = TensorSpec(dtype="float16", min_value=-1.0, max_value=1.0)
adapted, report = adapt(w, target)
print(report.transformations)
print(report.error_metrics)

# Round-trip error
err = compare(w, adapted.astype(np.float32))
print(err["max_absolute_error"])
```

## CLI

```bash
tensorgate inspect weights.npy
tensorgate analyze weights.npy
tensorgate normalize weights.npy --method symmetric
tensorgate convert weights.npy --dtype float16
tensorgate compare original.npy reconstructed.npy
tensorgate compatibility weights.npy --target target.json
tensorgate adapt weights.npy --target target.json
```

## Design rules

1. **No silent numerical changes.** Every transformation produces a recorded provenance step (source range, scale, offset, clipping, quantization, precision loss).
2. **Deterministic identity.** Content hash covers canonical bytes + dtype + shape + representation metadata.
3. **Boundary ownership.** TensorGate owns numbers. WaveBridge owns waveforms. MetaField owns fields. Operator ABI owns math. field-os owns admission.
4. **Optional integrations only.** Core never imports MetaField or WaveBridge internals.

## Shared Evidence Contract

Computational lineage (`tensorgate.lineage`) records input/output identities,
transformation or model version, and optional upstream evidence refs.
See `docs/SHARED_EVIDENCE_CONTRACT.md`.

## Repository layout

```
tensorgate/
  __init__.py
  tensor.py          # TensorSpec, BackendTensorSpec, as_numpy
  descriptor.py      # TensorDescriptor + content hash
  inspect.py         # analyze / describe
  normalize.py       # range / maxabs / standard / symmetric / affine
  transform.py       # dtype, clip, transform
  quantize.py        # quantize / dequantize
  layout.py
  serialization.py   # .npy + .tg.json side-car
  provenance.py      # TransformationRecord chain
  lineage.py         # ComputationalLineage (Shared Evidence Contract)
  compatibility.py   # check_compatibility / adapt / compare
  errors.py
  cli.py
  integrations/
    wavebridge.py
    metafield.py
    operator_abi.py
tests/
examples/
docs/
```

## Tests

```bash
pytest -q
```

## License

MIT
