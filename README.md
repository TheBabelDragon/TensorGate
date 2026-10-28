# TensorGate

**TensorGate is where tensors go when they need to cross a boundary.**

Canonical numerical compatibility boundary for tensors and neural-network weights across the BabelDragon compute stack.

## What TensorGate is

- Numerical representation and compatibility layer
- Explicit, inspectable transformations (never silent)
- Deterministic content hashing and provenance chains
- Computational lineage (input/output identities, transform vs model version)
- Thin adapters for WaveBridge / MetaField / operator ABI boundaries
- **Latent-state representation contract** for Neuralese-compatible boundary exchange

## What TensorGate is NOT

| System | Role |
|--------|------|
| **WaveBridge** | Physical waveform transport |
| **MetaField** | Field / intelligence semantics |
| **field-os** | Observation / state admission |
| **Neuralese runtime** | Inference / latent recurrence (out of scope) |

## Install

```bash
pip install -e .
pip install -e ".[dev]"
```

Python ≥ 3.10. Core dependency: **numpy**. PyTorch is optional.

## Quick start

```python
import numpy as np
from tensorgate import analyze, normalize, adapt, TensorSpec, compare

w = np.random.randn(128, 64).astype(np.float32)
desc = analyze(w)
out, meta = normalize(w, method="symmetric")
```

## Latent-state representation (Neuralese contract)

TensorGate provides a **versioned latent-state representation contract** for
identifying, comparing, and serializing neural latent tensors at numerical
boundaries — without becoming an inference runtime.

Matching shape and dtype alone do **not** imply semantic compatibility across
models or layers.

See [docs/NEURALESE_REPRESENTATION_CONTRACT.md](docs/NEURALESE_REPRESENTATION_CONTRACT.md).

```python
import numpy as np
from tensorgate import LatentStateSpec, describe_latent_state, check_latent_compatibility

x = np.zeros((4, 32), dtype=np.float32)
spec = LatentStateSpec(
    shape=(4, 32), dtype="float32",
    representation_semantics="hidden_state",
    model_id="demo", layer_id="block.0",
)
print(describe_latent_state(x, spec).latent_id[:16])
```

## Shared Evidence Contract

Computational lineage (`tensorgate.lineage`) records input/output identities,
transformation or model version, and optional upstream evidence refs.
See `docs/SHARED_EVIDENCE_CONTRACT.md`.

## Tests

```bash
pytest -q
```

## License

MIT
