# Neuralese Representation Contract

Versioned, validated contract for **representing, identifying, comparing,
serializing, and tracking provenance** of latent-state tensors exchanged
between compatible neural-network components.

Implemented in `tensorgate.latent` on top of existing TensorGate identity,
descriptor, compatibility, and lineage infrastructure.

## Scope

**In scope**

- Immutable `LatentStateSpec` (schema, producer, location, numerical contract)
- Deterministic latent identity derived from tensor content + identity metadata
- Serialization / deserialization with schema versioning
- Compatibility levels including **inconclusive** when metadata is missing
- Numerical comparison without silent reshape / normalize / project
- Computational lineage hooks (deterministic transforms vs nondeterministic generation)

**Out of scope (explicit non-goals)**

- Model loading, inference, token generation
- Cross-model latent alignment or a universal "latent language"
- Claiming that matching shape/dtype implies semantic interchangeability
- Quantum simulation, agents, or Neuralese runtimes

This contract does **not** standardize the *meaning* of hidden states across
arbitrary models, layers, or architectures.

## Schema

| Field | Identity-relevant | Required | Description |
|-------|-------------------|----------|-------------|
| `schema_id` | yes | yes | `tensorgate.latent_state` |
| `schema_version` | yes | yes | Semver; major must be supported (currently `1`) |
| `shape` | yes | yes | Tensor shape |
| `dtype` | yes | yes | NumPy dtype string |
| `layout` | yes | yes | `C`, `F`, or `unknown` |
| `representation_semantics` | yes | for EXACT | Free-form semantic label (e.g. `hidden_state`) |
| `model_id` | yes | for EXACT | Producer model identifier |
| `model_revision` | yes | no | Weight/revision or manifest hash |
| `architecture_id` | yes | no | Architecture family identifier |
| `layer_id` | yes | for EXACT | Layer/block identifier |
| `sequence_position` | yes | no | Timestep / sequence index |
| `runtime_id` | no | no | Optional runtime tag |
| `mask_id` / `reference_id` | no | no | Optional interpretation aids |
| `extras` | no | no | Non-identity annotations only |

Missing optional fields are **not** treated as verified compatibility.

## Identity and serialization

Latent identity reuses TensorGate's SHA-256 content hash over little-endian
array bytes, augmented with identity-relevant latent metadata (schema, model,
layer, semantics, layout, …).

```python
import numpy as np
from tensorgate import (
    LatentStateSpec,
    describe_latent_state,
    serialize_latent_state,
    deserialize_latent_state,
    check_latent_compatibility,
)

arr = np.zeros((2, 16), dtype=np.float32)
spec = LatentStateSpec(
    shape=(2, 16),
    dtype="float32",
    representation_semantics="hidden_state",
    model_id="toy-mlp",
    model_revision="sha256:abc",
    architecture_id="mlp-v1",
    layer_id="block.0",
)

desc = describe_latent_state(arr, spec)
envelope = serialize_latent_state(arr, spec)
arr2, spec2, ident = deserialize_latent_state(envelope, require_data=True)
assert ident.content_hash == envelope["latent_id"]
```

Deserialization is **fail-closed**: malformed JSON, unsupported schema major
versions, shape/dtype inconsistency, and declared/recomputed identity mismatch
raise `SerializationError` / `ValidationError`.

Filesystem paths and timestamps must not be placed in identity fields.

## Compatibility levels

| Level | Meaning |
|-------|---------|
| `exact` | Schema, shape, dtype/layout, producer, layer, and semantics align |
| `numerical_convertible` | Shape matches; dtype/layout differ but conversion is explicitly allowed |
| `incompatible` | Hard mismatch (shape, schema major, model/layer, …) |
| `inconclusive` | Required producer/layer/semantics metadata missing |

Matching shape and dtype **alone never** yield `exact` compatibility across
different models or layers.

```python
from tensorgate import LatentStateSpec, check_latent_compatibility

a = LatentStateSpec(
    shape=(8,), dtype="float32", representation_semantics="h",
    model_id="A", layer_id="enc.0",
)
b = LatentStateSpec(
    shape=(8,), dtype="float32", representation_semantics="h",
    model_id="B", layer_id="enc.0",
)
report = check_latent_compatibility(a, b)
assert report.compatible is False  # different models
```

## Comparison

`compare_latent_states` reports numerical metrics (max abs diff, RMSE, cosine)
without transforming either array. Shape mismatches are reported; arrays are
not silently aligned.

## Lineage

Use `build_latent_lineage` to record boundary crossings. **Default kind for
model hidden-state generation is `NONDETERMINISTIC`**: the ability to hash an
output tensor does not make the generating computation reproducible.

Deterministic dtype/layout conversions may use `ComputationKind.DETERMINISTIC`
and must not set `model_id`.

## Model-specific limitations

- Latent geometry is model- and layer-specific; this contract only records
  declared metadata.
- No cross-model mapping is provided or implied.
- Float16/complex support follows NumPy; quantization of latents is out of
  scope here (use TensorGate quantize APIs explicitly if needed).

## Consuming from a future Neuralese runtime

A research runtime can depend on this contract by:

1. Emitting `LatentStateSpec` + serialized envelopes at layer boundaries.
2. Checking `check_latent_compatibility` before reuse.
3. Attaching `build_latent_lineage` records to evidence bundles.

The runtime should treat TensorGate as a **numerical boundary library**, not
as an inference engine. TensorGate must not grow model-execution features to
satisfy Neuralese experiments.

## Related docs

- [NUMERICAL_CONTRACT.md](NUMERICAL_CONTRACT.md)
- [COMPATIBILITY.md](COMPATIBILITY.md)
- [PROVENANCE.md](PROVENANCE.md)
- [SHARED_EVIDENCE_CONTRACT.md](SHARED_EVIDENCE_CONTRACT.md)
