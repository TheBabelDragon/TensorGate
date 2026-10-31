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

## Compatibility levels

| Level | Meaning |
|-------|---------|
| `exact` | Schema, shape, dtype/layout, producer, layer, and semantics align |
| `numerical_convertible` | Shape matches; dtype differs but conversion is explicitly allowed; other checks pass |
| `incompatible` | Hard mismatch (shape, schema major, model/layer, **layout**, …) |
| `inconclusive` | Required producer/layer/semantics metadata missing |

Matching shape and dtype **alone never** yield `exact` compatibility across
different models or layers.

### Compatibility precedence

1. **Hard incompatibilities** (schema, shape, model, layer, layout, …) → `INCOMPATIBLE`
2. **Missing required metadata** → `INCONCLUSIVE` (even if dtype conversion is allowed)
3. **Dtype differs + conversion allowed + complete metadata** → `NUMERICAL_CONVERTIBLE`
4. Otherwise → `EXACT`

`allow_dtype_conversion=True` permits **dtype** mismatch only. It does **not**
permit layout mismatch and does **not** prove semantic equivalence.

## Comparison conventions (hardened)

### Complex arrays

Complex inputs are promoted to `complex128` and compared in complex arithmetic.
Imaginary components are **never** discarded via real float casting.

| Metric | Definition |
|--------|------------|
| `max_abs_diff` | max \|aᵢ − bᵢ\| (complex magnitude) |
| `rmse` | √(mean(\|aᵢ − bᵢ\|²)) |
| `cosine` | Hermitian: Re(⟨a,b⟩) / (\|a\|\|b\|); `None` if either norm is zero |

### Non-finite values

- All non-finite → status `no_finite_values`
- Mixed finite / non-finite → status `partial_nonfinite` (metrics over finite subset only)
- Full finite → status `compared`

### Tolerances

`rtol` and `atol` must be finite and non-negative; otherwise `ValidationError`.

## Identity and serialization

Latent identity reuses TensorGate's SHA-256 content hash over little-endian
array bytes, augmented with identity-relevant latent metadata.

Deserialization is **fail-closed** on malformed data, unsupported schema major
versions, and identity mismatches.

## Lineage

Default kind for model hidden-state generation is `NONDETERMINISTIC`.
Hashing an output tensor does not make the generating computation reproducible.

## Related docs

- [NUMERICAL_CONTRACT.md](NUMERICAL_CONTRACT.md)
- [COMPATIBILITY.md](COMPATIBILITY.md)
- [PROVENANCE.md](PROVENANCE.md)
- [SHARED_EVIDENCE_CONTRACT.md](SHARED_EVIDENCE_CONTRACT.md)
