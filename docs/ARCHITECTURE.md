# TensorGate Architecture

## Position in the collective

```
MODEL / OPERATOR / FIELD
         │
         ▼
    TensorGate          ← numerical representation & compatibility
         │
 ┌───────┼────────┐
 ▼       ▼        ▼
runtime  storage  transport
 │       │        │
PyTorch  tensors  WaveBridge
future   artifacts optical / audio / …
TPU/FPGA
```

## Owned concerns

- dtype, shape, rank, layout
- numerical statistics (min/max/mean/std/rms/abs_max)
- finite / NaN / Inf accounting
- sparsity
- scale / zero-point / quantization metadata
- deterministic content hash
- explicit transformation records (provenance)
- compatibility declarations (`TensorSpec`) and adaptation
- round-trip error metrics

## Explicitly not owned

- Wilson–Dirac (or any) operator mathematics → `metafield-operator-abi`
- Field semantics, geometry, ticks → MetaField / field-os
- Physical channel coding, WAV, lasers, PAM → WaveBridge
- Distributed artifact fabric → Aurora
- Scheduling / freight → MultiFlow / FreightFlow

## Dependency direction

```
TensorGate
   ↑  optional integration only
   │
MetaField / WaveBridge / operator tooling
```

No cycles. MultiFlow, FreightFlow, self-state-kernel, and field-os runtime remain independent.
