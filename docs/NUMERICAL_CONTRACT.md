# Numerical Contract

## Core rule

**TensorGate must never silently alter numerical meaning.**

Example: the vector `[-0.8, 0.2, 0.7]` must not become `[-1.0, 0.25, 0.875]` without a recorded transformation that states:

- source range
- target range
- scale
- offset
- clipping (if any)
- quantization parameters
- resulting error metrics

## Transformation record

Every mutating operation returns metadata containing at least:

```json
{
  "operation": "normalize/symmetric",
  "parameters": { "scale": 1.25, "offset": 0.0, ... },
  "input_hash": "...",
  "output_hash": "...",
  "software_id": "tensorgate/0.1.0"
}
```

## Content identity

Hash covers:

1. canonical contiguous little-endian bytes of the array
2. dtype string
3. shape
4. scale / zero_point / quantization / layout when present

Two tensors that differ in any of the above receive different hashes.
