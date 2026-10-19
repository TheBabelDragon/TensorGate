# Verified Model Transformation

**Status:** Implemented on `feat/verified-model-transformation`

TensorGate can apply **explicit, recorded numerical transformations** to selected tensors inside a local Hugging Face / safetensors model directory, then write a new artifact and a machine-readable evidence report.

This is a numerical compatibility and provenance boundary, **not** an inference runtime.

## Design rules

1. Source artifact is never modified.
2. Only explicitly named tensors are transformed; all others are byte-identical passthrough.
3. Every applied operation produces input/output content hashes, shape/dtype, parameters, and (when applicable) numerical error metrics.
4. Output is re-inspected; the report includes a `ComputationalLineage` conforming to `docs/SHARED_EVIDENCE_CONTRACT.md`.
5. Unsupported operations and non-finite values are reported, never silently ignored.
6. Core remains free of mandatory PyTorch / Transformers / MetaField / Aurora dependencies. `safetensors` is optional (same as model inspection).

## Supported operations

| Operation       | Parameters (common)                          | Notes |
|-----------------|----------------------------------------------|-------|
| `identity`      | —                                            | Explicit no-op; still recorded |
| `normalize`     | `method` (symmetric/maxabs/range/standard), `target_min`, `target_max`, `clip_outliers` | Reuses `tensorgate.normalize` |
| `convert_dtype` | `dtype` (required), `casting` (safe/unsafe)  | Reuses `tensorgate.transform.convert_dtype` |
| `clip`          | `min_value`, `max_value`                     | Reuses `tensorgate.transform.clip` |
| `quantize`      | `bits`, `scheme` (symmetric/asymmetric), `signed` | Reuses `tensorgate.quantize` |

Any other operation name is rejected at validation time.

## Non-finite policy (`on_nonfinite`)

| Value       | Behaviour |
|-------------|-----------|
| `reject`    | Skip the tensor (keep original), record rejection (default) |
| `preserve`  | Apply the transform; keep NaN/Inf as-is; emit warning |
| `error`     | Raise `ValidationError` immediately |

TensorGate never invents replacement values for NaN/Inf.

## Python API

```python
from pathlib import Path
from tensorgate import transform_model, inspect_model

report = transform_model(
    source_dir="./MiniModel",
    output_dir="./MiniModel-fp16-embed",
    transformations={
        "model.embed.weight": {
            "operation": "normalize",
            "method": "symmetric",
        },
        "lm_head.weight": {
            "operation": "convert_dtype",
            "dtype": "float16",
        },
    },
    on_nonfinite="reject",
)

print(report.summary())
print(report.source_manifest_hash)
print(report.output_manifest_hash)
print(report.lineage.transformation_id)  # "model_transform"

# Output is a normal HF-style directory
m = inspect_model("./MiniModel-fp16-embed")
assert m.manifest_hash == report.output_manifest_hash
```

The report is also written to:

```
<output_dir>/tensorgate_transform_report.json
```

Deterministic fields (hashes, shapes, dtypes, lineage) exclude filesystem paths and timestamps.

## CLI

```bash
tensorgate model transform ./MiniModel ./MiniModel-out \
  --transform model.embed.weight=normalize:method=symmetric \
  --transform lm_head.weight=convert_dtype:dtype=float16 \
  --on-nonfinite reject

# Machine-readable report on stdout
tensorgate model transform ./MiniModel ./MiniModel-out \
  --transform model.layers.0.weight=identity \
  --json
```

## Artifact layout (output)

```
output_dir/
  model.safetensors.index.json   # rewritten weight_map
  model-0000N-of-0000M.safetensors
  config.json                    # copied from source when present
  tensorgate_transform_report.json
```

Shard filenames are preserved so existing loaders that rely on the index continue to work.

## Memory behaviour

- One tensor is loaded at a time via `safetensors.safe_open` / `get_tensor`.
- The full model is never materialised in RAM.
- Peak memory ≈ largest single tensor + small constant overhead.

## Limitations

- One primary operation per named tensor per call (compose by chaining transform_model calls if needed).
- Shape-changing transforms are not supported; shape/rank mismatches surface as validation issues.
- Quantized integer outputs are stored as native NumPy integer dtypes inside safetensors; consumers must understand the scale/zero-point from the report if they need dequantization.
- No guarantee of behavioural equivalence after transformation — only structural validity and recorded numerical error metrics.
- Multi-process / concurrent writers to the same `output_dir` are not coordinated.

## Verification

```bash
pytest tests/test_model_transform.py -q
pytest -q   # full suite; existing tests must remain green
```
