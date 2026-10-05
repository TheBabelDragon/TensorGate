# Hugging Face / Safetensors Model Inspection

TensorGate can inspect multi-shard Hugging Face model directories that use
the standard safetensors layout. This is the first integration step toward
using TensorGate as a numerical compatibility boundary in front of
Transformers / vLLM — **not** an inference runtime.

## Target layout (Qwen3-4B example)

```
Qwen3-4B/
  config.json
  model.safetensors.index.json
  model-00001-of-00003.safetensors
  model-00002-of-00003.safetensors
  model-00003-of-00003.safetensors
  …
```

`model.safetensors.index.json` contains:

```json
{
  "metadata": { "total_size": 8044936192 },
  "weight_map": {
    "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
    "model.layers.0.…": "model-00001-of-00003.safetensors",
    …
  }
}
```

## Workflow

```bash
# Install optional dependency
pip install "tensorgate[safetensors]"
# or: pip install safetensors

# Human-readable report
tensorgate model inspect ./Qwen3-4B

# Deterministic machine-readable JSON
tensorgate model inspect ./Qwen3-4B --json > manifest.json

# Skip per-tensor NaN/Inf scan (metadata + file hashes only)
tensorgate model inspect ./Qwen3-4B --no-value-check
```

Python API:

```python
from tensorgate import inspect_model

manifest = inspect_model("./Qwen3-4B", model_id="Qwen/Qwen3-4B")
print(manifest.summary())
print(manifest.manifest_hash)

for t in manifest.tensors:
    spec = t.to_tensor_spec()   # TensorGate TensorSpec
    print(t.name, t.shape, t.dtype, t.source_shard)
```

## What is produced

`ModelManifest` contains:

| Field | Description |
|-------|-------------|
| `model_id` | Identifier (directory name or override) |
| `config` | Parsed `config.json` (if present) |
| `shards` | Per-shard filename, path, SHA-256, size, tensor name list |
| `tensors` | Per-tensor name, shape, dtype, element count, source shard, NaN/Inf counts |
| `total_parameters` | Sum of element counts |
| `total_bytes` | Sum of shard file sizes |
| `manifest_hash` | Deterministic SHA-256 over canonical payload |
| `validation_errors` | Missing shards, duplicates, index/shard mismatches, … |
| `validation_warnings` | Non-finite values, extra tensors not in index, … |

## Design guarantees

1. **One tensor at a time** — never loads the full model into RAM.
2. **Deterministic** — two runs on unchanged files produce identical JSON and `manifest_hash`.
3. **Reuses TensorGate types** — each tensor maps to `TensorSpec` via `to_tensor_spec()`.
4. **No inference** — no Transformers, vLLM, quantization execution, or model forward pass.
5. **Explicit validation** — missing/duplicate/corrupt artifacts are reported, never silent.

## Validation coverage

- Missing shards
- Missing tensors (index claims a key that is absent from the mapped shard)
- Duplicate tensors across shards
- Malformed `model.safetensors.index.json`
- Non-finite values (NaN / ±Inf) when value checking is enabled
- Shard content changes (SHA-256 of each shard file)

## Downstream path (future)

```
Qwen safetensors
       ↓
TensorGate inspect_model  →  ModelManifest
       ↓
analyze / compatibility / adapt / compare  (per-tensor)
       ↓
validated model artifact
       ↓
Transformers / vLLM
```

This milestone stops at inspection + deterministic provenance.
