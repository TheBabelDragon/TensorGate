# Shared Evidence Contract — TensorGate (Phase 4)

**Status:** Implemented — computational lineage for numerical boundary crossings.

## Ownership

| Layer | Owns |
|-------|------|
| CANtheon / CANgate | physical transport evidence envelope |
| MetaField | field semantics + optional `EvidenceLineage` |
| **TensorGate** | numerical identity, transforms, computational lineage |
| MultiFlow / Aurora | decisions / distributed integrity (later) |

## What TensorGate records

- **Input identities** — content hash, shape, dtype, optional schema
- **Transformation / model version** — operation id + `tensorgate/x.y.z` or model id/version
- **Output identity** — content hash of result
- **Kind** — `DETERMINISTIC` (replayable) vs `NONDETERMINISTIC` (inference)
- **Provenance chain** — existing `TransformationRecord` list
- **Evidence refs** — optional upstream CANtheon/MetaField event ids (plain dicts)

## API

```python
from tensorgate import analyze, normalize, build_lineage, ComputationKind, attach_evidence_ref

desc = analyze(weights)
out, meta = normalize(weights, method="symmetric")
out_desc = analyze(out)

lin = build_lineage(
    operation="normalize/symmetric",
    input_hashes=[desc.content_hash],
    output_hash=out_desc.content_hash,
    kind=ComputationKind.DETERMINISTIC,
)
lin = attach_evidence_ref(lin, event_id="...", source_id="temp_sensor_01")
```

## Rules

1. No silent numerical changes (existing TensorGate rule).
2. DETERMINISTIC lineage must not set `model_id`.
3. Missing input/output hashes fail closed.
4. Lineage schema major mismatch raises `CompatibilityError`.
5. Core does not import MetaField or CANtheon.

## Verification

```bash
pytest tests/test_lineage.py -q
```
