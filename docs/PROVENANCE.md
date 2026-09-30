# Provenance

Every transformation is a link in a chain:

```
original
   ↓ normalize/symmetric
normalized
   ↓ quantize/symmetric/8bit
quantized
   ↓ convert_dtype → float16
exported
```

Each `TransformationRecord` stores:

- operation name
- parameters (scale, offset, bits, …)
- input content hash
- output content hash
- software/version identifier
- informational timestamp (never used as identity)

Provenance survives serialization via the `.tg.json` side-car.
