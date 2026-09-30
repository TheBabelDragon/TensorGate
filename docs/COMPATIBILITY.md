# Compatibility

## TensorSpec

A target declares requirements:

- dtype
- shape / rank
- allowed value range
- layout
- quantization scheme
- scale / zero_point
- signed/unsigned
- NaN / Inf policy
- alignment / precision

## API

```python
report = check_compatibility(source, target)   # non-mutating
adapted, report = adapt(tensor, target)        # mutating, fully recorded
```

`adapt()` produces:

1. the adapted array
2. a `CompatibilityReport` with issues, transformation list, provenance chain, and error metrics

## BackendTensorSpec

Future accelerators declare capabilities (supported dtypes, ranks, quantization formats, alignment). Compatibility can be evaluated before execution. No fake backends are implemented in v0.1.
