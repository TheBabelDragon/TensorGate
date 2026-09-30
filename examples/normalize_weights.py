#!/usr/bin/env python3
"""Flagship example: inspect → normalize → adapt → error report."""

import numpy as np
from tensorgate import analyze, normalize, adapt, compare, TensorSpec

rng = np.random.default_rng(0)
weights = rng.normal(0.0, 0.4, size=(256, 128)).astype(np.float32)
weights[0, 0] = 3.5  # outlier

print("=== Source ===")
print(analyze(weights).summary())

print("\n=== Normalize (symmetric) ===")
normed, meta = normalize(weights, method="symmetric")
print(f"scale={meta['scale']:.6f}  clipped={meta.get('clipped', 0)}")
print(analyze(normed).summary())

print("\n=== Adapt to float16 + range [-1, 1] ===")
target = TensorSpec(dtype="float16", min_value=-1.0, max_value=1.0)
adapted, report = adapt(weights, target)
print("transformations:", report.transformations)
print("issues:", report.issues)
print("error_metrics:", report.error_metrics)

print("\n=== Round-trip compare (float32 view) ===")
err = compare(weights, adapted.astype(np.float32))
for k, v in err.items():
    print(f"  {k}: {v}")
