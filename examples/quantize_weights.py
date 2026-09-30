#!/usr/bin/env python3
"""Quantize → dequantize round-trip with error metrics."""

import numpy as np
from tensorgate import quantize, dequantize, compare, analyze

rng = np.random.default_rng(1)
w = rng.uniform(-1.5, 1.5, size=(64, 64)).astype(np.float32)

q, qmeta = quantize(w, bits=8, scheme="symmetric")
print("Quantized:", analyze(q).summary())
print("scale=", qmeta["scale"], "zero_point=", qmeta["zero_point"])

recon, _ = dequantize(q, scale=qmeta["scale"], zero_point=qmeta["zero_point"])
err = compare(w, recon)
print("max_abs_error=", err["max_absolute_error"])
print("sqnr_db=", err["sqnr_db"])
