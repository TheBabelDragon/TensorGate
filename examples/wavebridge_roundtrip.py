#!/usr/bin/env python3
"""Numerical payload hand-off for WaveBridge (no physical channel)."""

import numpy as np
from tensorgate.integrations.wavebridge import to_wavebridge_payload, from_wavebridge_payload
from tensorgate import compare

rng = np.random.default_rng(2)
w = rng.standard_normal(1024).astype(np.float32)

payload, meta = to_wavebridge_payload(w, method="symmetric", dtype="float32")
print("Payload range:", payload.min(), payload.max())
print("Note:", meta["note"])

recovered, rmeta = from_wavebridge_payload(payload, source_meta=meta)
err = compare(payload, recovered)
print("recovery max_abs_error=", err["max_absolute_error"])
