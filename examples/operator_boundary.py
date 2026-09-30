#!/usr/bin/env python3
"""Numerical boundary around an operator invocation (no Wilson–Dirac math)."""

import numpy as np
from tensorgate.integrations.operator_abi import validate_inputs, validate_output
from tensorgate import TensorSpec

psi = np.random.randn(16).astype(np.float64)
U = np.random.randn(16, 16).astype(np.float64)

print(validate_inputs(psi, U, specs=[
    TensorSpec(dtype="float64", rank=1),
    TensorSpec(dtype="float64", rank=2),
]))

out = psi * 0.5
print(validate_output(out, expected=TensorSpec(dtype="float64", rank=1)))
