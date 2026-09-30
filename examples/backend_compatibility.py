#!/usr/bin/env python3
"""Backend capability declaration (no fake accelerator implementation)."""

from tensorgate import TensorSpec, BackendTensorSpec, Layout

cpu = BackendTensorSpec(
    name="cpu",
    supported_dtypes=("float32", "float64", "float16", "int8", "int32"),
    supported_layouts=(Layout.C, Layout.F),
    quantization_formats=("none", "symmetric", "asymmetric"),
)

fpga = BackendTensorSpec(
    name="fpga-stub",
    supported_dtypes=("float32", "int8"),
    supported_layouts=(Layout.C,),
    max_rank=4,
    quantization_formats=("symmetric",),
    notes="Capability declaration only — no FPGA runtime in TensorGate.",
)

spec = TensorSpec(dtype="float32", rank=2, quantization="symmetric")
print("CPU accepts:", cpu.accepts(spec))
print("FPGA accepts:", fpga.accepts(spec))

bad = TensorSpec(dtype="float64", rank=6)
print("FPGA accepts float64 rank-6:", fpga.accepts(bad))
