"""Minimal reproducible example: verified model transformation.

Creates a tiny synthetic HF-style model, applies normalize + dtype convert
to selected tensors, writes a new artifact, and prints the evidence report.

Run from repo root:

  PYTHONPATH=. python examples/verified_model_transform.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from tensorgate import transform_model, inspect_model


def _make_toy_model(root: Path) -> Path:
    model_dir = root / "ToyModel"
    model_dir.mkdir()
    t1 = {
        "embed.weight": np.linspace(-1, 1, 24, dtype=np.float32).reshape(4, 6),
        "layer.0.weight": np.eye(4, dtype=np.float32),
    }
    t2 = {
        "lm_head.weight": np.arange(16, dtype=np.float32).reshape(4, 4),
    }
    save_file(t1, str(model_dir / "model-00001-of-00002.safetensors"))
    save_file(t2, str(model_dir / "model-00002-of-00002.safetensors"))
    weight_map = {
        "embed.weight": "model-00001-of-00002.safetensors",
        "layer.0.weight": "model-00001-of-00002.safetensors",
        "lm_head.weight": "model-00002-of-00002.safetensors",
    }
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=2),
        encoding="utf-8",
    )
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "toy"}), encoding="utf-8"
    )
    return model_dir


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = _make_toy_model(tmp)
        out = tmp / "ToyModel-transformed"

        report = transform_model(
            src,
            out,
            transformations={
                "embed.weight": {
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
        print()
        print("Per-tensor results:")
        for t in report.tensors:
            status = "APPLIED" if t.applied else ("REJECTED" if t.rejected else "passthrough")
            print(
                f"  {t.name}: {status}  "
                f"{t.input_dtype}{list(t.input_shape)} → {t.output_dtype}{list(t.output_shape)}  "
                f"hash {t.input_hash[:12]}… → {t.output_hash[:12]}…"
            )

        # Confirm output is loadable
        m = inspect_model(out)
        assert m.manifest_hash == report.output_manifest_hash
        print()
        print(f"Output re-inspection OK  manifest_hash={m.manifest_hash[:16]}…")
        print(f"Lineage kind={report.lineage.kind.value}  id={report.lineage.transformation_id}")


if __name__ == "__main__":
    main()
