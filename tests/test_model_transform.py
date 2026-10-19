"""Tests for verified model-artifact transformation.

Covers successful deterministic transforms, source immutability,
passthrough of untouched tensors, failure modes, lineage integrity,
and numerical error reporting.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("safetensors")

from safetensors.numpy import save_file, load_file

from tensorgate import (
    inspect_model,
    transform_model,
    TensorTransformSpec,
    ModelTransformReport,
    ComputationKind,
    SUPPORTED_OPS,
)
from tensorgate.errors import SerializationError, ValidationError
from tensorgate.model_transform import transform_model_report_path


# ---------------------------------------------------------------------------
# Fixture helpers (mirrors test_model_safetensors patterns)
# ---------------------------------------------------------------------------

def _write_shard(path: Path, tensors: dict) -> None:
    save_file(tensors, str(path))


def _make_mini_model(tmp: Path, *, with_nan: bool = False) -> Path:
    model_dir = tmp / "MiniModel"
    model_dir.mkdir()

    t1 = {
        "model.embed.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        "model.layers.0.weight": np.ones((2, 2), dtype=np.float32),
    }
    t2 = {
        "model.layers.1.weight": np.full((2, 3), 0.5, dtype=np.float32),
        "lm_head.weight": np.eye(3, dtype=np.float32),
    }
    if with_nan:
        t2["model.layers.1.weight"] = t2["model.layers.1.weight"].copy()
        t2["model.layers.1.weight"][0, 0] = np.nan

    _write_shard(model_dir / "model-00001-of-00002.safetensors", t1)
    _write_shard(model_dir / "model-00002-of-00002.safetensors", t2)

    weight_map = {
        "model.embed.weight": "model-00001-of-00002.safetensors",
        "model.layers.0.weight": "model-00001-of-00002.safetensors",
        "model.layers.1.weight": "model-00002-of-00002.safetensors",
        "lm_head.weight": "model-00002-of-00002.safetensors",
    }
    index = {"metadata": {"total_size": 1000}, "weight_map": weight_map}
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "mini", "architectures": ["MiniForCausalLM"]}),
        encoding="utf-8",
    )
    return model_dir


def _shard_sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Successful deterministic transformation
# ---------------------------------------------------------------------------

def test_transform_normalize_deterministic(tmp_path):
    src = _make_mini_model(tmp_path)
    out = tmp_path / "out_norm"

    report1 = transform_model(
        src,
        out,
        {"model.embed.weight": {"operation": "normalize", "method": "symmetric"}},
    )
    assert isinstance(report1, ModelTransformReport)
    assert report1.source_manifest_hash
    assert report1.output_manifest_hash
    assert report1.source_manifest_hash != report1.output_manifest_hash

    # Find the transformed tensor
    applied = [t for t in report1.tensors if t.name == "model.embed.weight"]
    assert len(applied) == 1
    t = applied[0]
    assert t.applied is True
    assert t.operation == "normalize"
    assert t.input_hash != t.output_hash
    assert t.error_metrics  # should have numerical metrics

    # Untouched tensors keep identity
    for name in ("model.layers.0.weight", "model.layers.1.weight", "lm_head.weight"):
        r = next(x for x in report1.tensors if x.name == name)
        assert r.applied is False
        assert r.operation == "passthrough"
        assert r.input_hash == r.output_hash

    # Re-run into a fresh dir → identical report identity fields
    out2 = tmp_path / "out_norm2"
    report2 = transform_model(
        src,
        out2,
        {"model.embed.weight": {"operation": "normalize", "method": "symmetric"}},
    )
    assert report1.source_manifest_hash == report2.source_manifest_hash
    assert report1.output_manifest_hash == report2.output_manifest_hash
    h1 = {t.name: t.output_hash for t in report1.tensors}
    h2 = {t.name: t.output_hash for t in report2.tensors}
    assert h1 == h2


def test_source_immutability(tmp_path):
    src = _make_mini_model(tmp_path)
    # Snapshot source shard hashes
    before = {
        p.name: _shard_sha(p)
        for p in src.glob("*.safetensors")
    }
    before_index = (src / "model.safetensors.index.json").read_text()
    before_cfg = (src / "config.json").read_text()

    out = tmp_path / "out_immut"
    transform_model(
        src,
        out,
        {"lm_head.weight": {"operation": "convert_dtype", "dtype": "float16"}},
    )

    after = {
        p.name: _shard_sha(p)
        for p in src.glob("*.safetensors")
    }
    assert before == after
    assert (src / "model.safetensors.index.json").read_text() == before_index
    assert (src / "config.json").read_text() == before_cfg


def test_passthrough_preserves_bytes(tmp_path):
    src = _make_mini_model(tmp_path)
    out = tmp_path / "out_pass"
    transform_model(
        src,
        out,
        {"model.embed.weight": {"operation": "identity"}},
    )
    # Load both and compare untouched tensor
    src_t = load_file(str(src / "model-00002-of-00002.safetensors"))
    out_t = load_file(str(out / "model-00002-of-00002.safetensors"))
    np.testing.assert_array_equal(src_t["lm_head.weight"], out_t["lm_head.weight"])


def test_output_reinspection(tmp_path):
    src = _make_mini_model(tmp_path)
    out = tmp_path / "out_re"
    report = transform_model(
        src,
        out,
        {"model.layers.0.weight": {"operation": "normalize", "method": "maxabs"}},
    )
    # Fresh inspect must succeed and match report hash (same model_id)
    m = inspect_model(out, model_id=report.output_model_id)
    assert m.manifest_hash == report.output_manifest_hash
    assert len(m.tensors) == 4
    assert (out / "config.json").is_file()
    assert (out / "model.safetensors.index.json").is_file()
    assert transform_model_report_path(out).is_file()


def test_lineage_integrity(tmp_path):
    src = _make_mini_model(tmp_path)
    out = tmp_path / "out_lin"
    report = transform_model(
        src,
        out,
        {
            "model.embed.weight": {"operation": "normalize", "method": "symmetric"},
            "lm_head.weight": {"operation": "convert_dtype", "dtype": "float16"},
        },
    )
    lin = report.lineage
    assert lin.kind == ComputationKind.DETERMINISTIC
    assert lin.transformation_id == "model_transform"
    assert lin.model_id == ""  # DETERMINISTIC must not set model_id
    assert lin.output is not None
    assert lin.output.content_hash == report.output_manifest_hash
    assert lin.inputs
    # Round-trip via to_dict / from_dict
    from tensorgate.lineage import ComputationalLineage
    restored = ComputationalLineage.from_dict(lin.to_dict())
    assert restored.transformation_id == lin.transformation_id
    assert restored.output.content_hash == lin.output.content_hash


def test_quantize_and_error_metrics(tmp_path):
    src = _make_mini_model(tmp_path)
    out = tmp_path / "out_q"
    report = transform_model(
        src,
        out,
        {"model.embed.weight": {"operation": "quantize", "bits": 8, "scheme": "symmetric"}},
    )
    t = next(x for x in report.tensors if x.name == "model.embed.weight")
    assert t.applied
    assert t.output_dtype in ("int8", "uint8")
    assert "max_absolute_error" in t.error_metrics or t.error_metrics == {}


# ---------------------------------------------------------------------------
# Failure / safety paths
# ---------------------------------------------------------------------------

def test_unknown_tensor_raises(tmp_path):
    src = _make_mini_model(tmp_path)
    out = tmp_path / "out_unk"
    with pytest.raises(ValidationError, match="Unknown tensor"):
        transform_model(
            src,
            out,
            {"does.not.exist": {"operation": "normalize"}},
            raise_on_error=True,
        )


def test_unsupported_op_rejected(tmp_path):
    src = _make_mini_model(tmp_path)
    out = tmp_path / "out_badop"
    with pytest.raises(ValidationError, match="Unsupported transformation"):
        transform_model(
            src,
            out,
            {"model.embed.weight": {"operation": "frobnicate"}},
        )


def test_refuse_overwrite_source(tmp_path):
    src = _make_mini_model(tmp_path)
    with pytest.raises(SerializationError, match="distinct from source"):
        transform_model(
            src,
            src,
            {"model.embed.weight": {"operation": "identity"}},
        )


def test_refuse_nonempty_output(tmp_path):
    src = _make_mini_model(tmp_path)
    out = tmp_path / "out_ne"
    out.mkdir()
    (out / "dummy.txt").write_text("x")
    with pytest.raises(SerializationError, match="not empty"):
        transform_model(
            src,
            out,
            {"model.embed.weight": "identity"},
            overwrite_output=False,
        )


def test_nonfinite_reject_policy(tmp_path):
    src = _make_mini_model(tmp_path, with_nan=True)
    out = tmp_path / "out_nan"
    report = transform_model(
        src,
        out,
        {"model.layers.1.weight": {"operation": "normalize", "method": "symmetric"}},
        on_nonfinite="reject",
        raise_on_error=False,
    )
    t = next(x for x in report.tensors if x.name == "model.layers.1.weight")
    assert t.rejected is True
    assert t.applied is False
    assert "model.layers.1.weight" in report.rejected_operations
    # Original value preserved in output
    src_t = load_file(str(src / "model-00002-of-00002.safetensors"))
    out_t = load_file(str(out / "model-00002-of-00002.safetensors"))
    assert np.isnan(out_t["model.layers.1.weight"][0, 0])


def test_nonfinite_error_policy(tmp_path):
    src = _make_mini_model(tmp_path, with_nan=True)
    out = tmp_path / "out_nan_err"
    with pytest.raises(ValidationError, match="non-finite"):
        transform_model(
            src,
            out,
            {"model.layers.1.weight": {"operation": "normalize"}},
            on_nonfinite="error",
        )


def test_missing_source_dir(tmp_path):
    with pytest.raises(SerializationError, match="does not exist"):
        transform_model(
            tmp_path / "nope",
            tmp_path / "out",
            {"x": "identity"},
        )


def test_empty_transformations(tmp_path):
    src = _make_mini_model(tmp_path)
    with pytest.raises(ValidationError, match="must not be empty"):
        transform_model(src, tmp_path / "out", {})


def test_convert_dtype_requires_param(tmp_path):
    src = _make_mini_model(tmp_path)
    out = tmp_path / "out_cd"
    report = transform_model(
        src,
        out,
        {"model.embed.weight": {"operation": "convert_dtype"}},  # missing dtype
        raise_on_error=False,
    )
    t = next(x for x in report.tensors if x.name == "model.embed.weight")
    assert t.rejected is True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_model_transform(tmp_path, capsys):
    from tensorgate.cli import main

    src = _make_mini_model(tmp_path)
    out = tmp_path / "cli_out"
    main([
        "model", "transform",
        str(src), str(out),
        "--transform", "model.embed.weight=normalize:method=symmetric",
        "--transform", "lm_head.weight=convert_dtype:dtype=float16",
    ])
    captured = capsys.readouterr()
    assert "source_model_id=MiniModel" in captured.out
    assert (out / "tensorgate_transform_report.json").is_file()
    m = inspect_model(out)
    assert len(m.tensors) == 4


def test_cli_model_transform_json(tmp_path, capsys):
    from tensorgate.cli import main

    src = _make_mini_model(tmp_path)
    out = tmp_path / "cli_json"
    main([
        "model", "transform",
        str(src), str(out),
        "--transform", "model.layers.0.weight=identity",
        "--json",
    ])
    data = json.loads(capsys.readouterr().out)
    assert "source_manifest_hash" in data
    assert "lineage" in data
    assert data["lineage"]["kind"] == "DETERMINISTIC"


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------

def test_tensor_transform_spec_validation():
    s = TensorTransformSpec(operation="normalize", parameters={"method": "symmetric"})
    assert s.operation == "normalize"
    with pytest.raises(ValidationError):
        TensorTransformSpec(operation="not_a_real_op")


def test_supported_ops_frozen():
    assert "normalize" in SUPPORTED_OPS
    assert "quantize" in SUPPORTED_OPS
    assert "frobnicate" not in SUPPORTED_OPS
