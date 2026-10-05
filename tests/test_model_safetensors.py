"""Tests for Hugging Face / safetensors multi-shard model inspection.

Uses only small synthetic fixtures. Does NOT download real Qwen weights
in the normal test suite.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("safetensors")

from safetensors.numpy import save_file

from tensorgate import inspect_model, ModelManifest, TensorMeta, TensorSpec
from tensorgate.errors import SerializationError, ValidationError
from tensorgate.model import _manifest_hash


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _write_shard(path: Path, tensors: dict) -> None:
    save_file(tensors, str(path))


def _make_mini_model(tmp: Path, *, with_nan: bool = False, missing_shard: bool = False) -> Path:
    """Create a tiny 2-shard HF-style model directory."""
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
    if not missing_shard:
        _write_shard(model_dir / "model-00002-of-00002.safetensors", t2)

    weight_map = {
        "model.embed.weight": "model-00001-of-00002.safetensors",
        "model.layers.0.weight": "model-00001-of-00002.safetensors",
        "model.layers.1.weight": "model-00002-of-00002.safetensors",
        "lm_head.weight": "model-00002-of-00002.safetensors",
    }
    index = {
        "metadata": {"total_size": 1000},
        "weight_map": weight_map,
    }
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "mini", "architectures": ["MiniForCausalLM"], "hidden_size": 4}),
        encoding="utf-8",
    )
    return model_dir


# ---------------------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------------------

def test_inspect_basic(tmp_path):
    model_dir = _make_mini_model(tmp_path)
    m = inspect_model(model_dir)
    assert isinstance(m, ModelManifest)
    assert m.model_id == "MiniModel"
    assert len(m.shards) == 2
    assert len(m.tensors) == 4
    assert m.total_parameters == 12 + 4 + 6 + 9  # 3*4 + 2*2 + 2*3 + 3*3
    assert m.manifest_hash
    assert m.config.get("model_type") == "mini"
    names = {t.name for t in m.tensors}
    assert "model.embed.weight" in names
    assert "lm_head.weight" in names
    # TensorSpec compatibility
    for t in m.tensors:
        spec = t.to_tensor_spec()
        assert isinstance(spec, TensorSpec)
        assert spec.dtype == t.dtype
        assert spec.shape == t.shape


def test_manifest_deterministic(tmp_path):
    model_dir = _make_mini_model(tmp_path)
    m1 = inspect_model(model_dir)
    m2 = inspect_model(model_dir)
    assert m1.manifest_hash == m2.manifest_hash
    d1 = json.dumps(m1.to_dict(), sort_keys=True, separators=(",", ":"))
    d2 = json.dumps(m2.to_dict(), sort_keys=True, separators=(",", ":"))
    assert d1 == d2


def test_shard_hashes_stable(tmp_path):
    model_dir = _make_mini_model(tmp_path)
    m = inspect_model(model_dir)
    assert all(len(s.sha256) == 64 for s in m.shards)
    # Re-inspect should produce identical shard hashes
    m2 = inspect_model(model_dir)
    h1 = {s.filename: s.sha256 for s in m.shards}
    h2 = {s.filename: s.sha256 for s in m2.shards}
    assert h1 == h2


def test_missing_shard(tmp_path):
    model_dir = _make_mini_model(tmp_path, missing_shard=True)
    m = inspect_model(model_dir)
    assert any("Missing shard" in e for e in m.validation_errors)
    # Only tensors from the present shard
    assert len(m.tensors) == 2
    assert all(t.source_shard.startswith("model-00001") for t in m.tensors)


def test_missing_shard_raises(tmp_path):
    model_dir = _make_mini_model(tmp_path, missing_shard=True)
    with pytest.raises(ValidationError, match="Missing shard"):
        inspect_model(model_dir, raise_on_error=True)


def test_nan_detection(tmp_path):
    model_dir = _make_mini_model(tmp_path, with_nan=True)
    m = inspect_model(model_dir, check_values=True)
    nan_tensors = [t for t in m.tensors if t.nan_count > 0]
    assert len(nan_tensors) == 1
    assert nan_tensors[0].name == "model.layers.1.weight"
    assert any("Non-finite" in w for w in m.validation_warnings)


def test_no_value_check_skips_nan(tmp_path):
    model_dir = _make_mini_model(tmp_path, with_nan=True)
    m = inspect_model(model_dir, check_values=False)
    assert all(t.nan_count == 0 for t in m.tensors)
    assert not any("Non-finite" in w for w in m.validation_warnings)


def test_malformed_index(tmp_path):
    model_dir = tmp_path / "bad"
    model_dir.mkdir()
    (model_dir / "model.safetensors.index.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(SerializationError, match="Malformed"):
        inspect_model(model_dir)


def test_missing_index_and_single_file(tmp_path):
    model_dir = tmp_path / "single"
    model_dir.mkdir()
    tensors = {"w": np.zeros((2, 2), dtype=np.float32)}
    _write_shard(model_dir / "model.safetensors", tensors)
    m = inspect_model(model_dir)
    assert len(m.tensors) == 1
    assert m.tensors[0].name == "w"
    assert m.tensors[0].source_shard == "model.safetensors"


def test_no_model_files(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SerializationError, match="Neither"):
        inspect_model(empty)


def test_duplicate_tensor_across_shards(tmp_path):
    model_dir = tmp_path / "dup"
    model_dir.mkdir()
    t = {"shared.weight": np.ones((2,), dtype=np.float32)}
    _write_shard(model_dir / "a.safetensors", t)
    _write_shard(model_dir / "b.safetensors", t)
    index = {
        "weight_map": {
            "shared.weight": "a.safetensors",
        }
    }
    (model_dir / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    m = inspect_model(model_dir)
    # Second occurrence should be flagged as duplicate
    assert any("Duplicate tensor" in e for e in m.validation_errors)


def test_tensor_meta_to_dict_and_spec(tmp_path):
    model_dir = _make_mini_model(tmp_path)
    m = inspect_model(model_dir)
    t = m.tensors[0]
    d = t.to_dict()
    assert d["name"] == t.name
    assert d["shape"] == list(t.shape)
    spec = t.to_tensor_spec()
    assert spec.rank == len(t.shape)


def test_cli_model_inspect_human(tmp_path, capsys):
    from tensorgate.cli import main

    model_dir = _make_mini_model(tmp_path)
    main(["model", "inspect", str(model_dir)])
    out = capsys.readouterr().out
    assert "model_id=MiniModel" in out
    assert "shards=2" in out
    assert "manifest_hash=" in out


def test_cli_model_inspect_json_deterministic(tmp_path, capsys):
    from tensorgate.cli import main

    model_dir = _make_mini_model(tmp_path)
    main(["model", "inspect", str(model_dir), "--json"])
    out1 = capsys.readouterr().out
    main(["model", "inspect", str(model_dir), "--json"])
    out2 = capsys.readouterr().out
    assert out1 == out2
    data = json.loads(out1)
    assert "manifest_hash" in data
    assert data["model_id"] == "MiniModel"
    assert len(data["tensors"]) == 4


def test_load_model_directory_alias(tmp_path):
    from tensorgate import load_model_directory

    model_dir = _make_mini_model(tmp_path)
    m = load_model_directory(model_dir)
    assert m.model_id == "MiniModel"


# ---------------------------------------------------------------------------
# Optional integration test (explicit enable only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("TENSORGATE_QWEN_FIXTURE") != "1",
    reason="Set TENSORGATE_QWEN_FIXTURE=1 and provide a local Qwen3-4B dir to enable",
)
def test_qwen3_4b_integration():
    """Optional: inspect a real local Qwen3-4B directory.

    Enable with:
      TENSORGATE_QWEN_FIXTURE=1 TENSORGATE_QWEN_PATH=/path/to/Qwen3-4B pytest -k qwen
    """
    path = os.environ.get("TENSORGATE_QWEN_PATH")
    assert path, "TENSORGATE_QWEN_PATH must point to a local Qwen3-4B directory"
    m = inspect_model(path, model_id="Qwen/Qwen3-4B")
    assert len(m.shards) >= 1
    assert len(m.tensors) > 100
    assert m.manifest_hash
    # Two runs must be identical
    m2 = inspect_model(path, model_id="Qwen/Qwen3-4B")
    assert m.manifest_hash == m2.manifest_hash
