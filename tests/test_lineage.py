"""Shared Evidence Contract — TensorGate computational lineage."""

import numpy as np
import pytest

from tensorgate import (
    analyze,
    normalize,
    ComputationKind,
    TensorIdentity,
    ComputationalLineage,
    build_lineage,
    check_lineage_schema,
    attach_evidence_ref,
    identity_from_descriptor,
    ValidationError,
    CompatibilityError,
)


def test_tensor_identity_roundtrip():
    t = TensorIdentity(
        content_hash="abc",
        shape=(2, 3),
        dtype="float32",
        schema_id="s1",
        schema_version="1.0",
        label="weights",
    )
    d = t.to_dict()
    t2 = TensorIdentity.from_dict(d)
    assert t2.content_hash == "abc"
    assert t2.shape == (2, 3)
    assert t2.dtype == "float32"


def test_build_lineage_deterministic():
    w = np.random.randn(8, 4).astype(np.float32)
    desc = analyze(w)
    out, meta = normalize(w, method="symmetric")
    out_desc = analyze(out)

    lin = build_lineage(
        operation="normalize/symmetric",
        input_hashes=[desc.content_hash],
        output_hash=out_desc.content_hash,
        parameters={"method": "symmetric"},
        kind=ComputationKind.DETERMINISTIC,
        input_identities=[identity_from_descriptor(desc, label="input")],
        output_identity=identity_from_descriptor(out_desc, label="output"),
    )
    assert lin.kind == ComputationKind.DETERMINISTIC
    assert lin.transformation_id == "normalize/symmetric"
    assert len(lin.inputs) == 1
    assert lin.output is not None
    assert lin.output.content_hash == out_desc.content_hash
    assert len(lin.provenance.chain) == 1
    check_lineage_schema(lin)


def test_deterministic_rejects_model_id():
    with pytest.raises(ValidationError):
        build_lineage(
            operation="infer",
            input_hashes=["a"],
            output_hash="b",
            kind=ComputationKind.DETERMINISTIC,
            model_id="resnet",
        )


def test_nondeterministic_allows_model():
    lin = build_lineage(
        operation="model_forward",
        input_hashes=["in1"],
        output_hash="out1",
        kind=ComputationKind.NONDETERMINISTIC,
        model_id="toy-mlp",
        model_version="0.2.0",
    )
    assert lin.kind == ComputationKind.NONDETERMINISTIC
    assert lin.model_id == "toy-mlp"
    assert lin.model_version == "0.2.0"


def test_missing_input_hash_fails():
    with pytest.raises(ValidationError):
        build_lineage(
            operation="x",
            input_hashes=[],
            output_hash="o",
        )


def test_missing_output_hash_fails():
    with pytest.raises(ValidationError):
        build_lineage(
            operation="x",
            input_hashes=["i"],
            output_hash="",
        )


def test_lineage_schema_incompatible():
    lin = build_lineage(
        operation="noop",
        input_hashes=["a"],
        output_hash="b",
    )
    lin.schema_version = "1.0.0"
    with pytest.raises(CompatibilityError):
        check_lineage_schema(lin)


def test_lineage_roundtrip_dict():
    lin = build_lineage(
        operation="convert_dtype",
        input_hashes=["h1"],
        output_hash="h2",
        parameters={"dtype": "float16"},
    )
    lin = attach_evidence_ref(
        lin,
        event_id="node|0x100|temp|1",
        source_id="temp_sensor_01",
        schema_id="cantheon.metafield_boundary",
        schema_version="0.3.0",
    )
    d = lin.to_dict()
    lin2 = ComputationalLineage.from_dict(d)
    assert lin2.transformation_id == "convert_dtype"
    assert lin2.inputs[0].content_hash == "h1"
    assert lin2.output.content_hash == "h2"
    assert len(lin2.evidence_refs) == 1
    assert lin2.evidence_refs[0]["event_id"] == "node|0x100|temp|1"
    check_lineage_schema(lin2)


def test_provenance_preserved_across_chain():
    lin1 = build_lineage(
        operation="normalize/maxabs",
        input_hashes=["a"],
        output_hash="b",
    )
    lin2 = build_lineage(
        operation="quantize/8bit",
        input_hashes=["b"],
        output_hash="c",
        prior=lin1.provenance,
    )
    assert len(lin2.provenance.chain) == 2
    assert lin2.provenance.chain[0].operation == "normalize/maxabs"
    assert lin2.provenance.chain[1].operation == "quantize/8bit"
