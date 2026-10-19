"""TensorGate command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .inspect import analyze, describe
from .normalize import normalize
from .transform import convert_dtype
from .compatibility import check_compatibility, adapt, compare
from .serialization import save_tensor, load_tensor
from .tensor import TensorSpec


def _load_array(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    if p.suffix in (".npz", ".npzz"):
        data = np.load(p)
        key = list(data.keys())[0]
        return data[key]
    raise SystemExit(f"Unsupported file type: {p.suffix}")


def cmd_inspect(args):
    arr = _load_array(args.file)
    print(describe(arr))


def cmd_analyze(args):
    arr = _load_array(args.file)
    desc = analyze(arr)
    print(json.dumps(desc.to_dict(), indent=2))


def cmd_normalize(args):
    arr = _load_array(args.file)
    out, meta = normalize(arr, method=args.method)
    out_path = args.output or str(Path(args.file).with_suffix(".normalized.npy"))
    save_tensor(out, out_path, extra={"normalize_meta": meta})
    print(json.dumps({k: v for k, v in meta.items() if k != "output_descriptor"}, indent=2, default=str))
    print(f"Wrote {out_path}")


def cmd_convert(args):
    arr = _load_array(args.file)
    out, meta = convert_dtype(arr, args.dtype)
    out_path = args.output or str(Path(args.file).with_suffix(f".{args.dtype}.npy"))
    save_tensor(out, out_path, extra={"convert_meta": meta})
    print(json.dumps(meta, indent=2, default=str))
    print(f"Wrote {out_path}")


def cmd_compare(args):
    a = _load_array(args.original)
    b = _load_array(args.reconstructed)
    print(json.dumps(compare(a, b), indent=2))


def cmd_compatibility(args):
    arr = _load_array(args.file)
    target = TensorSpec.from_dict(json.loads(Path(args.target).read_text()))
    report = check_compatibility(arr, target)
    print(json.dumps(report.to_dict(), indent=2, default=str))


def cmd_adapt(args):
    arr = _load_array(args.file)
    target = TensorSpec.from_dict(json.loads(Path(args.target).read_text()))
    adapted, report = adapt(arr, target)
    out_path = args.output or str(Path(args.file).with_suffix(".adapted.npy"))
    save_tensor(adapted, out_path, extra={"adapt_report": report.to_dict()})
    print(json.dumps(report.to_dict(), indent=2, default=str))
    print(f"Wrote {out_path}")


def cmd_model_inspect(args):
    """Inspect a local Hugging Face / safetensors model directory."""
    from .model import inspect_model

    manifest = inspect_model(
        args.model_dir,
        model_id=args.model_id,
        check_values=not args.no_value_check,
    )
    if args.json:
        # Deterministic JSON: sorted keys, stable separators
        print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print(manifest.summary())
        if manifest.validation_errors:
            sys.exit(1)


def cmd_model_transform(args):
    """Apply verified transformations to selected tensors of a local model."""
    from .model_transform import transform_model, SUPPORTED_OPS

    # Parse --transform NAME=OP[:key=val,...]
    transformations = {}
    for item in args.transform:
        if "=" not in item:
            raise SystemExit(
                f"Invalid --transform '{item}'. Expected NAME=OP or NAME=OP:key=val,..."
            )
        name, rest = item.split("=", 1)
        if ":" in rest:
            op, param_str = rest.split(":", 1)
            params = {}
            for part in param_str.split(","):
                if "=" not in part:
                    raise SystemExit(f"Invalid parameter in --transform: {part}")
                k, v = part.split("=", 1)
                # light type coercion
                if v.lower() in ("true", "false"):
                    params[k] = v.lower() == "true"
                else:
                    try:
                        params[k] = int(v)
                    except ValueError:
                        try:
                            params[k] = float(v)
                        except ValueError:
                            params[k] = v
            transformations[name] = {"operation": op, **params}
        else:
            transformations[name] = rest

    if not transformations:
        raise SystemExit(
            "At least one --transform NAME=OP is required. "
            f"Supported ops: {sorted(SUPPORTED_OPS)}"
        )

    report = transform_model(
        args.source_dir,
        args.output_dir,
        transformations,
        model_id=args.model_id,
        output_model_id=args.output_model_id,
        on_nonfinite=args.on_nonfinite,
        check_values=not args.no_value_check,
        overwrite_output=args.overwrite,
    )
    if args.json:
        d = report.to_dict()
        d.pop("source_path", None)
        d.pop("output_path", None)
        print(json.dumps(d, indent=2, sort_keys=True, ensure_ascii=True))
    else:
        print(report.summary())
        print(f"\nWrote transformed model to {args.output_dir}")
        print(f"Report: {args.output_dir}/tensorgate_transform_report.json")
    if report.validation_errors or report.rejected_operations:
        sys.exit(1)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="tensorgate",
        description="TensorGate — numerical compatibility boundary for tensors",
    )
    parser.add_argument("--version", action="version", version=f"tensorgate {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="Human-readable tensor summary")
    p.add_argument("file")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("analyze", help="Full JSON descriptor")
    p.add_argument("file")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("normalize", help="Normalize tensor")
    p.add_argument("file")
    p.add_argument("--method", default="symmetric")
    p.add_argument("-o", "--output", default=None)
    p.set_defaults(func=cmd_normalize)

    p = sub.add_parser("convert", help="Convert dtype")
    p.add_argument("file")
    p.add_argument("--dtype", required=True)
    p.add_argument("-o", "--output", default=None)
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("compare", help="Error metrics between two tensors")
    p.add_argument("original")
    p.add_argument("reconstructed")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("compatibility", help="Check compatibility against target.json")
    p.add_argument("file")
    p.add_argument("--target", required=True)
    p.set_defaults(func=cmd_compatibility)

    p = sub.add_parser("adapt", help="Adapt tensor to target.json")
    p.add_argument("file")
    p.add_argument("--target", required=True)
    p.add_argument("-o", "--output", default=None)
    p.set_defaults(func=cmd_adapt)

    # --- model subcommand group ---
    p_model = sub.add_parser("model", help="Multi-shard Hugging Face / safetensors model tools")
    model_sub = p_model.add_subparsers(dest="model_command", required=True)

    p_mi = model_sub.add_parser(
        "inspect",
        help="Inspect a local HF model directory (index + safetensors shards)",
    )
    p_mi.add_argument(
        "model_dir",
        help="Path to model directory (contains model.safetensors.index.json + shards)",
    )
    p_mi.add_argument(
        "--model-id",
        default=None,
        help="Override model identifier (default: directory name)",
    )
    p_mi.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable deterministic JSON manifest",
    )
    p_mi.add_argument(
        "--no-value-check",
        action="store_true",
        help="Skip per-tensor NaN/Inf scan (metadata + hashes only)",
    )
    p_mi.set_defaults(func=cmd_model_inspect)

    p_mt = model_sub.add_parser(
        "transform",
        help="Apply verified, recorded transformations to selected tensors",
    )
    p_mt.add_argument(
        "source_dir",
        help="Path to source model directory (never modified)",
    )
    p_mt.add_argument(
        "output_dir",
        help="Destination directory for the transformed model",
    )
    p_mt.add_argument(
        "--transform",
        action="append",
        default=[],
        metavar="NAME=OP[:key=val,...]",
        help=(
            "Tensor transform, e.g. "
            "'model.embed.weight=normalize:method=symmetric' or "
            "'lm_head.weight=convert_dtype:dtype=float16'. "
            "Repeatable. Supported ops: identity, normalize, convert_dtype, clip, quantize"
        ),
    )
    p_mt.add_argument("--model-id", default=None, help="Override source model id")
    p_mt.add_argument(
        "--output-model-id",
        default=None,
        help="Identifier for the output artifact",
    )
    p_mt.add_argument(
        "--on-nonfinite",
        choices=["reject", "preserve", "error"],
        default="reject",
        help="Policy when a selected tensor contains NaN/Inf (default: reject)",
    )
    p_mt.add_argument(
        "--no-value-check",
        action="store_true",
        help="Skip per-tensor NaN/Inf scan",
    )
    p_mt.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory",
    )
    p_mt.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable deterministic JSON report",
    )
    p_mt.set_defaults(func=cmd_model_transform)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
