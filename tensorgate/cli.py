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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
