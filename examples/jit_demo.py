"""Run small FugaTensor graphs through ASK's CPU or CUDA JIT paths."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import fugatensor as ft
from ask import CompilerEnvironment


def _details(engine):
    directory = Path(engine.last_artifact_dir)
    ir_path = directory / "frontend.mlir"
    ir_path.write_text(engine.last_ask_ir)
    return {
        "cache": engine.cache_info(),
        "ir_file": str(ir_path),
        "fuga_ir_file": str(directory / "fuga.ir"),
        "artifact_dir": str(directory),
        "phase": engine.last_report.get("phase"),
        "external_compiler": engine.last_report.get("fusion", {}).get("compiler"),
        "ir_cache": engine.last_report.get("ir_cache", {}).get("status"),
        "executable_cache": engine.last_report.get("executable_cache", {}).get("status"),
    }


def arithmetic_demo(device):
    def arithmetic(x, y):
        return (x + y) * 2

    x = ft.tensor([[1, 2], [3, 4]], device=device)
    y = ft.tensor([[10, 20], [30, 40]], device=device)
    first = arithmetic(x, y).tolist()
    expected = [[22.0, 44.0], [66.0, 88.0]]
    if first != expected:
        raise AssertionError((first, expected))
    changed = ft.tensor([[5, 6], [7, 8]], device=device)
    second = arithmetic(changed, y).tolist()
    expected_second = [[30.0, 52.0], [74.0, 96.0]]
    if second != expected_second:
        raise AssertionError((second, expected_second))
    return {"device": device, "expression": "(x + y) * 2", "first": first,
            "new_input": second, "shape": x.shape, "dtype": x.dtype,
            "verified": True, **_details(x.engine)}


def triton_demo():
    source = Path(__file__).with_name("kernels.py")
    scale = ft.triton.kernel(
        source, entry="scale_kernel", input="inp", output="out",
        constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
        grid=lambda spec: ((spec.numel + 127) // 128, 1, 1),
        options={"num_warps": 4},
    )

    def mixed(x):
        return scale(x + 1) + 3

    x = ft.tensor([[1, 2], [3, 4]], device="cuda")
    first = mixed(x).tolist()
    expected = [[7.0, 9.0], [11.0, 13.0]]
    if first != expected:
        raise AssertionError((first, expected))
    changed = ft.tensor([[5, 6], [7, 8]], device="cuda")
    second = mixed(changed).tolist()
    expected_second = [[15.0, 17.0], [19.0, 21.0]]
    if second != expected_second:
        raise AssertionError((second, expected_second))
    return {"device": "cuda", "expression": "ASK(x + 1) -> Triton(* 2) -> ASK(+ 3)",
            "first": first, "new_input": second, "shape": x.shape, "dtype": x.dtype,
            "user_kernel": str(source), "verified": True, **_details(x.engine)}


def configure_tools(args):
    """Keep local toolchain paths separate from tensor calculations."""
    if args.device in ("cpu", "both"):
        ft.configure(device="cpu", ask_opt=args.ask_opt, cache_dir=args.cache_dir / "cpu")
    if args.device in ("cuda", "both"):
        options = {name: getattr(args, name) for name in ('ask_opt', 'llc', 'ptxas')
                   if getattr(args, name) is not None}
        if not args.triton:
            options['compilers'] = {}
        elif args.triton_python or args.triton_worker:
            if not (args.triton_python and args.triton_worker):
                raise ValueError('provide both --triton-python and --triton-worker')
            options['compilers'] = {'triton': CompilerEnvironment(
                python=args.triton_python, worker=args.triton_worker)}
        ft.configure(device='cuda', cache_dir=args.cache_dir / 'cuda', **options)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "both"), default="cpu")
    parser.add_argument("--triton", action="store_true", help="also run the hand-written Triton kernel example")
    parser.add_argument("--ask-opt", default=os.environ.get("ASK_OPT"))
    parser.add_argument("--triton-python", default=os.environ.get("FUGATENSOR_TRITON_PYTHON"))
    parser.add_argument("--triton-worker", default=os.environ.get("FUGATENSOR_TRITON_WORKER"))
    parser.add_argument("--llc", default=os.environ.get("FUGATENSOR_LLC"))
    parser.add_argument("--ptxas", default=os.environ.get("FUGATENSOR_PTXAS"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/fugatensor-demo"))
    parser.add_argument("--json", type=Path, help="also save the validation summary")
    args = parser.parse_args(argv)
    if args.triton and args.device == "cpu":
        parser.error("--triton requires --device cuda or both")
    records = []
    try:
        configure_tools(args)
        if args.device in ("cpu", "both"):
            records.append(arithmetic_demo("cpu"))
        if args.device in ("cuda", "both"):
            records.append(arithmetic_demo("cuda"))
            if args.triton:
                records.append(triton_demo())
    finally:
        ft.shutdown()
    summary = {"verified": True, "runs": records}
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
