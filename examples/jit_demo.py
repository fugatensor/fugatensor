"""Run small FugaTensor graphs through ASK's CPU or CUDA JIT paths."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import fugatensor as ft
from ask import CompilerEnvironment


def _details(function):
    directory = Path(function.last_artifact_dir)
    ir_path = directory / "frontend.mlir"
    ir_path.write_text(function.last_ir)
    return {
        "cache": function.cache_info(),
        "ir_file": str(ir_path),
        "artifact_dir": str(directory),
        "phase": function.last_report.get("phase"),
        "external_compiler": function.last_report.get("fusion", {}).get("compiler"),
        "ir_cache": function.last_report.get("ir_cache", {}).get("status"),
        "executable_cache": function.last_report.get("executable_cache", {}).get("status"),
    }


def cpu_demo():
    @ft.jit
    def arithmetic(x, y):
        return (x + y) * 2

    x = ft.tensor([[1, 2], [3, 4]])
    y = ft.tensor([[10, 20], [30, 40]])
    first = arithmetic(x, y).tolist()
    expected = [[22.0, 44.0], [66.0, 88.0]]
    if first != expected:
        raise AssertionError((first, expected))
    changed = ft.tensor([[5, 6], [7, 8]])
    second = arithmetic(changed, y).tolist()
    expected_second = [[30.0, 52.0], [74.0, 96.0]]
    if second != expected_second:
        raise AssertionError((second, expected_second))
    return {"device": "cpu", "expression": "(x + y) * 2", "first": first,
            "new_input": second, "shape": x.shape, "dtype": x.dtype,
            "verified": True, **_details(arithmetic)}


def cuda_demo():
    source = Path(__file__).with_name("kernels.py")
    scale = ft.triton.kernel(
        source, entry="scale_kernel", input="inp", output="out",
        constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
        grid=lambda spec: ((spec.numel + 127) // 128, 1, 1),
        options={"num_warps": 4},
    )

    @ft.jit
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
            "user_kernel": str(source), "verified": True, **_details(mixed)}


def configure_tools(args):
    """Keep local toolchain paths separate from tensor calculations."""
    if args.device in ("cpu", "both"):
        ft.configure(device="cpu", ask_opt=args.ask_opt, cache_dir=args.cache_dir / "cpu")
    if args.device in ("cuda", "both"):
        missing = [name for name in ("triton_python", "triton_worker", "llc", "ptxas")
                   if not getattr(args, name)]
        if missing:
            raise ValueError("CUDA demo needs " + ", ".join("--" + name.replace("_", "-") for name in missing))
        ft.configure(device="cuda", ask_opt=args.ask_opt,
                     compilers={"triton": CompilerEnvironment(python=args.triton_python, worker=args.triton_worker)},
                     llc=args.llc, ptxas=args.ptxas, cache_dir=args.cache_dir / "cuda")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "both"), default="cpu")
    parser.add_argument("--ask-opt", default=os.environ.get("ASK_OPT"))
    parser.add_argument("--triton-python", default=os.environ.get("FUGATENSOR_TRITON_PYTHON"))
    parser.add_argument("--triton-worker", default=os.environ.get("FUGATENSOR_TRITON_WORKER"))
    parser.add_argument("--llc", default=os.environ.get("FUGATENSOR_LLC"))
    parser.add_argument("--ptxas", default=os.environ.get("FUGATENSOR_PTXAS"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/fugatensor-demo"))
    parser.add_argument("--json", type=Path, help="also save the validation summary")
    args = parser.parse_args(argv)
    records = []
    try:
        configure_tools(args)
        if args.device in ("cpu", "both"):
            records.append(cpu_demo())
        if args.device in ("cuda", "both"):
            records.append(cuda_demo())
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
