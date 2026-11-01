"""A user-written Triton kernel stays inside the Fuga -> ASK compilation path.

After `python3 tools/bootstrap.py`, run:
    .venv/bin/python examples/mixed_triton.py
"""
from pathlib import Path

import fugatensor as ft


def main():
    scale = ft.triton.kernel(
        Path(__file__).with_name("kernels.py"), entry="scale_kernel",
        input="inp", output="out",
        constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
        grid=lambda spec: ((spec.numel + 127) // 128,),
    )

    @ft.jit
    def mixed(x):
        return scale(x + 1) + 3

    try:
        x = ft.tensor([1, 2, 3], device="cuda")
        result = mixed(x).tolist()
        assert result == [7.0, 9.0, 11.0], result
        assert mixed.last_report["phase"] == "fused_llvm"
        assert mixed.last_report["fusion"]["compiler"] == "triton"
        print(result)
        print("\nFuga IR:")
        print(mixed.last_fuga_ir)
        print("ASK IR and LLVM artifacts:", mixed.last_artifact_dir)
    finally:
        ft.shutdown()


if __name__ == "__main__":
    main()
