"""A user-written Triton kernel stays inside the Fuga -> ASK compilation path.

After installing both packages and configuring CUDA/Triton (see docs/安装.md), run:
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

    def mixed(x):
        return scale(x + 1) + 3

    with ft.Engine(device="cuda") as engine:
        x = ft.tensor([1, 2, 3], device="cuda")
        result = mixed(x).tolist()
        assert result == [7.0, 9.0, 11.0], result
        assert [step["kind"] for step in engine.last_report["execution_plan"]["steps"]] == ["owned", "external", "owned"]
        print(result)
        print("\nFuga IR:")
        print(engine.last_fuga_ir)
        print("ASK IR and compiled kernel artifacts:", engine.last_artifact_dir)


if __name__ == "__main__":
    main()
