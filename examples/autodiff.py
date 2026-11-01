"""Run with .venv/bin/python examples/autodiff.py --device both [--triton]."""
from __future__ import annotations

import argparse
from pathlib import Path

import fugatensor as ft


class Square(ft.fun):
    @ft.jit
    def forward(self, ctx, x):
        ctx.save_for_backward(x)
        return ft.square(x)

    @ft.jit
    def backward(self, ctx, dy):
        (x,) = ctx.saved_tensors
        return (2 * x * dy,)


def run_basic(device, dtype):
    square = Square()

    @ft.jit
    def loss(x):
        return square(x) + 3 * x

    @ft.jit
    def builtin_loss(x):
        return ft.square(x) + 3 * x

    @ft.jit
    def cube(x):
        return x * x * x

    @ft.jit
    def controlled(x, take_square, count):
        if take_square:
            x = ft.square(x)
        while count:
            x = x * 2
            count = count - 1
        return x

    with ft.Engine(device=device):
        scalar = ft.tensor(2, dtype=dtype, requires_grad=True)
        output = loss(scalar)
        output.backward()
        value, gradient = output.tolist(), scalar.grad.tolist()
        assert (value, gradient) == (10.0, 7.0)
        builtin_input = ft.tensor(2, dtype=dtype, requires_grad=True)
        builtin_loss(builtin_input).backward()
        assert builtin_input.grad.tolist() == gradient

        x = ft.tensor([1, 2, 3], dtype=dtype, requires_grad=True)
        square(x).backward(ft.tensor([1, 2, -1], dtype=dtype))
        vector_gradient = x.grad.tolist()
        assert vector_gradient == [2.0, 8.0, -6.0]
        print(f"{device} {dtype}: loss={value}, x.grad={gradient}, seeded backward={vector_gradient}")
        print("Forward ASK artifacts:", loss.last_artifact_dir)

        start = ft.tensor(2, dtype=dtype, requires_grad=True)
        selected = controlled(start, ft.tensor(True, dtype="bool"), ft.tensor(3, dtype="i32"))
        selected.backward()
        assert (selected.tolist(), start.grad.tolist()) == (32.0, 32.0)
        assert len(controlled.last_segments) > 1
        print(f"{device} runtime if/while: output=32.0, x.grad=32.0, segments={len(controlled.last_segments)}")

        higher = ft.tensor(2, dtype=dtype, requires_grad=True)
        cube(higher).backward(create_graph=True)
        first = higher.grad
        higher.zero_grad()
        first.backward(create_graph=True)
        second = higher.grad
        higher.zero_grad()
        second.backward()
        derivatives = [first.tolist(), second.tolist(), higher.grad.tolist()]
        assert derivatives == [12.0, 12.0, 6.0]
        print(f"{device} cube at 2: first/second/third derivatives={derivatives}")


def run_triton():
    scale = ft.triton.kernel(
        Path(__file__).with_name("kernels.py"), entry="scale_kernel", input="inp", output="out",
        constants=lambda spec: {"N": spec.numel, "BLOCK": 128},
        grid=lambda spec: ((spec.numel + 127) // 128,),
    )

    class UserScale(ft.fun):
        @ft.jit
        def forward(self, ctx, x):
            return scale(x)

        @ft.jit
        def backward(self, ctx, dy):
            return (scale(dy),)

    op = UserScale()

    @ft.jit
    def mixed(x):
        return op(x + 1) + 3

    with ft.Engine(device="cuda"):
        x = ft.tensor([1, 2, 3], requires_grad=True)
        output = mixed(x)
        output.backward(ft.tensor([1, 2, -1]))
        result, gradient = output.tolist(), x.grad.tolist()
        assert result == [7.0, 9.0, 11.0]
        assert gradient == [2.0, 4.0, -2.0]
        assert mixed.last_report["phase"] == "fused_llvm"
        assert mixed.last_report["fusion"]["compiler"] == "triton"
        print(f"cuda user Triton: forward={result}, x.grad={gradient}")

        higher = mixed(ft.tensor(3, requires_grad=True))
        try:
            higher.backward(create_graph=True)
        except ValueError as error:
            assert "external" in str(error), error
            print("Triton higher-order rejected before reverse execution:", error)
        else:
            raise AssertionError("opaque Triton higher-order request must be rejected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "both"), default="cpu")
    parser.add_argument("--dtype", choices=("f32", "f16", "bf16"), default="f32")
    parser.add_argument("--triton", action="store_true", help="also run user Triton forward and backward")
    args = parser.parse_args()
    for device in (("cpu", "cuda") if args.device == "both" else (args.device,)):
        run_basic(device, args.dtype)
    if args.triton:
        run_triton()


if __name__ == "__main__":
    main()
