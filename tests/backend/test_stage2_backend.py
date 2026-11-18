"""Backend bridge, partitioning, Host execution, and CUDA pre_copy contract checks for Stage 2 ops."""
from __future__ import annotations

import math
from pathlib import Path
import shutil
import tempfile

import pytest

from fugatensor._ask import Compiler, CUDAKernel, CUDAProgram, Module, ScalarType, TensorType
from fugatensor.backends.host import HostBackend, HostStorage
from tests._helpers import make_ask_compiler
from tests.backend.support.host_runner import llc_path, open_cuda_session, ptxas_path


def _stage2_host_module() -> Module:
    m = Module()
    f = m.graph(
        "calculate",
        inputs={
            "x": TensorType((4,), "f32"),
            "i": TensorType((4,), "i32"),
            "idx": TensorType((3,), "i64"),
            "base": TensorType((3, 2), "f32"),
            "upd": TensorType((3, 2), "f32"),
        },
        results=[
            TensorType((4,), "f32"),
            TensorType((4,), "i32"),
            TensorType((), "i64"),
            TensorType((3, 2), "f32"),
            TensorType((2, 2), "f32"),
        ],
    )
    x, i, idx, base, upd = (
        f.inputs["x"],
        f.inputs["i"],
        f.inputs["idx"],
        f.inputs["base"],
        f.inputs["upd"],
    )
    math_out = f.primitive("sin", (x,), TensorType((4,), "f32"), {})
    sh_amt = f.splat(f.constant(1, dtype="i32"), (4,))
    int_out = f.primitive("shift_left", (f.primitive("bitwise_xor", (i, sh_amt), TensorType((4,), "i32"), {}), sh_amt), TensorType((4,), "i32"), {})
    amx = f.splat(f.primitive("argmax", (x,), ScalarType("i64"), {"axis": 0}), ())
    zero_pad = f.constant(0.5, dtype="f32")
    sl = f.primitive("slice", (base,), TensorType((2, 2), "f32"), {"starts": (2, 0), "stops": (0, 2), "steps": (-1, 1)})
    pd = f.primitive("pad", (sl, zero_pad), TensorType((3, 2), "f32"), {"pad_low": (1, 0), "pad_high": (0, 0), "interior": (0, 0)})
    scat = f.primitive("scatter_add", (pd, idx, upd), TensorType((3, 2), "f32"), {"axis": 0})
    gth = f.primitive("gather", (scat, f.primitive("iota", (), TensorType((2,), "i64"), {"dimension": 0}),), TensorType((2, 2), "f32"), {"axis": 0})
    f.return_(math_out, int_out, amx, scat, gth)
    return m


def _scatter_plan_module(*, single_step: bool) -> Module:
    m = Module()
    base_t = TensorType((4, 2), "f32")
    idx_t = TensorType((3,), "i32")
    upd_t = TensorType((3, 2), "f32")
    f = m.graph("main", inputs={"base": base_t, "idx": idx_t, "upd": upd_t}, results=[base_t])
    base, idx, upd = f.inputs["base"], f.inputs["idx"], f.inputs["upd"]
    if single_step:
        out = f.primitive("scatter_add", (base, idx, upd), base_t, {"axis": 0})
    else:
        scaled = f.mul(upd, f.splat(f.constant(2.0, dtype="f32"), (3, 2)))
        scat = f.primitive("scatter_add", (base, idx, scaled), base_t, {"axis": 0})
        out = f.add(scat, f.splat(f.constant(1.0, dtype="f32"), (4, 2)))
    f.return_(out)
    return m


class TestStage2Backend:
    def test_host_execution_of_stage2_primitives(self):
        compiler = make_ask_compiler()
        with tempfile.TemporaryDirectory() as tmp:
            exe = HostBackend(compiler, tmp).compile(_stage2_host_module())
            x = [0.2, 1.5, -0.3, 1.5]
            i = [1, 2, -4, 7]
            idx = [2, 0, 2]
            base = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
            upd = [[10.0, 20.0], [1.0, 2.0], [3.0, 4.0]]
            flat_base = [v for row in base for v in row]
            flat_upd = [v for row in upd for v in row]
            math_out, int_out, amx, scat, gth = exe.run([
                HostStorage.from_flat(x, shape=(4,), dtype="f32"),
                HostStorage.from_flat(i, shape=(4,), dtype="i32"),
                HostStorage.from_flat(idx, shape=(3,), dtype="i64"),
                HostStorage.from_flat(flat_base, shape=(3, 2), dtype="f32"),
                HostStorage.from_flat(flat_upd, shape=(3, 2), dtype="f32"),
            ])
            for a, b in zip(math_out.to_flat(), [math.sin(v) for v in x]):
                assert math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)
            assert list(int_out.to_flat()) == [((v ^ 1) << 1) for v in i]
            assert list(amx.to_flat()) == [1]  # first tie at 1.5
            # sl reverses rows 2, 1 -> [[5, 6], [3, 4]]; pd adds [0.5, 0.5] at top -> [[0.5, 0.5], [5, 6], [3, 4]]
            # scatter_add at idx=[2, 0, 2] -> row0: [1.5, 2.5], row1: [5, 6], row2: [16, 28]
            expected_scat = [1.5, 2.5, 5.0, 6.0, 16.0, 28.0]
            assert [round(v, 4) for v in scat.to_flat()] == expected_scat
            assert [round(v, 4) for v in gth.to_flat()] == [1.5, 2.5, 5.0, 6.0]

    def test_partition_isolates_scatter_add_step(self):
        compiler = Compiler()
        part = compiler.partition(_scatter_plan_module(single_step=False))
        plan = part.execution_plan
        assert [s["kind"] for s in plan["steps"]] == ["owned", "owned", "owned"]
        single = compiler.partition(_scatter_plan_module(single_step=True))
        assert [s["kind"] for s in single.execution_plan["steps"]] == ["owned"]

    @pytest.mark.cuda
    def test_cuda_single_step_and_multi_step_scatter_pre_copy_and_graph_reuse(self):
        compiler = make_ask_compiler()
        with open_cuda_session() as session:
            arch = session.capability[0] * 10 + session.capability[1]
            for single_step in (True, False):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    out_dir = root / "compiled"
                    artifact = compiler.compile_cuda(
                        _scatter_plan_module(single_step=single_step),
                        target={"backend": "cuda", "arch": arch},
                        llc=llc_path(),
                        ptxas=ptxas_path(),
                        cache_dir=root / "cache",
                        output_dir=out_dir,
                    )
                    if single_step:
                        assert isinstance(artifact, CUDAKernel)
                    else:
                        assert isinstance(artifact, CUDAProgram)
                        shutil.move(out_dir, root / "relocated")
                        artifact = CUDAProgram.load(root / "relocated")

                    for use_graph in (False, True):
                        prepared = artifact.prepare(session, graph=use_graph)
                        # Deliberately use non-zero base values and OOB indices, plus uninitialized reused destination
                        b_vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
                        i_vals = [1, -1, 1]
                        u_vals = [1.0, 2.0, 99.0, 99.0, 3.0, 4.0]
                        for round_idx in range(2):
                            scale = 1.0 + round_idx
                            b = session.from_host([v * scale for v in b_vals], shape=(4, 2), dtype="f32")
                            idx = session.from_host(i_vals, shape=(3,), dtype="i32")
                            upd = session.from_host([v * scale for v in u_vals], shape=(3, 2), dtype="f32")
                            dirty_out = session.from_host([999.0] * 8, shape=(4, 2), dtype="f32")
                            prepared.submit_many([b, idx, upd], [dirty_out]).wait()
                            mult = 1.0 if single_step else 2.0
                            bias = 0.0 if single_step else 1.0
                            expected = [
                                10.0 * scale + bias, 20.0 * scale + bias,
                                30.0 * scale + mult * 4.0 * scale + bias, 40.0 * scale + mult * 6.0 * scale + bias,
                                50.0 * scale + bias, 60.0 * scale + bias,
                                70.0 * scale + bias, 80.0 * scale + bias,
                            ]
                            assert dirty_out.to_host() == pytest.approx(expected, rel=1e-5, abs=1e-5)
