"""In-process compiler bindings, with an explicit ask-opt compatibility path."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .ir import Module
from ._version import __version__
from .cuda import CUDAError
from .results import PrepareResult, PartitionResult, CodegenResult, split_report, stage_result


class CompilerError(RuntimeError):
    def __init__(self, message: str, *, diagnostics: str = "", returncode: int | None = None) -> None:
        self.diagnostics = diagnostics
        self.returncode = returncode
        super().__init__(message + ("\n" + diagnostics.strip() if diagnostics.strip() else ""))


def _result(stage, ir, report, **options):
    """Convert a CLI wire response into the same stage contract as native calls."""
    try:
        report = json.loads(report) if isinstance(report, str) else report
        contracts, diagnostics = split_report(report)
        if type(report.get("schema_version")) is not int or report["schema_version"] != 1:
            raise ValueError("unsupported ASK compiler result schema")
        phases = {"prepare": ("prepared", "optimized"), "partition": ("partitioned",),
                  "codegen": (options.get("target", "").replace("-", "_"),)}
        if stage not in phases or report.get("phase") not in phases[stage]:
            raise ValueError("ASK wire result phase differs from its requested stage")
        return stage_result(stage, ir, contracts, diagnostics, **options)
    except (ValueError, TypeError, KeyError, CUDAError) as error:
        raise CompilerError(f"invalid ASK {stage} result: {error}") from error


def _native_result(stage, result, **options):
    try:
        contracts = {"shape_snapshot": json.loads(result.shape_snapshot_json),
                     "external_requests": json.loads(result.external_requests_json)}
        if stage == "partition":
            contracts.update(execution_plan=json.loads(result.execution_plan_json),
                             step_sources=result.step_sources)
        elif stage == "codegen":
            if result.abi_json is not None:
                contracts["abi"] = json.loads(result.abi_json)
            if result.cuda_artifact_json is not None:
                contracts["cuda_artifact"] = json.loads(result.cuda_artifact_json)
            if result.output_path != options["output_path"]:
                raise ValueError("native object output path differs from the requested path")
        return stage_result(stage, result.ir, contracts, json.loads(result.diagnostics_json), **options)
    except (ValueError, TypeError, KeyError, AttributeError, CUDAError) as error:
        raise CompilerError(f"invalid ASK {stage} result: {error}") from error


class Compiler:
    """Use in-process bindings by default; an executable selects CLI compatibility.

    ``timeout`` applies to CLI/tool subprocesses. Native preparation owns its
    context for the duration of the call and does not offer asynchronous cancellation.
    """
    def __init__(self, executable: str | os.PathLike[str] | None = None, *, timeout: float = 60.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        self._native = None
        if executable is None:
            try:
                from . import _native
            except ImportError as exc:
                raise CompilerError("FugaTensor native compiler bindings are not installed. For CLI compatibility "
                                    "path, pass Compiler(executable='/path/to/ask-opt').") from exc
            self._native = _native
            native_version = json.loads(_native.capabilities()).get("ask_version")
            if native_version != __version__:
                raise CompilerError(f"FugaTensor expects ASK SDK {__version__}, but loaded "
                                    f"{native_version!r}; rebuild FugaTensor against the matching SDK")
            package = Path(_native.__file__).resolve().parent
            candidates = [package / "_bin" / "ask-opt", package.parent.parent / "bin" / "ask-opt"]
            candidates.extend(parent / "bin" / "ask-opt" for parent in package.parents)
            self.executable = str(next((path for path in candidates if path.is_file()), Path("ask-opt")))
        else:
            self.executable = os.fspath(executable)
            if not self.executable:
                raise ValueError("executable must not be empty")

    def _run(self, arguments: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run([self.executable, *arguments], capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=self.timeout,
                                    check=False)
        except FileNotFoundError as exc:
            raise CompilerError(f"native ASK executable not found: {self.executable}") from exc
        except subprocess.TimeoutExpired as exc:
            diagnostics = exc.stderr or ""
            if isinstance(diagnostics, bytes):
                diagnostics = diagnostics.decode("utf-8", errors="replace")
            raise CompilerError("native ASK compiler timed out", diagnostics=diagnostics) from exc
        except OSError as exc:
            raise CompilerError(f"cannot start native ASK compiler: {exc}") from exc
        if result.returncode < 0:
            raise CompilerError(f"native ASK compiler terminated by signal {-result.returncode}",
                                diagnostics=result.stderr or result.stdout,
                                returncode=result.returncode)
        if result.returncode:
            raise CompilerError("native ASK compiler rejected the request",
                                diagnostics=result.stderr or result.stdout,
                                returncode=result.returncode)
        return result

    def capabilities(self) -> dict[str, Any]:
        if self._native is not None:
            return json.loads(self._native.capabilities())
        result = self._run(["--capabilities"])
        try:
            capabilities = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise CompilerError("native ASK compiler returned invalid capability JSON",
                                diagnostics=result.stdout) from exc
        if not isinstance(capabilities, dict):
            raise CompilerError("native ASK capability report must be a JSON object")
        return capabilities

    def prepare(self, module: Module | str) -> PrepareResult:
        return self._process(module, ["--prepare"])

    def partition(self, module: Module | str, *, optimize=True) -> PartitionResult:
        """Partition a verified graph into a sequential CUDA execution plan."""
        source = module.to_mlir() if isinstance(module, Module) else module
        if not isinstance(source, str):
            raise TypeError("partition expects a Module or MLIR string")
        if type(optimize) is not bool:
            raise TypeError("partition options must be bool values")
        if self._native is not None:
            try:
                result = self._native.partition_typed(source, optimize=optimize)
            except RuntimeError as exc:
                raise CompilerError("native ASK compiler rejected partitioning", diagnostics=str(exc)) from exc
            return _native_result("partition", result)
        with tempfile.TemporaryDirectory(prefix="ask-partition-") as directory:
            root = Path(directory)
            (root / "input.mlir").write_text(source, encoding="utf-8")
            flags = ["--partition"]
            if not optimize:
                flags.append("--no-optimize")
            self._run([str(root / "input.mlir"), *flags, "-o", str(root / "output.mlir"),
                       "--report-json", str(root / "report.json")])
            return _result("partition", (root / "output.mlir").read_text(), (root / "report.json").read_text())

    def compile_cuda(self, module: Module | str, *, cache_dir, output_dir,
                     llc=None, ptxas=None, llvm_link=None, libdevice=None, compilers=None, target=None, block_size=128,
                     ptx_version=80):
        """Compile an ASK graph or a graph containing an external compiler node.

        Pure ASK graphs need an explicit target (backend=cuda, arch=SM number)
        and explicit llc/ptxas paths. External nodes retain their compiler's target
        and execution policy. No third-party compiler is loaded for a pure graph.
        """
        from .gpu import compile_graph
        return compile_graph(self, module, cache_dir=cache_dir, output_dir=output_dir,
                             compilers=compilers, llc=llc, ptxas=ptxas, llvm_link=llvm_link, libdevice=libdevice, target=target,
                             block_size=block_size, ptx_version=ptx_version)

    def emit_host_object(self, module: Module | str, output, *, optimize=True) -> CodegenResult:
        """Write a native host object and return its ABI and diagnostics separately."""
        return self._codegen(module, "host-object", output=os.fspath(output), optimize=optimize)

    def emit_host_llvm(self, module: Module | str, *, optimize=True) -> CodegenResult:
        """Lower a self-owned graph to native host LLVM assembly in process."""
        return self._codegen(module, "host-llvm", optimize=optimize)

    def emit_cuda_llvm(self, module: Module | str, *, arch, block_size=128,
                       ptx_version=80, optimize=True) -> CodegenResult:
        """Lower a self-owned graph and return LLVM assembly with its CUDA ABI."""
        for name, value, lower, upper in (("arch", arch, 10, 999),
                                          ("block_size", block_size, 1, 1024),
                                          ("ptx_version", ptx_version, 10, 999)):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer between {lower} and {upper}")
        return self._codegen(module, "cuda-llvm", arch=arch, block_size=block_size,
                             ptx_version=ptx_version, optimize=optimize)

    def _codegen(self, module, target, *, output="", arch=0, block_size=128,
                 ptx_version=80, optimize=True):
        source = module.to_mlir() if isinstance(module, Module) else module
        if not isinstance(source, str):
            raise TypeError("code generation expects a Module or MLIR string")
        if type(optimize) is not bool:
            raise TypeError("optimize must be a bool")
        if target == "host-object":
            if not output:
                raise ValueError("host object generation requires an output path")
            output = str(Path(output).expanduser().resolve())
        if self._native is not None:
            native_target = {"host-llvm": self._native.CodegenTarget.HostLLVM,
                             "host-object": self._native.CodegenTarget.HostObject,
                             "cuda-llvm": self._native.CodegenTarget.CUDALLVM}[target]
            try:
                result = self._native.codegen_typed(source, native_target, output=output, arch=arch,
                    block_size=block_size, ptx_version=ptx_version, optimize=optimize)
            except RuntimeError as exc:
                raise CompilerError("native ASK code generation failed", diagnostics=str(exc)) from exc
            if result.target != native_target:
                raise CompilerError("native code generation result has an unexpected target")
            return _native_result("codegen", result, target=target, output_path=output)
        with tempfile.TemporaryDirectory(prefix="ask-codegen-") as directory:
            root = Path(directory)
            source_path, report_path = root / "input.mlir", root / "report.json"
            result_path = Path(output) if target == "host-object" else root / "output.ll"
            source_path.write_text(source, encoding="utf-8")
            mode = {"host-object": "--emit-object", "host-llvm": "--emit-llvm",
                    "cuda-llvm": "--emit-cuda-llvm"}[target]
            flags = [mode, "-o", str(result_path), "--report-json", str(report_path)]
            if optimize:
                flags.append("--optimize")
            if target == "cuda-llvm":
                flags.extend(["--cuda-arch", str(arch), "--cuda-block-size", str(block_size),
                              "--cuda-ptx-version", str(ptx_version)])
            self._run([str(source_path), *flags])
            return _result("codegen", "" if target == "host-object" else result_path.read_text(encoding="utf-8"),
                           report_path.read_text(encoding="utf-8"), target=target, output_path=output)

    def optimize(self, module: Module | str, *, fold_constants: bool = True,
                 eliminate_dead_code: bool = True,
                 fuse_elementwise: bool = True) -> PrepareResult:
        """Prepare and optimize ASK IR using the native compiler's explicit passes."""
        options = ["--optimize"]
        for enabled, flag in ((fold_constants, "--no-constant-fold"),
                              (eliminate_dead_code, "--no-dce"),
                              (fuse_elementwise, "--no-fusion")):
            if type(enabled) is not bool:
                raise TypeError("optimization options must be bool values")
            if not enabled:
                options.append(flag)
        return self._process(module, options)

    def _process(self, module: Module | str, options: list[str]) -> PrepareResult:
        if isinstance(module, Module):
            source = module.to_mlir()
        elif isinstance(module, str):
            source = module
        else:
            raise TypeError("prepare/optimize expects a Module or MLIR string")
        if self._native is not None:
            try:
                result = self._native.prepare_typed(
                    source, optimize="--optimize" in options,
                    fold_constants="--no-constant-fold" not in options,
                    eliminate_dead_code="--no-dce" not in options,
                    fuse_elementwise="--no-fusion" not in options)
            except RuntimeError as exc:
                raise CompilerError("native ASK compiler rejected the request", diagnostics=str(exc)) from exc
            return _native_result("prepare", result)
        with tempfile.TemporaryDirectory(prefix="ask-prepare-") as directory:
            path = Path(directory)
            input_path = path / "input.mlir"
            output_path = path / "prepared.mlir"
            report_path = path / "report.json"
            input_path.write_text(source, encoding="utf-8")
            result = self._run([str(input_path), "-o", str(output_path), *options,
                                "--report-json", str(report_path)])
            try:
                ir = output_path.read_text(encoding="utf-8")
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise CompilerError("native ASK compiler did not produce valid preparation output",
                                    diagnostics=result.stderr) from exc
            if not isinstance(report, dict):
                raise CompilerError("native ASK preparation report must be a JSON object")
            return _result("prepare", ir, report)
