"""Compiler stage results; execution contracts are separate from diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


CodegenTarget = Literal["host-llvm", "host-object", "cuda-llvm"]
_CONTRACT_FIELDS = frozenset(("shape_snapshot", "external_requests", "execution_plan",
                              "step_sources", "abi", "cuda_artifact"))


@dataclass(frozen=True)
class _CompilationResult:
    ir: str
    shape_snapshot: dict[str, Any]
    external_requests: list[dict[str, Any]]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class PrepareResult(_CompilationResult):
    pass


@dataclass(frozen=True)
class PartitionResult(_CompilationResult):
    execution_plan: dict[str, Any]
    step_sources: dict[str, str]


@dataclass(frozen=True)
class CodegenResult(_CompilationResult):
    target: CodegenTarget
    output_path: str
    abi: dict[str, Any] | None
    cuda_artifact: dict[str, Any] | None


def split_report(report):
    """Decode the CLI's versioned wire report without retaining a second contract."""
    if not isinstance(report, dict):
        raise ValueError("ASK compiler report must be a JSON object")
    return ({key: value for key, value in report.items() if key in _CONTRACT_FIELDS},
            {key: value for key, value in report.items() if key not in _CONTRACT_FIELDS})


def stage_result(stage, ir, contracts, diagnostics, *, target=None, output_path=""):
    """Check native and CLI envelopes at the same bridge entry point.

    Versioned plan/kernel payloads retain their existing semantic validators.
    No execution input is reconstructed from diagnostic text or fields.
    """
    if not isinstance(ir, str) or not isinstance(diagnostics, dict):
        raise ValueError("ASK compiler must return IR text and diagnostic metadata")
    if _CONTRACT_FIELDS.intersection(diagnostics):
        raise ValueError("ASK diagnostics contain execution contract fields")
    shape = contracts.get("shape_snapshot")
    requests = contracts.get("external_requests")
    if not isinstance(shape, dict):
        raise ValueError("ASK compiler result requires a shape_snapshot object")
    if not isinstance(requests, list) or any(not isinstance(request, dict) for request in requests):
        raise ValueError("ASK compiler result requires an external_requests array of objects")
    common = dict(ir=ir, shape_snapshot=shape, external_requests=requests, diagnostics=diagnostics)
    expected = {"shape_snapshot", "external_requests"}
    if stage in ("prepare", "partition") and not ir.strip():
        raise ValueError("ASK graph result requires nonempty IR text")
    if stage == "prepare":
        if set(contracts) != expected:
            raise ValueError("invalid ASK preparation result contract")
        return PrepareResult(**common)
    if stage == "partition":
        if set(contracts) != expected | {"execution_plan", "step_sources"}:
            raise ValueError("invalid ASK partition result contract")
        from .plan import validate_plan
        plan = validate_plan(contracts["execution_plan"])
        sources = contracts["step_sources"]
        if (not isinstance(sources, dict) or set(sources) != {step["id"] for step in plan["steps"]}
                or any(not isinstance(source, str) or not source.strip() for source in sources.values())):
            raise ValueError("ASK partition must provide exactly one MLIR source per step")
        return PartitionResult(**common, execution_plan=plan, step_sources=sources)
    if stage != "codegen" or target not in ("host-llvm", "host-object", "cuda-llvm"):
        raise ValueError("unsupported ASK code generation target")
    field = "cuda_artifact" if target == "cuda-llvm" else "abi"
    if set(contracts) != expected | {field} or not isinstance(contracts[field], dict):
        raise ValueError("ASK code generation result does not match its target contract")
    if target == "host-object":
        if ir or not isinstance(output_path, str) or not output_path:
            raise ValueError("ASK host object result requires an output path and empty IR")
    elif not ir.strip() or output_path:
        raise ValueError("ASK LLVM result requires IR text and no object output path")
    abi, artifact = contracts.get("abi"), contracts.get("cuda_artifact")
    if artifact is not None:
        from .cuda import _kernel_contract
        _kernel_contract(artifact)
        if artifact.get("format") != "llvm-ir" or artifact.get("payload") != ir:
            raise ValueError("ASK CUDA artifact payload differs from its LLVM IR")
    elif type(abi.get("schema_version")) is not int or abi["schema_version"] != 1:
        raise ValueError("unsupported ASK host ABI schema")
    return CodegenResult(**common, target=target, output_path=output_path, abi=abi,
                         cuda_artifact=artifact)
