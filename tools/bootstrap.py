#!/usr/bin/env python3
"""Install FugaTensor for development using an independently installed ASK SDK.

Pass --ask-prefix /path/to/ask-sdk. This script never builds ASK, installs an
ASK Python package, or reads a neighboring checkout. Fuga needs the ASK headers
and shared libraries, but no LLVM SDK. CUDA tools and compiler Python environments are
configured only when explicitly supplied. Supplying any tool or compiler path
replaces the entire saved GPU configuration; otherwise it is preserved.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import venv

try:
    import tomllib
except ImportError:
    tomllib = None


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def executable_path(value):
    # Resolving a venv Python symlink would select its base environment.
    selected = shutil.which(os.fspath(value))
    if selected is None:
        raise ValueError(f"executable not found: {value}")
    return Path(selected).absolute()


def compiler_overrides(values, selected=("triton",)):
    result = {}
    for item in values:
        name, separator, path = item.partition("=")
        if not separator or not path or name not in selected:
            raise ValueError("--compiler-python expects a bundled FugaTensor compiler ID followed by =/path/to/python")
        if name in result:
            raise ValueError(f"compiler Python was provided twice: {name}")
        result[name] = executable_path(path)
    return result


def build_requirements(root):
    if tomllib is None:
        raise RuntimeError("bootstrap requires Python 3.11 or newer to read build metadata")
    with (root / "pyproject.toml").open("rb") as stream:
        requirements = tomllib.load(stream)["build-system"]["requires"]
    if not isinstance(requirements, list) or any(not isinstance(item, str) for item in requirements):
        raise ValueError("build-system.requires must be a list of package requirements")
    return requirements


def missing_build_tools(environment):
    requirements = []
    cmake = shutil.which("cmake", path=environment.get("PATH"))
    if cmake is None:
        requirements.append("cmake>=3.28")
    else:
        result = subprocess.run([cmake, "--version"], text=True, capture_output=True,
                                check=True, env=environment)
        version = re.search(r"cmake version (\d+)\.(\d+)", result.stdout)
        if version is None or tuple(map(int, version.groups())) < (3, 28):
            requirements.append("cmake>=3.28")
    if shutil.which("ninja", path=environment.get("PATH")) is None:
        requirements.append("ninja")
    return requirements


def inspect_ask(prefix):
    """Check the installed SDK before touching the development environment.

    CMake validates FugaTensor's exact required SDK version and imported library
    files. This probe verifies that the SDK's CLI and package metadata agree.
    """
    prefix = Path(prefix).expanduser().resolve()
    cmake = next((prefix / name / "cmake/ASK" for name in ("lib", "lib64")
                  if (prefix / name / "cmake/ASK/ASKConfig.cmake").is_file()), None)
    headers = ("include/ask/Frontend/CompilerAPI.h", "include/ask/Runtime/CUDA.h")
    if cmake is None or any(not (prefix / header).is_file() for header in headers):
        raise ValueError("--ask-prefix must name an installed ASK SDK with CMake configuration and public headers")
    version_file = cmake / "ASKConfigVersion.cmake"
    version = (re.search(r'set\(PACKAGE_VERSION\s+"([^"]+)"\)', version_file.read_text(encoding="utf-8"))
               if version_file.is_file() else None)
    if version is None:
        raise ValueError("ASK SDK has missing or invalid CMake version metadata")
    executable = executable_path(prefix / "bin" / ("ask-opt.exe" if os.name == "nt" else "ask-opt"))
    result = subprocess.run([str(executable), "--capabilities"], capture_output=True, text=True, check=True)
    capabilities = json.loads(result.stdout)
    if not isinstance(capabilities, dict) or capabilities.get("ask_version") != version.group(1):
        raise RuntimeError("ASK SDK CMake metadata and compiler versions differ")
    if not capabilities.get("host_codegen") or not capabilities.get("cuda_graph_codegen"):
        raise RuntimeError("ASK SDK lacks native CPU/CUDA graph compilation")
    return {"prefix": str(prefix), "cmake_dir": str(cmake),
            "version": version.group(1), "capabilities": capabilities}


def engine_config(llvm_root, ptxas, environments, worker=None):
    compilers = {}
    for compiler_id, python_path in sorted(environments.items()):
        if compiler_id != "triton":
            raise ValueError(f"no bundled FugaTensor worker for {compiler_id!r}")
        selected_worker = Path(worker) if worker is not None else ROOT / "src/fugatensor/_triton_worker.py"
        if not selected_worker.is_file():
            raise ValueError(f"FugaTensor worker is missing: {selected_worker}")
        compilers[compiler_id] = {"python": str(python_path), "worker": str(selected_worker)}
    return {
        "schema_version": 1,
        "llc": str(llvm_root / "bin/llc") if llvm_root is not None else None,
        "ptxas": str(ptxas) if ptxas is not None else None,
        "compilers": compilers,
    }


def write_config(venv_root, configuration):
    destination = venv_root / "share/fugatensor/engine.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                         prefix="engine-", suffix=".json", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(configuration, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def read_existing_config(path):
    configuration = read_json(path)
    if (not isinstance(configuration, dict) or type(configuration.get("schema_version")) is not int
            or configuration["schema_version"] != 1):
        raise ValueError("existing engine configuration is invalid; use --reset-config to replace it")
    for key in ("llc", "ptxas"):
        value = configuration.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"existing {key} configuration is invalid; use --reset-config to replace it")
    compilers = configuration.get("compilers", {})
    if not isinstance(compilers, dict):
        raise ValueError("existing compiler configuration is invalid; use --reset-config to replace it")
    for compiler_id, fields in compilers.items():
        if (not isinstance(fields, dict) or set(fields) != {"python", "worker"}
                or any(not isinstance(value, str) or not value for value in fields.values())):
            raise ValueError(f"existing compiler {compiler_id} configuration is invalid; use --reset-config")
        if not Path(fields["worker"]).is_file():
            raise ValueError(f"existing compiler {compiler_id} worker is missing; "
                             "use --reset-config or configure its replacement explicitly")
    return configuration


def run(command, *, env=None):
    print("+ " + shlex.join([str(item) for item in command]), flush=True)
    subprocess.run([str(item) for item in command], check=True, env=env)


def python_json(python, program, *arguments):
    result = subprocess.run([str(python), "-I", "-c", program, *map(str, arguments)],
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"cannot inspect Python environment {python}: "
                           + (result.stderr.strip() or f"exit status {result.returncode}"))
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Python environment returned invalid metadata: {result.stdout}") from error


def installed_version(python, package):
    return python_json(
        python,
        "import importlib.metadata,json,sys\n"
        "try:\n v = importlib.metadata.version(sys.argv[1])\n"
        "except importlib.metadata.PackageNotFoundError:\n v = None\n"
        "print(json.dumps(v))",
        package,
    )


def unsatisfied_requirements(python, requirements):
    pending = []
    for req in requirements:
        if "==" in req:
            pkg, expected = req.split("==", 1)
            if installed_version(python, pkg.strip()) == expected.strip():
                continue
        elif req == "pip" and installed_version(python, "pip") is not None:
            continue
        pending.append(req)
    return pending


def ensure_venv(directory):
    if directory.exists() and not (directory / "pyvenv.cfg").is_file():
        raise ValueError(f"--venv exists but is not a Python virtual environment: {directory}")
    if not directory.exists():
        print(f"Creating isolated environment: {directory}", flush=True)
        venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(directory)
    python = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    metadata = python_json(python, "import json,sys; print(json.dumps({'prefix':sys.prefix,'base':sys.base_prefix}))")
    if Path(metadata["prefix"]).resolve() != directory or metadata["prefix"] == metadata["base"]:
        raise ValueError("bootstrap refuses to install into a non-isolated Python interpreter")
    configuration = (directory / "pyvenv.cfg").read_text(encoding="utf-8").lower()
    if re.search(r"include-system-site-packages\s*=\s*true", configuration):
        raise ValueError("bootstrap requires include-system-site-packages = false")
    return python


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--ask-prefix", type=Path, required=True, help="prefix of an independently installed ASK C++ SDK")
    result.add_argument("--venv", type=Path, default=ROOT / ".venv")
    result.add_argument("--compiler-python", action="append", default=[], metavar="ID=PYTHON",
                        help="configure a bundled FugaTensor worker with an existing compiler environment")
    result.add_argument("--llvm-root", type=Path, help="optional CUDA codegen SDK matching the ASK compiler")
    result.add_argument("--ptxas", help="optional CUDA assembler executable")
    result.add_argument("--reset-config", action="store_true",
                        help="replace saved GPU configuration; otherwise reuse it when no paths are supplied")
    result.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    return result


def bootstrap(args):
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    installed = inspect_ask(args.ask_prefix)
    capabilities = installed["capabilities"]
    llvm_root = args.llvm_root.expanduser().resolve() if args.llvm_root is not None else None
    if llvm_root is not None:
        executable_path(llvm_root / "bin/llc")
    ptxas = executable_path(args.ptxas) if args.ptxas is not None else None
    environment_root = args.venv.expanduser().resolve()
    python = ensure_venv(environment_root)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PATH"] = str(python.parent) + os.pathsep + environment.get("PATH", "")
    environment["CMAKE_BUILD_PARALLEL_LEVEL"] = str(args.jobs)
    environment["CMAKE_PREFIX_PATH"] = os.pathsep.join(filter(None, (
        installed["prefix"], environment.get("CMAKE_PREFIX_PATH"))))
    requirements = list(dict.fromkeys(["pip", *build_requirements(ROOT)]))
    requirements.extend(missing_build_tools(environment))
    pip_ready = subprocess.run([str(python), "-m", "pip", "--version"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               env=environment).returncode == 0
    pending_reqs = requirements if not pip_ready else unsatisfied_requirements(python, requirements)
    if pending_reqs:
        install = ([str(python), "-m", "pip"] if pip_ready else
                   [sys.executable, "-m", "pip", "--python", str(python)])
        run([*install, "install", *pending_reqs], env=environment)
    if llvm_root is not None:
        header = llvm_root / "include/llvm/Support/VCSRevision.h"
        revision = capabilities.get("llvm_revision")
        if not revision or not header.is_file() or revision not in header.read_text(encoding="utf-8"):
            raise ValueError("--llvm-root must match the LLVM revision of the ASK SDK")
    environments = compiler_overrides(args.compiler_python)
    config_path = environment_root / "share/fugatensor/engine.json"
    preserve_config = (config_path.is_file() and not args.reset_config and llvm_root is None
                       and ptxas is None and not args.compiler_python)
    configuration = (read_existing_config(config_path) if preserve_config else
                     engine_config(llvm_root, ptxas, environments))
    for compiler_id, fields in configuration.get("compilers", {}).items():
        compiler_python = fields["python"]
        if compiler_id != "triton":
            continue  # Other workers are configured explicitly through Engine/config JSON.
        dependency = read_json(ROOT / "src/fugatensor/triton.dependencies.lock.json")["compiler"]
        package, version = dependency["name"], dependency["version"]
        actual = installed_version(compiler_python, package)
        if actual != version:
            raise ValueError(f"compiler {compiler_id} needs {package}=={version}; selected Python has {actual!r}. "
                             "Install that dependency in the compiler environment separately.")
    # An explicit SDK selection must also replace a cached ASK_DIR from a prior build.
    run([python, "-m", "pip", "install", "--no-deps", "--no-build-isolation",
         "--config-settings=cmake.define.ASK_DIR=" + installed["cmake_dir"], "--editable", ROOT],
        env=environment)
    run([python, "-m", "pip", "check"], env=environment)
    if not preserve_config:
        config_path = write_config(environment_root, configuration)
    summary = {
        "python": str(python),
        "engine_config": str(config_path),
        "ask_version": installed["version"],
        "ask_prefix": installed["prefix"],
        "configured_compilers": sorted(configuration.get("compilers", {})),
        "llc": configuration.get("llc"),
        "ptxas": configuration.get("ptxas"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Ready: {shlex.quote(str(python))} {shlex.quote(str(ROOT / 'examples/jit_demo.py'))} --device cpu",
          flush=True)
    return summary


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        bootstrap(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"bootstrap failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
