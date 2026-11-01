#!/usr/bin/env python3
"""Build a local ASK engine and install FugaTensor into an isolated environment.

Run ``python3 tools/bootstrap.py`` from any directory. This FugaTensor setup
selects all registered ASK compiler adapters by default; it does not modify
ASK's own default build configuration or install into the launcher environment.
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
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]*\Z", re.ASCII)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compiler_catalog(ask_root):
    value = read_json(ask_root / "adapters/registry.json")
    if value.get("schema_version") != 1 or not isinstance(value.get("compilers"), dict):
        raise ValueError("unsupported ASK compiler registry")
    result = value["compilers"]
    if any(not _IDENTIFIER.fullmatch(name) or not isinstance(directory, str)
           or not _IDENTIFIER.fullmatch(directory) for name, directory in result.items()):
        raise ValueError("ASK compiler IDs and directories must be simple lowercase names")
    return result


def select_compilers(selection, catalog):
    if selection == "all":
        return tuple(sorted(catalog))
    if selection == "none":
        return ()
    names = selection.split(";")
    if any(not _IDENTIFIER.fullmatch(name) for name in names) or len(set(names)) != len(names):
        raise ValueError("--compilers expects all, none, or unique semicolon-separated compiler IDs")
    unknown = set(names) - set(catalog)
    if unknown:
        raise ValueError("ASK does not register these compilers: " + ", ".join(sorted(unknown)))
    return tuple(sorted(names))


def executable_path(value):
    # Resolving a venv Python symlink would select its base environment.
    selected = shutil.which(os.fspath(value))
    if selected is None:
        raise ValueError(f"executable not found: {value}")
    return Path(selected).absolute()


def compiler_overrides(values, selected):
    result = {}
    for item in values:
        name, separator, path = item.partition("=")
        if not separator or not path or name not in selected:
            raise ValueError("--compiler-python expects SELECTED_ID=/path/to/python")
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
        result = subprocess.run([cmake, "--version"], text=True, capture_output=True, check=True)
        version = re.search(r"cmake version (\d+)\.(\d+)", result.stdout)
        if version is None or tuple(map(int, version.groups())) < (3, 28):
            requirements.append("cmake>=3.28")
    if shutil.which("ninja", path=environment.get("PATH")) is None:
        requirements.append("ninja")
    return requirements


def wheel_command(python, ask_root, llvm_root, wheel_dir, build_dir, selection, environments):
    command = [str(python), "-m", "pip", "wheel", str(ask_root), "--no-deps", "--no-build-isolation",
               "--wheel-dir", str(wheel_dir),
               "-Ccmake.define.ASK_THIRD_PARTY_COMPILERS=" + selection,
               "-Ccmake.define.ASK_TARGET_BACKENDS=host",
               "-Ccmake.define.LLVM_DIR=" + str(llvm_root / "lib/cmake/llvm"),
               "-Ccmake.define.MLIR_DIR=" + str(llvm_root / "lib/cmake/mlir"),
               "-Cbuild-dir=" + str(build_dir / "{wheel_tag}")]
    for compiler_id, python_path in sorted(environments.items()):
        command.append(f"-Ccmake.define.ASK_{compiler_id.upper()}_PYTHON_EXECUTABLE={python_path}")
    return command


def engine_config(llvm_root, ptxas, environments, package_root, registrations, selection):
    compilers = {}
    for compiler_id, python_path in sorted(environments.items()):
        worker_name = registrations[compiler_id].get("worker")
        if not isinstance(worker_name, str) or Path(worker_name).name != worker_name:
            raise ValueError(f"compiler {compiler_id} requires a simple registered worker filename")
        worker = package_root / "share/ask/adapters" / compiler_id / worker_name
        if not worker.is_file():
            raise ValueError(f"installed ASK worker is missing: {worker}")
        compilers[compiler_id] = {"python": str(python_path), "worker": str(worker)}
    return {"schema_version": 1, "engine_selection": selection,
            "llc": str(llvm_root / "bin/llc"),
            "ptxas": str(ptxas) if ptxas is not None else None,
            "compilers": compilers}


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
    return python_json(python, "import importlib.metadata,json,sys; print(json.dumps(importlib.metadata.version(sys.argv[1])))",
                       package)


def discover_ptxas(explicit, packages, *, fallback_python=None, env=None):
    if explicit is not None:
        return executable_path(explicit)
    found = shutil.which("ptxas", path=(env or os.environ).get("PATH"))
    if found:
        return Path(found).absolute()
    # Inspect wheel file metadata; do not import a compiler or initialize CUDA.
    program = ("import importlib.metadata,json,sys; from pathlib import Path; "
               "d=importlib.metadata.distribution(sys.argv[1]); "
               "print(json.dumps([str(Path(d.locate_file(p)).absolute()) for p in (d.files or ()) "
               "if p.name == 'ptxas' and Path(d.locate_file(p)).is_file()]))")
    for python, package in packages:
        for candidate in python_json(python, program, package):
            if os.access(candidate, os.X_OK):
                return Path(candidate)
    if fallback_python is not None:
        # A none build may reuse an assembler already installed in the venv;
        # this does not enable or import the compiler that shipped that file.
        program = ("import importlib.metadata,json; from pathlib import Path; "
                   "print(json.dumps([str(Path(d.locate_file(p)).absolute()) "
                   "for d in importlib.metadata.distributions() for p in (d.files or ()) "
                   "if p.name == 'ptxas' and Path(d.locate_file(p)).is_file()]))")
        for candidate in python_json(fallback_python, program):
            if os.access(candidate, os.X_OK):
                return Path(candidate)
    return None


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
    result.add_argument("--ask-root", type=Path, default=ROOT.parent / "ask")
    result.add_argument("--venv", type=Path, default=ROOT / ".venv")
    result.add_argument("--compilers", default="all", help="all (default), none, or quoted semicolon-separated IDs")
    result.add_argument("--compiler-python", action="append", default=[], metavar="ID=PYTHON",
                        help="reuse an existing compiler environment instead of installing its pinned package")
    result.add_argument("--llvm-root", type=Path, help="existing SDK matching ASK's LLVM lock")
    result.add_argument("--ptxas", help="CUDA assembler; otherwise discover PATH or installed compiler wheel files")
    result.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    return result


def bootstrap(args):
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    ask_root, environment_root = args.ask_root.expanduser().resolve(), args.venv.expanduser().resolve()
    catalog = compiler_catalog(ask_root)
    selected = select_compilers(args.compilers, catalog)
    overrides = compiler_overrides(args.compiler_python, selected)
    if args.ptxas is not None:
        executable_path(args.ptxas)
    llvm_lock = read_json(ask_root / "cmake/llvm.lock.json")
    llvm_root = (args.llvm_root.expanduser().resolve() if args.llvm_root is not None
                 else ask_root / ".deps" / llvm_lock["directory"])
    if args.llvm_root is None:
        # The fetcher validates an existing receipt without re-downloading it.
        run([sys.executable, ask_root / "tools/fetch_llvm.py"])
    elif not llvm_root.is_dir():
        raise ValueError(f"explicit LLVM SDK does not exist: {llvm_root}")
    for relative in ("bin/llc", "lib/cmake/llvm/LLVMConfig.cmake", "lib/cmake/mlir/MLIRConfig.cmake"):
        if not (llvm_root / relative).is_file():
            raise ValueError(f"LLVM SDK is missing {relative}")
    run([sys.executable, ask_root / "tools/fetch_nanobind.py"])

    python = ensure_venv(environment_root)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PATH"] = str(python.parent) + os.pathsep + environment.get("PATH", "")
    environment["CMAKE_BUILD_PARALLEL_LEVEL"] = str(args.jobs)
    requirements = list(dict.fromkeys(["pip", *build_requirements(ask_root), *build_requirements(ROOT)]))
    # Install build tools locally too when the launcher lacks them.
    requirements.extend(missing_build_tools(environment))
    pip_ready = subprocess.run([str(python), "-m", "pip", "--version"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    install = ([str(python), "-m", "pip"] if pip_ready else
               [sys.executable, "-m", "pip", "--python", str(python)])
    run([*install, "install", *requirements], env=environment)

    environments, packages, registrations = {}, [], {}
    for compiler_id in selected:
        directory = ask_root / "adapters" / catalog[compiler_id]
        registration = read_json(directory / "registration.json")
        registrations[compiler_id] = registration
        if registration.get("requires_python") is not True:
            raise ValueError(f"bootstrap needs an explicit installer for non-Python compiler {compiler_id}")
        lock = read_json(directory / "dependencies.lock.json")["compiler"]
        package, version = lock["name"], lock["version"]
        compiler_python = overrides.get(compiler_id, python)
        if compiler_id not in overrides:
            run([python, "-m", "pip", "install", package + "==" + version], env=environment)
        actual = installed_version(compiler_python, package)
        if actual != version:
            raise ValueError(f"compiler {compiler_id} needs {package}=={version}; selected Python has {actual}")
        environments[compiler_id] = compiler_python
        packages.append((compiler_python, package))

    mode = args.compilers if args.compilers in ("all", "none") else "selected-" + "-".join(selected)
    wheel_parent = ROOT / "build/bootstrap-wheels"
    wheel_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=mode + "-", dir=wheel_parent) as directory:
        staging = Path(directory)
        run(wheel_command(python, ask_root, llvm_root, staging, ROOT / "build/ask" / mode,
                          args.compilers, environments), env=environment)
        wheels = list(staging.glob("ask_compiler-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("ASK build did not produce exactly one wheel")
        run([python, "-m", "pip", "install", "--no-deps", "--force-reinstall", wheels[0]], env=environment)
    run([python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--editable", ROOT],
        env=environment)
    installed = python_json(python,
        "import json,ask; from pathlib import Path; from ask import Compiler; "
        "c=Compiler(); c._check_companion(); "
        "print(json.dumps({'capabilities':c.capabilities(),'package':str(Path(ask.__file__).resolve().parent)}))")
    capabilities = installed["capabilities"]
    if sorted(capabilities.get("third_party_compilers", [])) != list(selected):
        raise RuntimeError("installed ASK compiler selection differs from the requested build")
    if not capabilities.get("host_codegen") or not capabilities.get("cuda_graph_codegen"):
        raise RuntimeError("installed ASK engine lacks native CPU/CUDA graph compilation")
    ptxas = discover_ptxas(args.ptxas, packages, fallback_python=python, env=environment)
    configuration = engine_config(llvm_root, ptxas, environments, Path(installed["package"]),
                                  registrations, args.compilers)
    config_path = write_config(environment_root, configuration)
    summary = {"python": str(python), "engine_config": str(config_path),
               "third_party_compilers": list(selected), "llc": configuration["llc"], "ptxas": configuration["ptxas"]}
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
