#!/usr/bin/env python3
"""Install FugaTensor for development using an independently built ASK package.

Pass --ask-wheel /path/to/ask_compiler.whl, or install the exact ASK dependency
into --venv first. This script never builds ASK or reads a neighboring checkout.
CPU setup needs no LLVM SDK. CUDA tools and compiler Python environments are
configured only when explicitly supplied. Supplying any tool or compiler path
replaces the entire saved GPU configuration; otherwise it is preserved.
"""
from __future__ import annotations

import argparse
from email.parser import BytesParser
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
import zipfile

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


def compiler_overrides(values, selected):
    result = {}
    for item in values:
        name, separator, path = item.partition("=")
        if not separator or not path or name not in selected:
            raise ValueError("--compiler-python expects an installed ASK compiler ID followed by =/path/to/python")
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


def ask_requirement(root=None):
    root = ROOT if root is None else root
    if tomllib is None:
        raise RuntimeError("bootstrap requires Python 3.11 or newer to read project metadata")
    with (root / "pyproject.toml").open("rb") as stream:
        dependencies = tomllib.load(stream)["project"]["dependencies"]
    matches = [re.fullmatch(r"ask-compiler==([0-9]+\.[0-9]+\.[0-9]+)", item)
               for item in dependencies]
    versions = [match.group(1) for match in matches if match is not None]
    if len(versions) != 1:
        raise ValueError("FugaTensor must declare one exact ask-compiler version in pyproject.toml")
    return versions[0]


def require_ask_version(python, expected):
    actual = installed_version(python, "ask-compiler")
    if actual != expected:
        raise ValueError(f"FugaTensor requires ask-compiler=={expected}; selected environment has {actual!r}. "
                         "Build ASK separately in its own project, then pass --ask-wheel PATH "
                         "or install that wheel into --venv first.")


def check_ask_wheel(wheel, expected):
    with zipfile.ZipFile(wheel) as archive:
        records = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(records) != 1:
            raise ValueError("--ask-wheel must contain exactly one distribution metadata record")
        metadata = BytesParser().parsebytes(archive.read(records[0]))
    name = re.sub(r"[-_.]+", "-", metadata.get("Name", "")).lower()
    if name != "ask-compiler" or metadata.get("Version") != expected:
        raise ValueError(f"--ask-wheel must contain ask-compiler=={expected}")


def inspect_ask(python, expected):
    require_ask_version(python, expected)
    installed = python_json(
        python,
        "import json,sys,ask; from pathlib import Path; from ask import Compiler; "
        "from ask.adapters import AdapterRegistry; "
        "c=Compiler(); c._check_companion(); caps=c.capabilities(); "
        "registry=AdapterRegistry(caps); package=Path(ask.__file__).resolve().parent; "
        "assert package.is_relative_to(Path(sys.prefix).resolve()), 'ASK must be installed in the target venv'; "
        "print(json.dumps({'capabilities':caps,'version':ask.__version__,'package':str(package),"
        "'registrations':{name:registry.registration(name) for name in caps['third_party_compilers']}}))",
    )
    capabilities = installed["capabilities"]
    if installed["version"] != expected or capabilities.get("ask_version") != expected:
        raise RuntimeError("installed ASK metadata, Python package and native compiler versions differ")
    if not capabilities.get("host_codegen") or not capabilities.get("cuda_graph_codegen"):
        raise RuntimeError("installed ASK engine lacks native CPU/CUDA graph compilation")
    return installed


def engine_config(llvm_root, ptxas, environments, package_root, registrations):
    compilers = {}
    for compiler_id, python_path in sorted(environments.items()):
        worker_name = registrations[compiler_id].get("worker")
        if not isinstance(worker_name, str) or Path(worker_name).name != worker_name:
            raise ValueError(f"compiler {compiler_id} requires a simple registered worker filename")
        worker = package_root / "share/ask/adapters" / compiler_id / worker_name
        if not worker.is_file():
            raise ValueError(f"installed ASK worker is missing: {worker}")
        compilers[compiler_id] = {"python": str(python_path), "worker": str(worker)}
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


def read_existing_config(path, package_root, registrations):
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
        if compiler_id not in registrations:
            raise ValueError(f"existing configuration uses compiler {compiler_id!r}, which installed ASK "
                             "does not enable; use --reset-config or configure its replacement explicitly")
        if (not isinstance(fields, dict) or set(fields) != {"python", "worker"}
                or any(not isinstance(value, str) or not value for value in fields.values())):
            raise ValueError(f"existing compiler {compiler_id} configuration is invalid; use --reset-config")
        expected = engine_config(None, None, {compiler_id: fields["python"]},
                                 package_root, registrations)["compilers"][compiler_id]["worker"]
        if Path(fields["worker"]).resolve() != Path(expected).resolve():
            raise ValueError(f"existing compiler {compiler_id} worker differs from installed ASK; "
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
    result.add_argument("--ask-wheel", type=Path, help="install an independently built local ASK wheel")
    result.add_argument("--venv", type=Path, default=ROOT / ".venv")
    result.add_argument("--compiler-python", action="append", default=[], metavar="ID=PYTHON",
                        help="configure an installed ASK adapter with an existing compiler environment")
    result.add_argument("--llvm-root", type=Path, help="optional CUDA codegen SDK matching installed ASK")
    result.add_argument("--ptxas", help="optional CUDA assembler executable")
    result.add_argument("--reset-config", action="store_true",
                        help="replace saved GPU configuration; otherwise reuse it when no paths are supplied")
    result.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    # Diagnose old combined-build commands instead of silently reinterpreting them.
    result.add_argument("--ask-root", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--compilers", help=argparse.SUPPRESS)
    result.add_argument("--skip-ask", action="store_true", help=argparse.SUPPRESS)
    return result


def bootstrap(args):
    if args.ask_root is not None or args.compilers is not None or args.skip_ask:
        raise ValueError("--ask-root, --compilers and --skip-ask belonged to the removed combined build. "
                         "Build ASK and select its adapters in the ASK project separately; "
                         "use --ask-wheel PATH or an ASK wheel already installed in --venv.")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    expected_version = ask_requirement()
    wheel = args.ask_wheel.expanduser().resolve() if args.ask_wheel is not None else None
    if wheel is not None and (wheel.suffix != ".whl" or not wheel.is_file()):
        raise ValueError("--ask-wheel must name an existing local .whl file; build ASK separately first")
    if wheel is not None:
        check_ask_wheel(wheel, expected_version)
    llvm_root = args.llvm_root.expanduser().resolve() if args.llvm_root is not None else None
    if llvm_root is not None:
        executable_path(llvm_root / "bin/llc")
    ptxas = executable_path(args.ptxas) if args.ptxas is not None else None
    environment_root = args.venv.expanduser().resolve()
    python = ensure_venv(environment_root)
    if wheel is None:
        require_ask_version(python, expected_version)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PATH"] = str(python.parent) + os.pathsep + environment.get("PATH", "")
    environment["CMAKE_BUILD_PARALLEL_LEVEL"] = str(args.jobs)
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
    if wheel is not None:
        run([python, "-m", "pip", "install", "--no-deps", "--force-reinstall", wheel], env=environment)
    installed = inspect_ask(python, expected_version)
    capabilities = installed["capabilities"]
    if llvm_root is not None:
        header = llvm_root / "include/llvm/Support/VCSRevision.h"
        revision = capabilities.get("llvm_revision")
        if not revision or not header.is_file() or revision not in header.read_text(encoding="utf-8"):
            raise ValueError("--llvm-root must match the LLVM revision of the installed ASK package")
    registrations = installed["registrations"]
    environments = compiler_overrides(args.compiler_python, registrations)
    package_root = Path(installed["package"])
    config_path = environment_root / "share/fugatensor/engine.json"
    preserve_config = (config_path.is_file() and not args.reset_config and llvm_root is None
                       and ptxas is None and not args.compiler_python)
    configuration = (read_existing_config(config_path, package_root, registrations) if preserve_config else
                     engine_config(llvm_root, ptxas, environments, package_root, registrations))
    for compiler_id, fields in configuration.get("compilers", {}).items():
        compiler_python = fields["python"]
        if registrations[compiler_id].get("requires_python") is not True:
            raise ValueError(f"compiler {compiler_id} does not register a Python worker")
        worker = Path(fields["worker"])
        dependency = read_json(worker.parent / "dependencies.lock.json")["compiler"]
        package, version = dependency["name"], dependency["version"]
        actual = installed_version(compiler_python, package)
        if actual != version:
            raise ValueError(f"compiler {compiler_id} needs {package}=={version}; selected Python has {actual!r}. "
                             "Install that dependency in the compiler environment separately.")
    run([python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--editable", ROOT],
        env=environment)
    run([python, "-m", "pip", "check"], env=environment)
    if not preserve_config:
        config_path = write_config(environment_root, configuration)
    summary = {
        "python": str(python),
        "engine_config": str(config_path),
        "ask_version": expected_version,
        "third_party_compilers": capabilities["third_party_compilers"],
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
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        print(f"bootstrap failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
