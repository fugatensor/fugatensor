"""Check installation selection and persisted paths; real builds run separately."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("fuga_bootstrap", Path(__file__).resolve().parents[1] / "tools/bootstrap.py")
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


class BootstrapTests(unittest.TestCase):
    def test_default_is_all_and_selection_does_not_guess_compiler_directories(self):
        args = bootstrap.parser().parse_args([])
        self.assertEqual(args.compilers, "all")
        self.assertEqual(bootstrap.select_compilers("all", {"other": "other_dir", "triton": "triton"}),
                         ("other", "triton"))
        self.assertEqual(bootstrap.select_compilers("none", {"triton": "triton"}), ())
        self.assertEqual(bootstrap.select_compilers("triton", {"triton": "triton"}), ("triton",))
        for invalid in ("all;triton", "none;triton", "triton;triton", "missing", "triton;", "../triton"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                bootstrap.select_compilers(invalid, {"triton": "triton"})

    def test_wheel_command_keeps_spaces_and_semicolons_as_single_arguments(self):
        command = bootstrap.wheel_command(Path("/a venv/bin/python"), Path("/source ask"), Path("/sdk llvm"),
                                          Path("/wheel staging"), Path("/build area/all"), "triton;other",
                                          {"triton": Path("/worker env/bin/python")})
        self.assertEqual(command[0], "/a venv/bin/python")
        self.assertIn("/source ask", command)
        self.assertIn("-Ccmake.define.ASK_THIRD_PARTY_COMPILERS=triton;other", command)
        self.assertIn("-Ccmake.define.ASK_TRITON_PYTHON_EXECUTABLE=/worker env/bin/python", command)
        self.assertIn("-Cbuild-dir=/build area/all/{wheel_tag}", command)
        self.assertIn("-Ccmake.define.LLVM_DIR=/sdk llvm/lib/cmake/llvm", command)
        self.assertIn("--no-build-isolation", command)
        self.assertIn("--no-deps", command)

    @unittest.skipUnless(os.name == "posix", "executable symlinks need POSIX")
    def test_worker_python_symlink_keeps_its_virtual_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "base-python"
            executable.write_bytes(b"fixture")
            executable.chmod(0o700)
            link = root / "venv-python"
            link.symlink_to(executable)
            self.assertEqual(bootstrap.compiler_overrides([f"triton={link}"], ("triton",)), {"triton": link})
            with self.assertRaisesRegex(ValueError, "twice"):
                bootstrap.compiler_overrides([f"triton={link}", f"triton={link}"], ("triton",))
            with self.assertRaises(ValueError):
                bootstrap.compiler_overrides([f"other={link}"], ("triton",))

    def test_config_uses_installed_workers_and_no_source_ask_opt_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = root / "site/ask/share/ask/adapters/triton/exporter.py"
            worker.parent.mkdir(parents=True)
            worker.write_text("# installed fixture\n")
            config = bootstrap.engine_config(root / "sdk", root / "ptxas",
                {"triton": root / "worker-env/bin/python"}, root / "site/ask",
                {"triton": {"worker": "exporter.py"}}, "all")
            self.assertNotIn("ask_opt", config)
            self.assertEqual(config["compilers"]["triton"]["worker"], str(worker))
            self.assertEqual(config["engine_selection"], "all")
            target = bootstrap.write_config(root / "venv", config)
            self.assertEqual(target, root / "venv/share/fugatensor/engine.json")
            self.assertEqual(json.loads(target.read_text()), config)
            self.assertEqual(list(target.parent.iterdir()), [target])
            worker.unlink()
            with self.assertRaisesRegex(ValueError, "worker is missing"):
                bootstrap.engine_config(root / "sdk", None, {"triton": root / "python"},
                                        root / "site/ask", {"triton": {"worker": "exporter.py"}}, "all")

    def test_none_configuration_does_not_require_any_worker_or_assembler(self):
        config = bootstrap.engine_config(Path("/sdk"), None, {}, Path("/site/ask"), {}, "none")
        self.assertEqual(config["compilers"], {})
        self.assertIsNone(config["ptxas"])
        self.assertEqual(config["llc"], "/sdk/bin/llc")

    @unittest.skipUnless(os.name == "posix", "executable test fixtures need POSIX")
    def test_old_build_tools_are_upgraded_inside_the_local_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(bootstrap.missing_build_tools({"PATH": str(root)}), ["cmake>=3.28", "ninja"])
            cmake = root / "cmake"
            cmake.write_text("#!/bin/sh\necho 'cmake version 3.20.0'\n")
            cmake.chmod(0o700)
            self.assertEqual(bootstrap.missing_build_tools({"PATH": str(root)}), ["cmake>=3.28", "ninja"])
            cmake.write_text("#!/bin/sh\necho 'cmake version 3.28.2'\n")
            self.assertEqual(bootstrap.missing_build_tools({"PATH": str(root)}), ["ninja"])


if __name__ == "__main__":
    unittest.main()
