"""Saved build defaults apply once at setup; explicit installations stay isolated."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fugatensor._config import runtime_defaults
from fugatensor.engine import Engine, configure, shutdown


class EngineConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'engine.json'
        self.path.write_text(json.dumps({
            'schema_version': 1, 'engine_selection': 'all',
            'llc': '/saved/llc', 'ptxas': '/saved/ptxas',
            'compilers': {'triton': {'python': '/saved/python', 'worker': '/saved/worker.py'}},
        }))
        environment = patch.dict(os.environ, {'FUGATENSOR_CONFIG': str(self.path)}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        self.addCleanup(shutdown)

    def test_saved_paths_need_no_environment_exports_or_gpu_probe(self):
        with patch('fugatensor.engine.Compiler'), patch('fugatensor.engine.ExecutionSession') as session:
            with Engine(device='cuda') as engine:
                self.assertEqual((engine.llc, engine.ptxas), ('/saved/llc', '/saved/ptxas'))
                self.assertEqual(engine.compilers['triton'].worker, '/saved/worker.py')
                session.assert_not_called()
        self.assertEqual(runtime_defaults('cpu'), {})

    def test_environment_and_explicit_options_override_saved_values(self):
        with patch('fugatensor.engine.Compiler'), patch.dict(os.environ, {'FUGATENSOR_LLC': '/env/llc'}):
            with Engine(device='cuda', ptxas='/explicit/ptxas', compilers={}) as engine:
                self.assertEqual(engine.llc, '/env/llc')
                self.assertEqual(engine.ptxas, '/explicit/ptxas')
                self.assertEqual(engine.compilers, {})
            engine = configure(device='cuda', llc='/configured/llc', compilers={})
            self.assertEqual(engine.llc, '/configured/llc')
            self.assertEqual(engine.ptxas, '/saved/ptxas')

    def test_explicit_ask_installation_does_not_inherit_saved_workers(self):
        self.path.write_text('broken stale configuration')
        with patch('fugatensor.engine.Compiler') as compiler:
            with Engine(device='cuda', ask_opt='/other/ask-opt') as engine:
                self.assertIsNone(engine.llc)
                self.assertEqual(engine.compilers, {})
                compiler.assert_called_with(executable='/other/ask-opt')
            engine = configure(device='cuda', ask_opt='/other/ask-opt', ptxas='/other/ptxas')
            self.assertIsNone(engine.llc)
            self.assertEqual(engine.compilers, {})

    def test_environment_ask_installation_also_overrides_saved_workers(self):
        self.path.write_text('stale configuration from another installation')
        with patch('fugatensor.engine.Compiler') as compiler, \
             patch.dict(os.environ, {'ASK_OPT': '/environment/ask-opt'}):
            with Engine(device='cuda') as engine:
                self.assertIsNone(engine.llc)
                self.assertEqual(engine.compilers, {})
                compiler.assert_called_with(executable='/environment/ask-opt')

    def test_invalid_configuration_reports_the_problem(self):
        for content in ('invalid json', json.dumps({'schema_version': True}),
                        json.dumps({'schema_version': 1, 'llc': []}),
                        json.dumps({'schema_version': 1, 'compilers': []})):
            self.path.write_text(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                runtime_defaults('cuda')
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, 'does not name'):
            runtime_defaults('cuda')


if __name__ == '__main__':
    unittest.main()
